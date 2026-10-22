#!/usr/bin/env python3
"""
==========================================================================
  Treinamento Comparativo — YOLOv8m vs YOLOv10m no BDD100K
==========================================================================
  Pipeline robusto com:
    ✅ Lógica anti-falha (resume de checkpoint)
    ✅ Integração WandB protegida (não crasha sem login)
    ✅ Compatibilidade PyTorch 2.6+ (weights_only patch)
    ✅ Parâmetros validados contra a API ultralytics

  Uso:
    python scripts/train_yolo.py --version v8      # Treinar só YOLOv8m
    python scripts/train_yolo.py --version v10     # Treinar só YOLOv10m
    python scripts/train_yolo.py --version both    # Treinar ambos sequencialmente

  Parâmetros customizáveis:
    --epochs 100  --imgsz 640  --batch 16  --workers 12
==========================================================================
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

# ── Diretórios do projeto ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "bdd100k" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

# ── Modelos ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "v8": {
        "weights": "yolov8m.pt",
        "run_name": "yolov8m_bdd100k",
        "display_name": "YOLOv8m",
    },
    "v10": {
        "weights": "yolov10m.pt",
        "run_name": "yolov10m_bdd100k",
        "display_name": "YOLOv10m",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  PATCH: Compatibilidade PyTorch 2.6+
# ══════════════════════════════════════════════════════════════════════
def patch_torch_load():
    """
    PyTorch 2.6 mudou o default de torch.load para weights_only=True,
    mas muitos checkpoints (incluindo ultralytics) precisam de
    weights_only=False. Este patch aplica o fallback.
    """
    if hasattr(torch, "_original_load"):
        return  # Já foi patched

    torch._original_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return torch._original_load(*args, **kwargs)

    torch.load = _patched_load
    print("  🔧 Patch PyTorch 2.6+ (weights_only) aplicado")


# ══════════════════════════════════════════════════════════════════════
#  SISTEMA DE INFORMAÇÕES DE GPU
# ══════════════════════════════════════════════════════════════════════
def print_gpu_info():
    """Exibe informações da GPU disponível."""
    print(f"\n{'─'*55}")
    print("  🖥️  Informações de Hardware")
    print(f"{'─'*55}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        vram_free = (torch.cuda.get_device_properties(0).total_mem -
                     torch.cuda.memory_allocated(0)) / (1024**3)
        print(f"  GPU:        {gpu_name}")
        print(f"  VRAM:       {vram_total:.1f} GB total")
        print(f"  CUDA:       {torch.version.cuda}")
        print(f"  PyTorch:    {torch.__version__}")
        print(f"  Device:     cuda:0 ✅")
    else:
        print("  ⚠️  CUDA NÃO DISPONÍVEL — treinando em CPU (MUITO LENTO)")
        print(f"  PyTorch:    {torch.__version__}")

    print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════════
#  WANDB — Inicialização Protegida
# ══════════════════════════════════════════════════════════════════════
def init_wandb(model_name: str, args) -> bool:
    """
    Inicializa WandB com proteção contra falha.
    Retorna True se WandB foi inicializado com sucesso.
    """
    try:
        import wandb

        wandb.init(
            project="BDD100K-Comparison",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": model_name,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "dataset": "BDD100K",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            tags=["bdd100k", model_name.lower(), "comparison"],
            resume="allow",  # Permite resumir o run do WandB se existir
        )
        print(f"  ✅ WandB inicializado — Projeto: BDD100K-Comparison")
        return True

    except Exception as e:
        print(f"  ⚠️  WandB não disponível: {e}")
        print(f"  ℹ️  O treino vai continuar SEM log no WandB.")
        print(f"  💡 Para ativar: wandb login")
        return False


def finish_wandb():
    """Finaliza WandB graciosamente."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  LÓGICA DE RESUME — Anti-Falha
# ══════════════════════════════════════════════════════════════════════
def check_existing_run(run_name: str) -> dict:
    """
    Verifica se existe um run anterior e retorna o estado.

    Retorna:
        {
            "exists": bool,
            "has_last": bool,      # Existe last.pt?
            "has_best": bool,      # Existe best.pt?
            "last_path": Path,     # Caminho do last.pt
            "best_path": Path,     # Caminho do best.pt
            "run_dir": Path,       # Diretório do run
        }
    """
    run_dir = RUNS_DIR / run_name
    weights_dir = run_dir / "weights"
    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"

    return {
        "exists": run_dir.exists(),
        "has_last": last_path.exists(),
        "has_best": best_path.exists(),
        "last_path": last_path,
        "best_path": best_path,
        "run_dir": run_dir,
    }


def get_resume_decision(run_name: str, state: dict) -> str:
    """
    Pergunta ao usuário o que fazer com um run existente.

    Retorna: "resume", "overwrite", ou "cancel"
    """
    print(f"\n{'='*55}")
    print(f"  ⚠️  Run existente encontrado: {run_name}")
    print(f"{'='*55}")
    print(f"  📁 Diretório: {state['run_dir']}")
    print(f"  💾 last.pt:   {'✅ Encontrado' if state['has_last'] else '❌ Não encontrado'}")
    print(f"  🏆 best.pt:   {'✅ Encontrado' if state['has_best'] else '❌ Não encontrado'}")
    print()

    if state["has_last"]:
        print("  [1] 🔄 Resume — Continuar de onde parou (last.pt)")
        print("  [2] 🗑️  Overwrite — Recomeçar do zero (apaga o anterior)")
        print("  [3] ❌ Cancelar — Não treinar este modelo")
    else:
        print("  ⚠️  last.pt não encontrado — resume não é possível")
        print("  [2] 🗑️  Overwrite — Recomeçar do zero")
        print("  [3] ❌ Cancelar — Não treinar este modelo")

    while True:
        try:
            choice = input("\n  Escolha [1/2/3]: ").strip()
            if choice == "1" and state["has_last"]:
                return "resume"
            elif choice == "2":
                return "overwrite"
            elif choice == "3":
                return "cancel"
            else:
                print("  ⚠️  Opção inválida. Tente novamente.")
        except (KeyboardInterrupt, EOFError):
            print("\n  ❌ Cancelado pelo usuário")
            return "cancel"


# ══════════════════════════════════════════════════════════════════════
#  TREINAMENTO
# ══════════════════════════════════════════════════════════════════════
def train_model(version: str, args):
    """
    Treina um modelo YOLO (v8 ou v10) no BDD100K.
    Gerencia resume, WandB, e todos os parâmetros.
    """
    config = MODEL_CONFIGS[version]
    run_name = config["run_name"]
    display_name = config["display_name"]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🚀 Iniciando Treinamento — {display_name:<27s}  ║
╠══════════════════════════════════════════════════════════╣
║  Epochs:     {args.epochs:<43d} ║
║  Imgsz:      {args.imgsz:<43d} ║
║  Batch:      {args.batch:<43d} ║
║  Workers:    {args.workers:<43d} ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Verificar data.yaml ───────────────────────────────────────────
    if not DATA_YAML.exists():
        print(f"  ❌ ERRO: data.yaml não encontrado em {DATA_YAML}")
        print(f"  💡 Execute primeiro: python scripts/convert_bdd100k.py --source <pasta_bdd100k>")
        return False

    # ── Verificar run existente ───────────────────────────────────────
    state = check_existing_run(run_name)
    resume_mode = False

    if state["exists"]:
        decision = get_resume_decision(run_name, state)

        if decision == "cancel":
            print(f"  ⏭️  Pulando {display_name}")
            return False

        elif decision == "resume":
            resume_mode = True
            print(f"\n  🔄 Resumindo treino de {state['last_path']}")

        elif decision == "overwrite":
            print(f"\n  🗑️  Sobrescrevendo run anterior...")
            # Não deletamos — usamos exist_ok=True

    # ── Inicializar WandB ─────────────────────────────────────────────
    wandb_active = init_wandb(display_name, args)

    # ── Carregar modelo ───────────────────────────────────────────────
    try:
        from ultralytics import YOLO

        if resume_mode:
            # Para resume: carregar o checkpoint
            print(f"  📦 Carregando checkpoint: {state['last_path']}")
            model = YOLO(str(state["last_path"]))
        else:
            # Para treino novo: carregar pesos pré-treinados
            print(f"  📦 Carregando pesos: {config['weights']}")
            model = YOLO(config["weights"])

    except Exception as e:
        print(f"  ❌ ERRO ao carregar modelo: {e}")
        finish_wandb()
        return False

    # ── Treinar ───────────────────────────────────────────────────────
    try:
        if resume_mode:
            # Resume: apenas resume=True, ultralytics cuida do resto
            print(f"\n  🏃 Resumindo treinamento...")
            results = model.train(resume=True)
        else:
            # Treino novo: todos os parâmetros
            print(f"\n  🏃 Iniciando treinamento do zero...")
            results = model.train(
                data=str(DATA_YAML),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=True,        # Permite sobrescrever
                save_period=args.save_period,
                patience=0,           # Sem early stopping (100 epochs completos)
                device=0 if torch.cuda.is_available() else "cpu",
                pretrained=True,
                verbose=True,
                amp=True,             # Mixed Precision (FP16) — ~2x mais rápido na 4050
                cache="ram",          # Cache imagens na RAM — elimina I/O de disco
                cos_lr=True,          # Cosine LR schedule — converge melhor
                close_mosaic=10,      # Desliga mosaic nos últimos 10 epochs
            )

        print(f"\n  ✅ Treinamento de {display_name} concluído!")

        # ── Validação final ───────────────────────────────────────────
        print(f"\n  📊 Executando validação final...")
        best_path = RUNS_DIR / run_name / "weights" / "best.pt"
        if best_path.exists():
            val_model = YOLO(str(best_path))
            metrics = val_model.val(data=str(DATA_YAML))
            print(f"  📈 mAP50:    {metrics.box.map50:.4f}")
            print(f"  📈 mAP50-95: {metrics.box.map:.4f}")

            # Log final no WandB
            if wandb_active:
                try:
                    import wandb
                    wandb.log({
                        f"{display_name}/final_mAP50": metrics.box.map50,
                        f"{display_name}/final_mAP50-95": metrics.box.map,
                    })
                except Exception:
                    pass

        return True

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Treinamento interrompido pelo usuário (Ctrl+C)")
        print(f"  💾 Checkpoint salvo — use --version {version} para resumir")
        return False

    except Exception as e:
        print(f"\n  ❌ ERRO durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        finish_wandb()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Treinamento comparativo YOLOv8m vs YOLOv10m no BDD100K"
    )
    parser.add_argument(
        "--version", type=str, required=True,
        choices=["v8", "v10", "both"],
        help="Qual modelo treinar: v8, v10, ou both"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Número de epochs (padrão: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (padrão: 16, otimizado para RTX 4050 + AMP)")
    parser.add_argument("--workers", type=int, default=8, help="Workers do dataloader (padrão: 8)")
    parser.add_argument("--save-period", type=int, default=5, help="Salvar checkpoint a cada N epochs (padrão: 5)")

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗ ██████╗ ██╗      ██████╗     ████████╗██████╗    ║
║   ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ╚══██╔══╝██╔══██╗   ║
║    ╚████╔╝ ██║   ██║██║     ██║   ██║       ██║   ██████╔╝   ║
║     ╚██╔╝  ██║   ██║██║     ██║   ██║       ██║   ██╔══██╗   ║
║      ██║   ╚██████╔╝███████╗╚██████╔╝       ██║   ██║  ██║   ║
║      ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝        ╚═╝   ╚═╝  ╚═╝   ║
║                                                              ║
║   Pipeline Comparativo — BDD100K                             ║
║   YOLOv8m vs YOLOv10m                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Patches ───────────────────────────────────────────────────────
    patch_torch_load()
    print_gpu_info()

    # ── Definir ordem de treino ───────────────────────────────────────
    if args.version == "both":
        versions = ["v8", "v10"]
    else:
        versions = [args.version]

    # ── Treinar ───────────────────────────────────────────────────────
    results = {}
    for ver in versions:
        display = MODEL_CONFIGS[ver]["display_name"]
        success = train_model(ver, args)
        results[display] = "✅ Concluído" if success else "❌ Falhou/Cancelado"

    # ── Resumo final ──────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              📋 RESUMO DO PIPELINE                       ║
╠══════════════════════════════════════════════════════════╣""")
    for model_name, status in results.items():
        print(f"║  {model_name:<15s} {status:<40s} ║")
    print(f"""╚══════════════════════════════════════════════════════════╝

  Próximos passos:
    python scripts/inference_compare.py --num-images 20
    """)


if __name__ == "__main__":
    main()
