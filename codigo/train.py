#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
  PIPELINE UNIFICADO — YOLOv8m vs YOLOv10m no BDD100K
══════════════════════════════════════════════════════════════════════════
  Script orquestrador que executa TODO o pipeline em sequência:

    ETAPA 1 → Checagem de GPU e ambiente
    ETAPA 2 → Conversão do dataset BDD100K (JSON → YOLO TXT)
    ETAPA 3 → Treinamento YOLOv8m (100 epochs, resume automático)
    ETAPA 4 → Treinamento YOLOv10m (100 epochs, resume automático)
    ETAPA 5 → Comparação de resultados (métricas + visual lado a lado)

  Uso:
    python train.py --source ./bdd100k

  Opções:
    --source       Pasta do BDD100K extraído (obrigatório na 1ª vez)
    --skip-convert Pular conversão (se já rodou antes)
    --skip-train   Pular treino (só comparar resultados)
    --epochs       Número de epochs (padrão: 100)
    --batch        Batch size (padrão: 8, ajustado para 6GB VRAM)
    --num-images   Imagens para comparação visual (padrão: 20)
══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import csv
import shutil
import random
import argparse
import subprocess
import time
import traceback
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTES DO PROJETO
# ══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "bdd100k"
DATA_YAML = DATA_DIR / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison"

# ── Classes BDD100K (10 classes de detecção) ──────────────────────────
BDD100K_CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD100K_CLASSES)}

# ── Dimensões padrão das imagens BDD100K ──────────────────────────────
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# ── Configuração dos modelos ──────────────────────────────────────────
MODEL_CONFIGS = {
    "v8": {
        "weights": "yolov8m.pt",
        "run_name": "yolov8m_bdd100k",
        "display_name": "YOLOv8m",
        "color": "#3498db",
    },
    "v10": {
        "weights": "yolov10m.pt",
        "run_name": "yolov10m_bdd100k",
        "display_name": "YOLOv10m",
        "color": "#e74c3c",
    },
}

# ── Cores por classe para visualização (BGR) ─────────────────────────
CLASS_COLORS = [
    (0, 255, 0),       # person - verde
    (255, 165, 0),     # rider - laranja
    (0, 0, 255),       # car - vermelho
    (255, 0, 255),     # bus - magenta
    (0, 255, 255),     # truck - amarelo
    (255, 255, 0),     # bike - ciano
    (128, 0, 128),     # motor - roxo
    (0, 128, 255),     # traffic light - laranja escuro
    (255, 128, 0),     # traffic sign - azul claro
    (128, 128, 0),     # train - teal
]


def banner():
    """Exibe o banner do pipeline."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗   ██╗ ██████╗ ██╗      ██████╗     ████████╗██████╗       ║
║   ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗    ╚══██╔══╝██╔══██╗      ║
║    ╚████╔╝ ██║   ██║██║     ██║   ██║       ██║   ██████╔╝      ║
║     ╚██╔╝  ██║   ██║██║     ██║   ██║       ██║   ██╔══██╗      ║
║      ██║   ╚██████╔╝███████╗╚██████╔╝       ██║   ██║  ██║      ║
║      ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝        ╚═╝   ╚═╝  ╚═╝      ║
║                                                                  ║
║   Pipeline Unificado — BDD100K                                   ║
║   YOLOv8m vs YOLOv10m • GPU Check • Convert • Train • Compare   ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def step_header(step_num: int, total: int, title: str):
    """Imprime header de uma etapa."""
    print(f"""
{'━'*65}
  ETAPA {step_num}/{total} ─ {title}
{'━'*65}""")


# ══════════════════════════════════════════════════════════════════════
#  ETAPA 1: CHECAGEM DE GPU E AMBIENTE
# ══════════════════════════════════════════════════════════════════════
def step1_check_gpu():
    """Verifica GPU, CUDA, PyTorch e aplica patches necessários."""
    import torch

    step_header(1, 5, "CHECAGEM DE GPU E AMBIENTE")

    # ── Patch PyTorch 2.6+ ────────────────────────────────────────────
    if not hasattr(torch, "_original_load"):
        torch._original_load = torch.load

        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return torch._original_load(*args, **kwargs)

        torch.load = _patched_load
        print("  🔧 Patch PyTorch 2.6+ (weights_only=False) aplicado")

    # ── Informações do sistema ────────────────────────────────────────
    print(f"\n  📋 Ambiente:")
    print(f"     Python:     {sys.version.split()[0]}")
    print(f"     PyTorch:    {torch.__version__}")

    try:
        import ultralytics
        print(f"     Ultralytics:{ultralytics.__version__}")
    except ImportError:
        print("     Ultralytics: ❌ NÃO INSTALADO")
        print("\n  💡 Execute: pip install -r requirements.txt")
        sys.exit(1)

    # ── GPU ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)

        print(f"\n  🖥️  GPU:")
        print(f"     Modelo:     {gpu_name}")
        print(f"     VRAM:       {vram_gb:.1f} GB")
        print(f"     CUDA:       {torch.version.cuda}")
        print(f"     cuDNN:      {torch.backends.cudnn.version()}")
        print(f"     Device:     cuda:0 ✅")

        # Aviso de VRAM
        if vram_gb < 6:
            print(f"\n  ⚠️  VRAM baixa ({vram_gb:.1f}GB). Considere batch=4 ou imgsz=416")

        return True
    else:
        print(f"\n  ⚠️  GPU / CUDA:")
        print(f"     CUDA NÃO DISPONÍVEL")
        print(f"     Treinamento será em CPU (EXTREMAMENTE LENTO)")

        # Tentar rodar nvidia-smi para diagóstico
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"\n  ℹ️  nvidia-smi detectou GPU, mas PyTorch não tem suporte CUDA.")
                print(f"     Reinstale PyTorch com CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            else:
                print(f"\n  ℹ️  nvidia-smi falhou — sem driver NVIDIA instalado?")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"\n  ℹ️  nvidia-smi não encontrado — sem GPU NVIDIA?")

        resp = input("\n  Continuar sem GPU? [s/N]: ").strip().lower()
        if resp != "s":
            print("  ❌ Abortado.")
            sys.exit(0)
        return False


# ══════════════════════════════════════════════════════════════════════
#  ETAPA 2: CONVERSÃO DO DATASET
# ══════════════════════════════════════════════════════════════════════
def convert_bbox_to_yolo(x1, y1, x2, y2, img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    """
    Converte (x1,y1,x2,y2) absolutas → (x_center, y_center, w, h) normalizadas.
    Retorna None se bbox inválida.
    """
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = (x1 + x2) / (2.0 * img_w)
    y_center = (y1 + y2) / (2.0 * img_h)
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Clamp [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (x_center, y_center, w, h)


def find_source_paths(source_dir: Path) -> dict:
    """Auto-detecta JSONs e pastas de imagens no dataset extraído."""
    result = {
        "train_json": None, "val_json": None,
        "train_images": None, "val_images": None,
    }

    # Buscar JSONs
    for root, _, files in os.walk(source_dir):
        for f in files:
            fl = f.lower()
            if fl.endswith(".json"):
                path = Path(root) / f
                if "train" in fl and ("label" in fl or "det" in fl):
                    result["train_json"] = path
                elif "val" in fl and ("label" in fl or "det" in fl):
                    result["val_json"] = path

    # Fallback: qualquer JSON com train/val no nome
    for key, keyword in [("train_json", "train"), ("val_json", "val")]:
        if result[key] is None:
            for root, _, files in os.walk(source_dir):
                for f in files:
                    if f.endswith(".json") and keyword in f.lower():
                        result[key] = Path(root) / f
                        break
                if result[key]:
                    break

    # Buscar pastas de imagens
    for root, dirs, _ in os.walk(source_dir):
        root_path = Path(root)
        if root_path.name == "train" and "image" in str(root_path).lower():
            if list(root_path.glob("*.jpg"))[:1]:
                result["train_images"] = root_path
        elif root_path.name == "val" and "image" in str(root_path).lower():
            if list(root_path.glob("*.jpg"))[:1]:
                result["val_images"] = root_path

    return result


def process_json_labels(json_path: Path, labels_dir: Path, split: str) -> dict:
    """Converte um JSON de labels BDD100K → arquivos .txt YOLO."""
    from tqdm import tqdm

    stats = {"total": 0, "with_labels": 0, "boxes": 0, "skipped": 0,
             "classes": {n: 0 for n in BDD100K_CLASSES}}

    print(f"\n  📄 Processando {split}: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(data, desc=f"     [{split}]", unit="img", ncols=80):
        stats["total"] += 1
        img_name = frame.get("name", "")
        labels = frame.get("labels", [])
        if not img_name:
            continue

        txt_path = labels_dir / (Path(img_name).stem + ".txt")
        lines = []

        for label in labels:
            cat = label.get("category", "")
            box = label.get("box2d")
            if cat not in CLASS_TO_ID or box is None:
                stats["skipped"] += 1
                continue

            yolo = convert_bbox_to_yolo(float(box["x1"]), float(box["y1"]),
                                         float(box["x2"]), float(box["y2"]))
            if yolo is None:
                stats["skipped"] += 1
                continue

            cid = CLASS_TO_ID[cat]
            lines.append(f"{cid} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
            stats["boxes"] += 1
            stats["classes"][cat] += 1

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        if lines:
            stats["with_labels"] += 1

    return stats


def link_images(src: Path, dst: Path, split: str):
    """
    Cria junction/symlink das imagens de origem para a estrutura YOLO.
    Tenta: junction → symlink → cópia (nessa ordem).
    """
    from tqdm import tqdm

    dst.mkdir(parents=True, exist_ok=True)

    if src is None:
        print(f"  ⚠️  Pasta de imagens {split} não encontrada!")
        return

    src_r = src.resolve()
    dst_r = dst.resolve()

    if src_r == dst_r:
        print(f"  ✅ {split} imagens já no lugar correto")
        return

    # Se destino já tem muitas imagens, pular
    existing = list(dst.glob("*.jpg"))
    if len(existing) > 100:
        print(f"  ✅ {split} já contém {len(existing):,} imagens — pulando")
        return

    # Se é um junction/symlink existente, verificar se aponta certo
    if dst.is_symlink() or dst.exists():
        try:
            test_files = list(dst.glob("*.jpg"))[:1]
            if test_files:
                print(f"  ✅ {split} já configurado (junction/symlink)")
                return
        except Exception:
            pass

    print(f"  📁 Configurando imagens {split}...")
    print(f"     Origem:  {src_r}")
    print(f"     Destino: {dst_r}")

    # Tentar junction (Windows, sem precisar de admin)
    try:
        if dst.exists() and not any(dst.iterdir()):
            dst.rmdir()
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst_r), str(src_r)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Junction criado para {split}")
            return
    except Exception:
        pass

    # Tentar symlink
    try:
        if dst.exists() and not any(dst.iterdir()):
            dst.rmdir()
        os.symlink(str(src_r), str(dst_r), target_is_directory=True)
        print(f"  ✅ Symlink criado para {split}")
        return
    except (OSError, NotImplementedError):
        pass

    # Último recurso: cópia
    print(f"  📋 Copiando imagens {split}... (pode demorar)")
    imgs = list(src.glob("*.jpg"))
    for img in tqdm(imgs, desc=f"     Copiando {split}", unit="img", ncols=80):
        shutil.copy2(str(img), str(dst / img.name))
    print(f"  ✅ {len(imgs):,} imagens copiadas para {split}")


def create_data_yaml():
    """Gera data.yaml com caminhos absolutos."""
    import yaml

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    abs_path = str(DATA_DIR.resolve()).replace("\\", "/")

    data = {
        "path": abs_path,
        "train": "images/train",
        "val": "images/val",
        "nc": len(BDD100K_CLASSES),
        "names": BDD100K_CLASSES,
    }

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n  ✅ data.yaml → {DATA_YAML}")
    return DATA_YAML


def step2_convert_dataset(source_dir: Path):
    """Converte BDD100K de JSON para YOLO TXT."""
    step_header(2, 5, "CONVERSÃO DO DATASET BDD100K")

    # ── Checar se já convertido ───────────────────────────────────────
    train_labels_dir = DATA_DIR / "labels" / "train"
    val_labels_dir = DATA_DIR / "labels" / "val"

    if train_labels_dir.exists() and val_labels_dir.exists():
        train_count = len(list(train_labels_dir.glob("*.txt")))
        val_count = len(list(val_labels_dir.glob("*.txt")))
        if train_count > 1000 and val_count > 100:
            print(f"\n  ✅ Dataset já convertido!")
            print(f"     Train: {train_count:,} labels")
            print(f"     Val:   {val_count:,} labels")

            # Garantir que data.yaml existe
            if not DATA_YAML.exists():
                create_data_yaml()
            return

    # ── Validar source ────────────────────────────────────────────────
    if source_dir is None:
        if DATA_YAML.exists():
            print(f"\n  ✅ data.yaml já existe: {DATA_YAML}")
            return
        print("\n  ❌ ERRO: Precisa informar --source na primeira execução!")
        print("     Exemplo: python train.py --source ./bdd100k")
        sys.exit(1)

    source_dir = Path(source_dir).resolve()
    if not source_dir.exists():
        print(f"\n  ❌ ERRO: Pasta não encontrada: {source_dir}")
        sys.exit(1)

    print(f"\n  📂 Fonte: {source_dir}")

    # ── Detectar estrutura ────────────────────────────────────────────
    print("  🔍 Buscando arquivos...")
    paths = find_source_paths(source_dir)

    if paths["train_json"] is None or paths["val_json"] is None:
        print("\n  ❌ ERRO: JSONs de labels não encontrados!")
        print(f"     Procurei em: {source_dir}")
        print("\n  📁 Conteúdo da pasta:")
        for item in sorted(source_dir.iterdir())[:20]:
            prefix = "📁" if item.is_dir() else "📄"
            print(f"       {prefix} {item.name}")
        sys.exit(1)

    print(f"  ✅ Train JSON: {paths['train_json'].name}")
    print(f"  ✅ Val JSON:   {paths['val_json'].name}")
    print(f"  {'✅' if paths['train_images'] else '⚠️'}  Train imgs: {paths['train_images'] or 'NÃO ENCONTRADO'}")
    print(f"  {'✅' if paths['val_images'] else '⚠️'}  Val imgs:   {paths['val_images'] or 'NÃO ENCONTRADO'}")

    # ── Configurar imagens ────────────────────────────────────────────
    print(f"\n  📂 Configurando estrutura de diretórios...")
    link_images(paths["train_images"], DATA_DIR / "images" / "train", "train")
    link_images(paths["val_images"], DATA_DIR / "images" / "val", "val")

    # ── Converter labels ──────────────────────────────────────────────
    train_stats = process_json_labels(paths["train_json"], train_labels_dir, "TRAIN")
    val_stats = process_json_labels(paths["val_json"], val_labels_dir, "VAL")

    # ── Estatísticas ──────────────────────────────────────────────────
    for name, stats in [("TRAIN", train_stats), ("VAL", val_stats)]:
        print(f"\n  📊 {name}: {stats['total']:,} imagens, {stats['boxes']:,} bboxes")
        top_classes = sorted(stats["classes"].items(), key=lambda x: -x[1])[:5]
        for cls, cnt in top_classes:
            print(f"       {cls:>15s}: {cnt:>8,}")

    # ── Gerar data.yaml ───────────────────────────────────────────────
    create_data_yaml()

    print(f"\n  ✅ Conversão concluída!")


# ══════════════════════════════════════════════════════════════════════
#  ETAPA 3 & 4: TREINAMENTO
# ══════════════════════════════════════════════════════════════════════
def check_and_resume(run_name: str) -> str:
    """
    Verifica se existe um run anterior.
    Retorna: "resume", "overwrite", ou "new"
    """
    run_dir = RUNS_DIR / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"

    if not run_dir.exists():
        return "new"

    print(f"\n  ⚠️  Run existente: {run_name}")
    print(f"     📁 {run_dir}")
    print(f"     💾 last.pt: {'✅' if last_pt.exists() else '❌'}")
    print(f"     🏆 best.pt: {'✅' if best_pt.exists() else '❌'}")

    if last_pt.exists():
        print(f"\n     [1] 🔄 Resume — continuar de onde parou")
        print(f"     [2] 🗑️  Overwrite — recomeçar do zero")
        print(f"     [3] ❌ Cancelar")

        while True:
            try:
                choice = input("\n     Escolha [1/2/3]: ").strip()
                if choice == "1":
                    return "resume"
                elif choice == "2":
                    return "overwrite"
                elif choice == "3":
                    return "cancel"
                print("     Opção inválida.")
            except (KeyboardInterrupt, EOFError):
                return "cancel"
    else:
        print(f"\n     last.pt não encontrado — resume impossível")
        print(f"     [2] 🗑️  Overwrite — recomeçar do zero")
        print(f"     [3] ❌ Cancelar")

        while True:
            try:
                choice = input("\n     Escolha [2/3]: ").strip()
                if choice == "2":
                    return "overwrite"
                elif choice == "3":
                    return "cancel"
                print("     Opção inválida.")
            except (KeyboardInterrupt, EOFError):
                return "cancel"


def init_wandb_safe(model_name: str, args) -> bool:
    """Inicializa WandB com proteção contra falha."""
    try:
        import wandb
        import torch

        wandb.init(
            project="BDD100K-Comparison",
            name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config={
                "model": model_name,
                "epochs": args.epochs,
                "imgsz": 640,
                "batch": args.batch,
                "workers": 12,
                "dataset": "BDD100K",
                "amp": True,
                "cache": "ram",
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            },
            tags=["bdd100k", model_name.lower(), "comparison"],
            resume="allow",
        )
        print(f"  ✅ WandB ativo — Projeto: BDD100K-Comparison")
        return True
    except Exception as e:
        print(f"  ⚠️  WandB indisponível: {e}")
        print(f"  ℹ️  Treino continua sem WandB. Para ativar: wandb login")
        return False


def finish_wandb_safe():
    """Finaliza WandB sem crashar."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def train_single_model(version: str, args) -> dict:
    """
    Treina um modelo YOLO. Retorna dict com resultado.
    """
    import torch
    from ultralytics import YOLO

    config = MODEL_CONFIGS[version]
    run_name = config["run_name"]
    display = config["display_name"]

    print(f"\n  🚀 {display}")
    print(f"     Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: 640 | Workers: 12 | AMP: ✅ | Cache: RAM")

    # ── Verificar data.yaml ───────────────────────────────────────────
    if not DATA_YAML.exists():
        print(f"  ❌ data.yaml não encontrado!")
        return {"model": display, "status": "failed", "error": "data.yaml não encontrado"}

    # ── Resume check ──────────────────────────────────────────────────
    decision = check_and_resume(run_name)

    if decision == "cancel":
        print(f"  ⏭️  Pulando {display}")
        return {"model": display, "status": "skipped"}

    # ── WandB ─────────────────────────────────────────────────────────
    wandb_ok = init_wandb_safe(display, args)

    # ── Carregar modelo ───────────────────────────────────────────────
    try:
        if decision == "resume":
            last_pt = RUNS_DIR / run_name / "weights" / "last.pt"
            print(f"  📦 Carregando checkpoint: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"  📦 Baixando/carregando pesos: {config['weights']}")
            model = YOLO(config["weights"])
    except Exception as e:
        print(f"  ❌ Erro ao carregar modelo: {e}")
        finish_wandb_safe()
        return {"model": display, "status": "failed", "error": str(e)}

    # ── Treinar ───────────────────────────────────────────────────────
    start_time = time.time()
    try:
        if decision == "resume":
            print(f"  🔄 Resumindo treinamento...")
            results = model.train(resume=True)
        else:
            print(f"  🏃 Iniciando treinamento do zero...")
            results = model.train(
                data=str(DATA_YAML),
                epochs=args.epochs,
                imgsz=640,
                batch=args.batch,
                workers=12,
                project=str(RUNS_DIR),
                name=run_name,
                exist_ok=True,
                save_period=5,
                patience=0,           # Sem early stopping
                device=0 if torch.cuda.is_available() else "cpu",
                pretrained=True,
                verbose=True,
                amp=True,             # Mixed Precision (FP16) — ~2x mais rápido
                cache="ram",          # Cache imagens na RAM — elimina I/O de disco
                cos_lr=True,          # Cosine LR — converge melhor
                close_mosaic=10,      # Desliga mosaic nos últimos 10 epochs
            )

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)

        print(f"\n  ✅ {display} concluído em {hours}h{mins:02d}m")

        # ── Validação final ───────────────────────────────────────────
        best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
        metrics = {}
        if best_pt.exists():
            print(f"  📊 Validação final com best.pt...")
            val_model = YOLO(str(best_pt))
            val_results = val_model.val(data=str(DATA_YAML))
            metrics = {
                "mAP50": float(val_results.box.map50),
                "mAP50-95": float(val_results.box.map),
                "precision": float(val_results.box.mp),
                "recall": float(val_results.box.mr),
            }
            print(f"     mAP50:    {metrics['mAP50']:.4f}")
            print(f"     mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"     Precision:{metrics['precision']:.4f}")
            print(f"     Recall:   {metrics['recall']:.4f}")

            # Log final WandB
            if wandb_ok:
                try:
                    import wandb
                    wandb.log({f"{display}/final_{k}": v for k, v in metrics.items()})
                except Exception:
                    pass

        finish_wandb_safe()
        return {"model": display, "status": "completed", "time": f"{hours}h{mins:02d}m", **metrics}

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n  ⚠️  {display} interrompido (Ctrl+C) após {elapsed/60:.0f} min")
        print(f"  💾 Checkpoint salvo — rode novamente para resumir")
        finish_wandb_safe()
        return {"model": display, "status": "interrupted"}

    except Exception as e:
        print(f"\n  ❌ Erro no treino de {display}: {e}")
        traceback.print_exc()
        finish_wandb_safe()
        return {"model": display, "status": "failed", "error": str(e)}


def step3_train_v8(args) -> dict:
    """Treina YOLOv8m."""
    step_header(3, 5, "TREINAMENTO YOLOv8m")
    return train_single_model("v8", args)


def step4_train_v10(args) -> dict:
    """Treina YOLOv10m."""
    step_header(4, 5, "TREINAMENTO YOLOv10m")
    return train_single_model("v10", args)


# ══════════════════════════════════════════════════════════════════════
#  ETAPA 5: COMPARAÇÃO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════
def find_best_weights(version: str) -> Path:
    """Localiza o best.pt de um modelo treinado."""
    config = MODEL_CONFIGS[version]
    best = RUNS_DIR / config["run_name"] / "weights" / "best.pt"
    if best.exists():
        return best
    run_dir = RUNS_DIR / config["run_name"]
    if run_dir.exists():
        for p in sorted(run_dir.glob("**/best.pt")):
            return p
    return None


def draw_detections(image, results, model_name: str):
    """Desenha bounding boxes na imagem."""
    import cv2
    import numpy as np

    img = image.copy()
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                cls_name = BDD100K_CLASSES[cls_id] if cls_id < len(BDD100K_CLASSES) else f"cls{cls_id}"
                label = f"{cls_name} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Label do modelo
    cv2.rectangle(img, (0, 0), (len(model_name) * 18 + 20, 35), (0, 0, 0), -1)
    cv2.putText(img, model_name, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def step5_compare(args, train_results: list):
    """Compara resultados: métricas + visual lado a lado."""
    import torch
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    step_header(5, 5, "COMPARAÇÃO DE RESULTADOS")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Carregar modelos treinados ────────────────────────────────────
    models = {}
    for version, config in MODEL_CONFIGS.items():
        best = find_best_weights(version)
        if best is None:
            print(f"  ⚠️  {config['display_name']}: best.pt não encontrado — pulando")
            continue
        print(f"  📦 Carregando {config['display_name']}: {best}")
        models[version] = {
            "model": YOLO(str(best)),
            "config": config,
        }

    if not models:
        print("\n  ❌ Nenhum modelo encontrado para comparação.")
        print("     Os treinos falharam ou foram cancelados.")
        return

    # ── Validação formal ──────────────────────────────────────────────
    print(f"\n  📊 Validação formal...")
    metrics = {}
    for version, info in models.items():
        display = info["config"]["display_name"]
        try:
            print(f"     Validando {display}...")
            val = info["model"].val(data=str(DATA_YAML), verbose=False)
            metrics[version] = {
                "model": display,
                "mAP50": float(val.box.map50),
                "mAP50-95": float(val.box.map),
                "precision": float(val.box.mp),
                "recall": float(val.box.mr),
            }
            m = metrics[version]
            print(f"       mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}")
        except Exception as e:
            print(f"     ❌ Erro: {e}")
            metrics[version] = {"model": display, "error": str(e)}

    # ── Gráfico comparativo ───────────────────────────────────────────
    valid = {v: m for v, m in metrics.items() if "error" not in m}
    if len(valid) >= 2:
        metric_names = ["mAP50", "mAP50-95", "precision", "recall"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Comparativo YOLOv8m vs YOLOv10m — BDD100K",
                     fontsize=15, fontweight="bold", y=1.02)

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            names = [valid[v]["model"] for v in valid]
            vals = [valid[v][metric] for v in valid]
            colors = [MODEL_CONFIGS[v]["color"] for v in valid]
            bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        chart_path = output_dir / "metrics_comparison.png"
        plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  📊 Gráfico salvo: {chart_path}")

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = output_dir / "metrics_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "mAP50", "mAP50-95", "Precision", "Recall"])
        writer.writeheader()
        for v, m in valid.items():
            writer.writerow({
                "Model": m["model"],
                "mAP50": f"{m['mAP50']:.4f}",
                "mAP50-95": f"{m['mAP50-95']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
            })
    print(f"  📋 CSV salvo: {csv_path}")

    # ── Salvar resultados de treino ───────────────────────────────────
    train_csv = output_dir / "training_summary.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Status", "Time", "mAP50", "mAP50-95", "Precision", "Recall"])
        for r in train_results:
            writer.writerow([
                r.get("model", ""),
                r.get("status", ""),
                r.get("time", ""),
                f"{r.get('mAP50', 0):.4f}" if "mAP50" in r else "",
                f"{r.get('mAP50-95', 0):.4f}" if "mAP50-95" in r else "",
                f"{r.get('precision', 0):.4f}" if "precision" in r else "",
                f"{r.get('recall', 0):.4f}" if "recall" in r else "",
            ])
    print(f"  📋 Resumo treino: {train_csv}")

    # ── Comparação visual lado a lado ─────────────────────────────────
    if len(models) >= 1:
        img_dir = DATA_DIR / "images" / "val"
        if img_dir.exists():
            all_imgs = list(img_dir.glob("*.jpg"))
            num = min(args.num_images, len(all_imgs))
            if num > 0:
                selected = random.sample(all_imgs, num)
                print(f"\n  🖼️  Gerando {num} comparações visuais...")
                visual_dir = output_dir / "visual"
                visual_dir.mkdir(exist_ok=True)

                for idx, img_path in enumerate(sorted(selected), 1):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    annotated = []
                    for version, info in models.items():
                        results = info["model"].predict(source=str(img_path), conf=0.25,
                                                         iou=0.45, verbose=False)
                        ann = draw_detections(img, results, info["config"]["display_name"])
                        annotated.append(ann)

                    combined = np.hstack(annotated) if len(annotated) > 1 else annotated[0]
                    out_path = visual_dir / f"compare_{idx:03d}_{img_path.stem}.jpg"
                    cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"     [{idx}/{num}] {img_path.name} ✅")

                print(f"\n  ✅ Imagens comparativas em: {visual_dir}")

    # ── Resumo final ──────────────────────────────────────────────────
    print(f"\n  📁 Todos os resultados em: {output_dir}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN — ORQUESTRADOR
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline unificado: GPU check → Conversão → Treino → Comparação"
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Pasta do BDD100K extraído (obrigatório na 1ª vez)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Pular conversão do dataset")
    parser.add_argument("--skip-train", action="store_true",
                        help="Pular treino (só comparar resultados existentes)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Número de epochs (padrão: 100)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (padrão: 8 para RTX 2060 6GB)")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Imagens para comparação visual (padrão: 20)")

    args = parser.parse_args()

    banner()
    start_total = time.time()

    # ━━━━━━━━━━ ETAPA 1: GPU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    has_gpu = step1_check_gpu()

    # ━━━━━━━━━━ ETAPA 2: CONVERSÃO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not args.skip_convert:
        step2_convert_dataset(args.source)
    else:
        print(f"\n  ⏭️  Conversão pulada (--skip-convert)")
        if not DATA_YAML.exists():
            print(f"  ⚠️  data.yaml não existe! Rode sem --skip-convert na primeira vez.")
            sys.exit(1)

    # ━━━━━━━━━━ ETAPAS 3 & 4: TREINO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    train_results = []
    if not args.skip_train:
        r8 = step3_train_v8(args)
        train_results.append(r8)

        r10 = step4_train_v10(args)
        train_results.append(r10)
    else:
        print(f"\n  ⏭️  Treino pulado (--skip-train)")

    # ━━━━━━━━━━ ETAPA 5: COMPARAÇÃO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    step5_compare(args, train_results)

    # ━━━━━━━━━━ FINALIZAÇÃO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    total_time = time.time() - start_total
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              ✅ PIPELINE COMPLETO                                 ║
║                                                                  ║
║  Tempo total: {hours}h{mins:02d}m                                           ║
║                                                                  ║""")
    for r in train_results:
        status_icon = {"completed": "✅", "skipped": "⏭️", "interrupted": "⚠️", "failed": "❌"}.get(r.get("status"), "❓")
        model_str = f"{r['model']}: {status_icon} {r['status']}"
        if "mAP50" in r:
            model_str += f" (mAP50={r['mAP50']:.4f})"
        print(f"║  {model_str:<62s}  ║")

    print(f"""║                                                                  ║
║  Resultados em: results/comparison/                              ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
