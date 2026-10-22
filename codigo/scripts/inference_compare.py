#!/usr/bin/env python3
"""
==========================================================================
  Inferência Comparativa — YOLOv8m vs YOLOv10m
==========================================================================
  Carrega os best.pt de ambos os modelos treinados, roda inferência em
  imagens de validação, e gera:
    1. Imagens lado a lado com bounding boxes
    2. Tabela comparativa de métricas (mAP50, mAP50-95)
    3. Gráficos de distribuição de confiança

  Uso:
    python scripts/inference_compare.py                    # 10 imagens
    python scripts/inference_compare.py --num-images 30    # 30 imagens
    python scripts/inference_compare.py --source path/to/imgs  # imagens custom
==========================================================================
"""

import os
import sys
import random
import argparse
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para salvar imagens
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Diretórios do projeto ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
DATA_YAML = PROJECT_ROOT / "data" / "bdd100k" / "data.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison"

# ── Classes BDD100K ───────────────────────────────────────────────────
BDD100K_CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train",
]

# ── Cores para cada classe (BGR para OpenCV) ──────────────────────────
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

# ── Modelos ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "v8": {
        "run_name": "yolov8m_bdd100k",
        "display_name": "YOLOv8m",
        "color": "#3498db",  # Azul
    },
    "v10": {
        "run_name": "yolov10m_bdd100k",
        "display_name": "YOLOv10m",
        "color": "#e74c3c",  # Vermelho
    },
}


def find_best_weights(version: str) -> Path:
    """Localiza o best.pt de um modelo treinado."""
    config = MODEL_CONFIGS[version]
    best_path = RUNS_DIR / config["run_name"] / "weights" / "best.pt"
    if best_path.exists():
        return best_path
    # Fallback: procurar em subpastas numeradas
    run_dir = RUNS_DIR / config["run_name"]
    if not run_dir.exists():
        return None
    for p in sorted(run_dir.glob("**/best.pt")):
        return p
    return None


def load_models() -> dict:
    """Carrega os modelos treinados."""
    from ultralytics import YOLO

    models = {}
    for version, config in MODEL_CONFIGS.items():
        best_path = find_best_weights(version)
        if best_path is None:
            print(f"  ⚠️  {config['display_name']}: best.pt não encontrado — pulando")
            continue

        print(f"  📦 Carregando {config['display_name']}: {best_path}")
        model = YOLO(str(best_path))
        models[version] = {
            "model": model,
            "config": config,
            "path": best_path,
        }

    return models


def get_validation_images(source: str = None, num_images: int = 10) -> list:
    """Obtém imagens de validação para inferência."""
    if source:
        img_dir = Path(source)
    else:
        img_dir = PROJECT_ROOT / "data" / "bdd100k" / "images" / "val"

    if not img_dir.exists():
        print(f"  ❌ Diretório de imagens não encontrado: {img_dir}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]

    if not all_images:
        print(f"  ❌ Nenhuma imagem encontrada em: {img_dir}")
        return []

    # Selecionar aleatoriamente
    num_images = min(num_images, len(all_images))
    selected = random.sample(all_images, num_images)

    print(f"  📸 {num_images} imagens selecionadas de {len(all_images)} disponíveis")
    return sorted(selected)


def draw_detections(image: np.ndarray, results, model_name: str) -> np.ndarray:
    """Desenha bounding boxes e labels em uma imagem."""
    img = image.copy()
    h, w = img.shape[:2]

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())

                # Cor e label
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                cls_name = BDD100K_CLASSES[cls_id] if cls_id < len(BDD100K_CLASSES) else f"cls_{cls_id}"
                label = f"{cls_name} {conf:.2f}"

                # Desenhar box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Fundo do label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Label do modelo no canto superior
    cv2.rectangle(img, (0, 0), (len(model_name) * 18 + 20, 35), (0, 0, 0), -1)
    cv2.putText(img, model_name, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def create_side_by_side(img_path: Path, models: dict, output_dir: Path, idx: int):
    """Cria imagem comparativa lado a lado."""
    # Ler imagem original
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  Não foi possível ler: {img_path.name}")
        return None

    results_dict = {}
    annotated_imgs = {}

    for version, model_info in models.items():
        model = model_info["model"]
        display_name = model_info["config"]["display_name"]

        # Inferência
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.45,
            verbose=False,
        )
        results_dict[version] = results
        annotated_imgs[version] = draw_detections(img, results, display_name)

    # Montar lado a lado
    versions = list(models.keys())
    if len(versions) == 2:
        # Lado a lado horizontal
        combined = np.hstack([annotated_imgs[v] for v in versions])
    elif len(versions) == 1:
        combined = annotated_imgs[versions[0]]
    else:
        return None

    # Salvar
    output_path = output_dir / f"compare_{idx:03d}_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "image": img_path.name,
        "output": output_path,
        "detections": {
            v: len(results_dict[v][0].boxes) if results_dict[v] and len(results_dict[v]) > 0 else 0
            for v in versions
        }
    }


def run_validation_metrics(models: dict) -> dict:
    """Roda validação formal e coleta métricas."""
    metrics = {}

    for version, model_info in models.items():
        display_name = model_info["config"]["display_name"]
        print(f"\n  📊 Validando {display_name}...")

        try:
            model = model_info["model"]
            val_results = model.val(
                data=str(DATA_YAML),
                verbose=False,
            )

            metrics[version] = {
                "model": display_name,
                "mAP50": val_results.box.map50,
                "mAP50-95": val_results.box.map,
                "precision": val_results.box.mp,
                "recall": val_results.box.mr,
            }
            print(f"     mAP50:    {val_results.box.map50:.4f}")
            print(f"     mAP50-95: {val_results.box.map:.4f}")
            print(f"     Precisão: {val_results.box.mp:.4f}")
            print(f"     Recall:   {val_results.box.mr:.4f}")

        except Exception as e:
            print(f"  ❌ Erro na validação de {display_name}: {e}")
            metrics[version] = {"model": display_name, "error": str(e)}

    return metrics


def create_metrics_chart(metrics: dict, output_dir: Path):
    """Cria gráfico comparativo de métricas."""
    versions = [v for v in metrics if "error" not in metrics[v]]

    if len(versions) < 2:
        print("  ⚠️  Menos de 2 modelos — gráfico comparativo não será gerado")
        return

    # Dados
    metric_names = ["mAP50", "mAP50-95", "precision", "recall"]
    display_names = [metrics[v]["model"] for v in versions]
    colors = [MODEL_CONFIGS[v]["color"] for v in versions]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 5))
    fig.suptitle("Comparativo de Métricas — YOLOv8m vs YOLOv10m no BDD100K",
                 fontsize=14, fontweight="bold", y=1.02)

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        values = [metrics[v].get(metric, 0) for v in versions]
        bars = ax.bar(display_names, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Valores nas barras
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart_path = output_dir / "metrics_comparison.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Gráfico salvo em: {chart_path}")


def create_metrics_table(metrics: dict, output_dir: Path):
    """Salva tabela de métricas em CSV."""
    import csv

    csv_path = output_dir / "metrics_comparison.csv"
    fieldnames = ["Model", "mAP50", "mAP50-95", "Precision", "Recall"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for version, data in metrics.items():
            if "error" in data:
                continue
            writer.writerow({
                "Model": data["model"],
                "mAP50": f"{data['mAP50']:.4f}",
                "mAP50-95": f"{data['mAP50-95']:.4f}",
                "Precision": f"{data['precision']:.4f}",
                "Recall": f"{data['recall']:.4f}",
            })

    print(f"  📋 Tabela salva em: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inferência comparativa YOLOv8m vs YOLOv10m"
    )
    parser.add_argument(
        "--num-images", type=int, default=10,
        help="Número de imagens para comparação visual (padrão: 10)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Pasta com imagens para inferência (padrão: data/bdd100k/images/val)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Threshold de confiança (padrão: 0.25)"
    )
    parser.add_argument(
        "--skip-val", action="store_true",
        help="Pular validação formal (só faz comparação visual)"
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🔍 Inferência Comparativa — YOLOv8m vs YOLOv10m      ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Criar diretório de resultados ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Resultados em: {output_dir}")

    # ── Carregar modelos ──────────────────────────────────────────────
    print("\n  🔄 Carregando modelos...")
    models = load_models()

    if not models:
        print("\n  ❌ Nenhum modelo encontrado! Treine primeiro:")
        print("     python scripts/train_yolo.py --version both")
        sys.exit(1)

    # ── Validação formal ──────────────────────────────────────────────
    metrics = {}
    if not args.skip_val:
        print(f"\n{'─'*55}")
        print("  📊 Validação Formal")
        print(f"{'─'*55}")
        metrics = run_validation_metrics(models)
        create_metrics_chart(metrics, output_dir)
        create_metrics_table(metrics, output_dir)

    # ── Comparação visual ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  🖼️  Comparação Visual")
    print(f"{'─'*55}")

    images = get_validation_images(args.source, args.num_images)
    if not images:
        print("  ❌ Sem imagens para comparação")
        return

    comparison_results = []
    for idx, img_path in enumerate(images, 1):
        print(f"  [{idx}/{len(images)}] {img_path.name}...", end=" ")
        result = create_side_by_side(img_path, models, output_dir, idx)
        if result:
            comparison_results.append(result)
            det_info = " | ".join(
                f"{MODEL_CONFIGS[v]['display_name']}: {d} det."
                for v, d in result["detections"].items()
            )
            print(f"✅ ({det_info})")
        else:
            print("⚠️  falhou")

    # ── Resumo final ──────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║              ✅ COMPARAÇÃO CONCLUÍDA                     ║
╠══════════════════════════════════════════════════════════╣
║  Imagens comparativas:  {len(comparison_results):>3d} geradas                     ║
║  Resultados em:         results/comparison/{timestamp}  ║""")

    if metrics:
        print(f"║{'─'*56}║")
        for v, m in metrics.items():
            if "error" not in m:
                print(f"║  {m['model']:<12s} mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}  ║")

    print(f"╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
