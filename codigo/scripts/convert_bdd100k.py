#!/usr/bin/env python3
"""
==========================================================================
  BDD100K JSON → YOLO TXT  —  Conversor de Anotações
==========================================================================
  Converte os labels originais do BDD100K (JSON) para o formato YOLO
  (um .txt por imagem) e gera o data.yaml automaticamente.

  Dataset esperado: solesensei/bdd100k (Kaggle)
  Estrutura esperada após extração do ZIP:
    IC_2/
    └── bdd100k/          (ou bdd100k_images/ + bdd100k_labels/)
        ├── images/100k/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── bdd100k_labels_images_train.json
            └── bdd100k_labels_images_val.json

  Uso:
    python scripts/convert_bdd100k.py --source ./bdd100k --output ./data/bdd100k
==========================================================================
"""

import json
import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Mapeamento das 10 classes de detecção do BDD100K ──────────────────
BDD100K_CLASSES = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD100K_CLASSES)}

# ── Dimensões padrão das imagens BDD100K ──────────────────────────────
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def convert_bbox_to_yolo(x1: float, y1: float, x2: float, y2: float,
                         img_w: int = IMG_WIDTH, img_h: int = IMG_HEIGHT) -> tuple:
    """
    Converte coordenadas absolutas (x1, y1, x2, y2) em formato YOLO
    normalizado (x_center, y_center, width, height).

    Fórmulas:
        x_center = (x1 + x2) / (2 * img_w)
        y_center = (y1 + y2) / (2 * img_h)
        w        = (x2 - x1) / img_w
        h        = (y2 - y1) / img_h

    Todas as coordenadas são clamped para [0, 1].
    Retorna None se a bbox for inválida (largura ou altura <= 0).
    """
    # Validação: rejeitar bboxes inválidas
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = (x1 + x2) / (2.0 * img_w)
    y_center = (y1 + y2) / (2.0 * img_h)
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Clamp para [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (x_center, y_center, w, h)


def find_source_paths(source_dir: Path) -> dict:
    """
    Detecta a estrutura do dataset extraído e retorna os caminhos
    dos JSONs e das pastas de imagens.

    O solesensei/bdd100k pode ter múltiplas estruturas:
      - bdd100k/images/100k/{train,val}/
      - bdd100k/bdd100k/images/100k/{train,val}/
      - images/{train,val}/  (raro)
    """
    result = {
        "train_json": None,
        "val_json": None,
        "train_images": None,
        "val_images": None,
    }

    # ── Buscar JSONs de labels ────────────────────────────────────────
    for root, _, files in os.walk(source_dir):
        for f in files:
            path = Path(root) / f
            if "train" in f.lower() and f.endswith(".json") and "labels" in f.lower():
                result["train_json"] = path
            elif "val" in f.lower() and f.endswith(".json") and "labels" in f.lower():
                result["val_json"] = path

    # Fallback: buscar por nome parcial
    if result["train_json"] is None:
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.endswith(".json") and "train" in f.lower():
                    result["train_json"] = Path(root) / f
                    break

    if result["val_json"] is None:
        for root, _, files in os.walk(source_dir):
            for f in files:
                if f.endswith(".json") and "val" in f.lower():
                    result["val_json"] = Path(root) / f
                    break

    # ── Buscar pastas de imagens ──────────────────────────────────────
    for root, dirs, _ in os.walk(source_dir):
        root_path = Path(root)
        if root_path.name == "train" and "images" in str(root_path).lower():
            # Verificar se contém imagens
            sample = list(root_path.glob("*.jpg"))[:1]
            if sample:
                result["train_images"] = root_path
        elif root_path.name == "val" and "images" in str(root_path).lower():
            sample = list(root_path.glob("*.jpg"))[:1]
            if sample:
                result["val_images"] = root_path

    return result


def process_json(json_path: Path, labels_dir: Path, split_name: str) -> dict:
    """
    Processa um JSON de labels do BDD100K e gera os .txt YOLO.
    Retorna estatísticas da conversão.
    """
    stats = {
        "total_images": 0,
        "images_with_labels": 0,
        "total_boxes": 0,
        "skipped_boxes": 0,
        "class_counts": {name: 0 for name in BDD100K_CLASSES},
    }

    print(f"\n{'='*60}")
    print(f"  Processando {split_name}: {json_path.name}")
    print(f"{'='*60}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(data, desc=f"  [{split_name}] Convertendo", unit="img"):
        stats["total_images"] += 1
        image_name = frame.get("name", "")
        labels = frame.get("labels", [])

        if not image_name:
            continue

        # Nome do TXT = nome da imagem sem extensão
        txt_name = Path(image_name).stem + ".txt"
        txt_path = labels_dir / txt_name

        lines = []
        for label in labels:
            category = label.get("category", "")
            box2d = label.get("box2d", None)

            # Pular se não é uma classe de detecção ou não tem bbox
            if category not in CLASS_TO_ID or box2d is None:
                stats["skipped_boxes"] += 1
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            yolo_bbox = convert_bbox_to_yolo(x1, y1, x2, y2)
            if yolo_bbox is None:
                stats["skipped_boxes"] += 1
                continue

            class_id = CLASS_TO_ID[category]
            xc, yc, w, h = yolo_bbox
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            stats["total_boxes"] += 1
            stats["class_counts"][category] += 1

        # Escrever mesmo se vazio (imagem sem objetos = .txt vazio)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if lines:
            stats["images_with_labels"] += 1

    return stats


def create_data_yaml(output_dir: Path) -> Path:
    """
    Gera o data.yaml com caminhos ABSOLUTOS para evitar bugs de resolução.
    """
    yaml_path = output_dir / "data.yaml"
    abs_path = str(output_dir.resolve()).replace("\\", "/")

    data = {
        "path": abs_path,
        "train": "images/train",
        "val": "images/val",
        "nc": len(BDD100K_CLASSES),
        "names": BDD100K_CLASSES,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✅ data.yaml gerado em: {yaml_path}")
    return yaml_path


def setup_directory_structure(output_dir: Path, source_paths: dict):
    """
    Cria a estrutura de diretórios YOLO e cria symlinks/copia imagens.
    """
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Linkar ou copiar imagens
    for split, key in [("train", "train_images"), ("val", "val_images")]:
        src = source_paths[key]
        dst = output_dir / "images" / split

        if src is None:
            print(f"⚠️  Pasta de imagens {split} não encontrada!")
            continue

        # Se o destino já tem imagens, pular
        existing = list(dst.glob("*.jpg"))
        if len(existing) > 100:
            print(f"  ✅ {split}/images já contém {len(existing)} imagens — pulando")
            continue

        # Tentar symlink primeiro (mais rápido, economiza disco)
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()

        if src_resolved == dst_resolved:
            print(f"  ✅ {split}/images já aponta para a fonte — pulando")
            continue

        print(f"\n  📁 Copiando imagens {split}...")
        print(f"     De: {src_resolved}")
        print(f"     Para: {dst_resolved}")

        try:
            # Tentar criar symlink (requer privilégios no Windows)
            # Remove o diretório de destino vazio para criar o symlink
            if dst.exists() and not any(dst.iterdir()):
                dst.rmdir()
            os.symlink(str(src_resolved), str(dst_resolved), target_is_directory=True)
            print(f"  ✅ Symlink criado para {split}")
        except (OSError, NotImplementedError):
            # Fallback: copiar arquivos (Windows sem privilégios de admin)
            print(f"  ⚠️  Symlink falhou, copiando arquivos (pode demorar)...")
            # Ao invés de copiar tudo, vamos criar um junction point no Windows
            try:
                import subprocess
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst_resolved), str(src_resolved)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"  ✅ Junction criado para {split}")
                else:
                    raise OSError("Junction failed")
            except (OSError, subprocess.SubprocessError):
                # Último recurso: copiar
                print(f"  📋 Copiando arquivos... isso pode levar alguns minutos")
                for img in tqdm(list(src.glob("*.jpg")), desc=f"  Copiando {split}"):
                    shutil.copy2(str(img), str(dst / img.name))
                print(f"  ✅ {split} copiado")


def print_stats(stats: dict, split: str):
    """Imprime estatísticas da conversão."""
    print(f"\n{'─'*50}")
    print(f"  📊 Estatísticas — {split}")
    print(f"{'─'*50}")
    print(f"  Total de imagens:      {stats['total_images']:>8,}")
    print(f"  Imagens com labels:    {stats['images_with_labels']:>8,}")
    print(f"  Total de bboxes:       {stats['total_boxes']:>8,}")
    print(f"  Bboxes descartadas:    {stats['skipped_boxes']:>8,}")
    print(f"\n  Distribuição por classe:")
    for name, count in stats["class_counts"].items():
        bar = "█" * min(50, count // 1000)
        print(f"    {name:>15s}: {count:>8,}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Converte labels BDD100K (JSON) para formato YOLO (TXT)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Pasta raiz do dataset BDD100K extraído"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Pasta de saída para o dataset YOLO (padrão: ./data/bdd100k)"
    )
    args = parser.parse_args()

    # Determinar diretórios
    project_root = Path(__file__).resolve().parent.parent
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve() if args.output else project_root / "data" / "bdd100k"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║       BDD100K → YOLO  —  Conversor de Anotações         ║
╠══════════════════════════════════════════════════════════╣
║  Fonte:  {str(source_dir):<47s} ║
║  Saída:  {str(output_dir):<47s} ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Detectar caminhos ─────────────────────────────────────────────
    print("🔍 Buscando arquivos do dataset...")
    source_paths = find_source_paths(source_dir)

    # Validar que encontrou os JSONs
    if source_paths["train_json"] is None:
        print("❌ ERRO: Não encontrei o JSON de labels de treino!")
        print(f"   Procurei em: {source_dir}")
        print("   Esperado: arquivo .json com 'train' e 'labels' no nome")
        print("\n   Conteúdo da pasta raiz:")
        for item in sorted(source_dir.iterdir())[:20]:
            print(f"     {'📁' if item.is_dir() else '📄'} {item.name}")
        sys.exit(1)

    if source_paths["val_json"] is None:
        print("❌ ERRO: Não encontrei o JSON de labels de validação!")
        sys.exit(1)

    print(f"  ✅ Train JSON: {source_paths['train_json']}")
    print(f"  ✅ Val JSON:   {source_paths['val_json']}")
    print(f"  {'✅' if source_paths['train_images'] else '⚠️'}  Train imgs: {source_paths['train_images'] or 'NÃO ENCONTRADO'}")
    print(f"  {'✅' if source_paths['val_images'] else '⚠️'}  Val imgs:   {source_paths['val_images'] or 'NÃO ENCONTRADO'}")

    # ── Criar estrutura de diretórios ─────────────────────────────────
    print("\n📂 Criando estrutura de diretórios...")
    setup_directory_structure(output_dir, source_paths)

    # ── Converter labels ──────────────────────────────────────────────
    train_stats = process_json(
        source_paths["train_json"],
        output_dir / "labels" / "train",
        "TRAIN"
    )

    val_stats = process_json(
        source_paths["val_json"],
        output_dir / "labels" / "val",
        "VAL"
    )

    # ── Estatísticas ──────────────────────────────────────────────────
    print_stats(train_stats, "TRAIN")
    print_stats(val_stats, "VAL")

    # ── Gerar data.yaml ───────────────────────────────────────────────
    yaml_path = create_data_yaml(output_dir)

    # ── Resumo final ──────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                 ✅ CONVERSÃO CONCLUÍDA                    ║
╠══════════════════════════════════════════════════════════╣
║  data.yaml:  {str(yaml_path):<43s} ║
║  Labels:     {str(output_dir / 'labels'):<43s} ║
║  Images:     {str(output_dir / 'images'):<43s} ║
╚══════════════════════════════════════════════════════════╝

  Próximo passo:
    python scripts/train_yolo.py --version both
    """)


if __name__ == "__main__":
    main()
