# 2. Instalação

[← Visão geral](01-visao-geral.md) · [Índice](../README.md) · [Próximo: Como executar →](03-como-executar.md)

---

## Requisitos

### Hardware

| Recurso | Mínimo | Observação |
|---|---|---|
| GPU | NVIDIA com 6 GB de VRAM | Em CPU o treino é inviável (dias por época) |
| RAM | 16 GB | Com `cache="ram"` fixo no código, o dataset completo pede ~48 GB. Veja o [problema 5](05-problemas-conhecidos.md#5-cache-em-ram) |
| Disco | ~25 GB livres | ~10 GB do dataset + labels convertidos + pesos e checkpoints |

Configuração usada nos treinos finais do [artigo](08-artigo-sicfei.md): **GPU NVIDIA RTX 4060, Intel Core i7 de 13ª geração (série U) e 16 GB de RAM**, batch 16, com 20h34min a 23h24min de treino por modelo (100 épocas, BDD100K completo).

Referência das primeiras execuções: com batch 4 e só 1.160 imagens de treino, cada época levou cerca de **55 minutos**.

### Software

| Item | Versão |
|---|---|
| Python | 3.10 ou mais novo |
| Driver NVIDIA + CUDA | CUDA 12.x |
| Sistema | Linux (onde o projeto rodou) ou Windows |

Versões que efetivamente rodaram o projeto: **torch 2.11.0+cu126** e **ultralytics 8.4.34**. O `requirements.txt` só define versões mínimas (`>=`), então uma instalação nova pode trazer versões diferentes (veja o [problema 12](05-problemas-conhecidos.md#12-reprodutibilidade)).

---

## Passo 1 — Ambiente Python

Na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r codigo/requirements.txt
```

O `requirements.txt` instala:

| Pacote | Para quê |
|---|---|
| `torch`, `torchvision` | Base de deep learning |
| `ultralytics` | Implementação do YOLOv8 e do YOLOv10 |
| `wandb` | Acompanhamento dos treinos (opcional, não quebra sem login) |
| `opencv-python`, `Pillow` | Leitura e desenho nas imagens |
| `matplotlib`, `seaborn` | Gráficos |
| `numpy`, `tqdm`, `huggingface_hub` | Utilitários |

### Conferir se a GPU está funcionando

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Se aparecer `False`, o PyTorch instalado não tem suporte a CUDA. Reinstale com a versão CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Se mesmo assim der `False`, rode `nvidia-smi`. Se o comando não existir ou falhar, falta instalar o driver NVIDIA.

---

## Passo 2 — Dataset BDD100K

### Download

**Opção A — site do Kaggle:** baixe [`solesensei/bdd100k`](https://www.kaggle.com/datasets/solesensei/bdd100k) pelo navegador.

**Opção B — Kaggle CLI** (não testado neste projeto):

```bash
pip install kaggle
# Coloque seu token em ~/.kaggle/kaggle.json (Kaggle → Settings → Create New Token)
kaggle datasets download solesensei/bdd100k -p codigo/
```

### Extração

Extraia **tudo** dentro de `codigo/bdd100k/`. A estrutura esperada é parecida com:

```
codigo/bdd100k/
└── bdd100k/
    ├── images/
    │   ├── 100k/                 ← ESTA é a pasta usada na detecção
    │   │   ├── train/   (~70.000 .jpg)
    │   │   ├── val/     (~10.000 .jpg)
    │   │   └── test/    (~20.000 .jpg)
    │   └── 10k/                  ← subconjunto de segmentação (NÃO usar)
    └── labels/
        ├── bdd100k_labels_images_train.json
        └── bdd100k_labels_images_val.json
```

> [!IMPORTANT]
> Na execução anterior a extração ficou incompleta (1.160 de 70.000 imagens de treino) e nenhum aviso foi exibido. **Sempre confira a contagem antes de treinar:**

```bash
ls codigo/bdd100k/bdd100k/images/100k/train | wc -l    # esperado: ~70000
ls codigo/bdd100k/bdd100k/images/100k/val   | wc -l    # esperado: ~10000
```

No Windows (PowerShell):

```powershell
(Get-ChildItem codigo\bdd100k\bdd100k\images\100k\train).Count
```

> [!WARNING]
> Se existirem as pastas `images/10k/` ou `bdd100k_seg/`, o código pode escolher a pasta errada ([problema 6](05-problemas-conhecidos.md#6-detecção-de-pastas)). Até a correção, a forma mais segura é deixar em `codigo/bdd100k/` apenas `images/100k/` e `labels/`.

---

## Passo 3 — Pesos pré-treinados

Não é preciso baixar nada: na primeira execução o Ultralytics baixa `yolov8m.pt` e `yolov10m.pt` automaticamente para a pasta de trabalho.

## Passo 4 — Weights & Biases (opcional)

```bash
wandb login
```

Sem login, o treino segue normalmente e mostra o aviso `WandB indisponível`.

---

## O que fica fora do Git

| Item | Como obter |
|---|---|
| `.venv/` | Passo 1 |
| `codigo/bdd100k/` (dataset original) | Passo 2 |
| `yolov8m.pt`, `yolov10m.pt` | Baixados automaticamente no 1º treino |
| `codigo/data/bdd100k/` (labels YOLO + `data.yaml`) | Gerados pelo `train.py --source` |
| `codigo/runs/`, `codigo/results/`, `codigo/wandb/` | Gerados ao treinar e comparar |

---

[← Visão geral](01-visao-geral.md) · [Índice](../README.md) · [Próximo: Como executar →](03-como-executar.md)
