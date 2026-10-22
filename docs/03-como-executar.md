# 3. Como executar

[← Instalação](02-instalacao.md) · [Índice](../README.md) · [Próximo: Arquitetura do código →](04-arquitetura-do-codigo.md)

---

Há duas formas de rodar:

1. **Pipeline completo** com `train.py`: faz tudo em sequência (recomendado).
2. **Scripts separados** em `scripts/`: cada etapa isolada, útil para repetir só uma parte.

Todos os comandos abaixo partem da pasta `codigo/`:

```bash
cd codigo
```

---

## Opção 1 — Pipeline completo (`train.py`)

```bash
# Primeira execução: converte o dataset e treina os dois modelos
python train.py --source ./bdd100k --batch 8

# Dataset já convertido antes
python train.py --skip-convert

# Só comparar modelos já treinados (gera gráfico, CSV e imagens)
python train.py --skip-convert --skip-train

# Teste rápido com menos épocas e mais imagens na comparação
python train.py --skip-convert --epochs 50 --num-images 30
```

### Opções do `train.py`

| Opção | Padrão | Descrição |
|---|---|---|
| `--source PASTA` | — | Pasta do BDD100K extraído. Obrigatória na primeira vez |
| `--skip-convert` | desligado | Pula a etapa 2 (conversão). Exige que `data/bdd100k/data.yaml` exista |
| `--skip-train` | desligado | Pula as etapas 3 e 4 (treino) e vai direto para a comparação |
| `--epochs N` | `100` | Número de épocas de cada modelo |
| `--batch N` | `8` | Tamanho do batch (8 para 6 GB de VRAM; use 4 se faltar memória) |
| `--num-images N` | `20` | Quantas imagens de validação entram na comparação visual |

### O que acontece em cada etapa

| Etapa | Função | O que faz | Interação |
|---|---|---|---|
| 1 | `step1_check_gpu` | Mostra versões de Python/PyTorch/Ultralytics e dados da GPU | Sem GPU, pergunta `Continuar sem GPU? [s/N]` |
| 2 | `step2_convert_dataset` | Acha os JSONs e as imagens, cria links em `data/bdd100k/images/`, gera `labels/*.txt` e `data.yaml`. Pula se já houver mais de 1.000 labels de treino | — |
| 3 | `step3_train_v8` | Treina o YOLOv8m em `runs/detect/yolov8m_bdd100k/` | Se o run já existe, mostra o menu abaixo |
| 4 | `step4_train_v10` | Treina o YOLOv10m em `runs/detect/yolov10m_bdd100k/` | Idem |
| 5 | `step5_compare` | Valida os `best.pt`, gera gráfico, CSVs e imagens lado a lado | — |

### Menu de run existente (etapas 3 e 4)

Se a pasta do run já existir, o script pergunta:

```
[1] 🔄 Resume — continuar de onde parou
[2] 🗑️  Overwrite — recomeçar do zero
[3] ❌ Cancelar
```

- **Resume** continua a partir de `weights/last.pt`.
- **Overwrite** recomeça, mas **não apaga** os arquivos antigos da pasta ([problema 9](05-problemas-conhecidos.md#9-overwrite)).
- **Cancelar** pula o modelo (o status fica `skipped`).

> [!CAUTION]
> Rodando sem terminal interativo (`nohup`, SLURM, Colab), o `input()` falha e o código trata isso como **Cancelar**: o treino é pulado sem aviso. Use um terminal interativo ou `tmux` anexado ([problema 8](05-problemas-conhecidos.md#8-input-interativo)).

### Interromper e continuar

Apertar `Ctrl+C` durante o treino salva o estado. Na próxima execução, escolha **[1] Resume**.

---

## Opção 2 — Scripts separados

Também partindo de `codigo/`.

### `scripts/convert_bdd100k.py` — conversão

```bash
python scripts/convert_bdd100k.py --source ./bdd100k
python scripts/convert_bdd100k.py --source ./bdd100k --output ./data/bdd100k
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--source PASTA` | obrigatória | Pasta raiz do BDD100K extraído |
| `--output PASTA` | `codigo/data/bdd100k` | Onde gravar o dataset YOLO |

Diferente do `train.py`, este script sempre converte de novo, mesmo que os labels já existam.

### `scripts/train_yolo.py` — treino

```bash
python scripts/train_yolo.py --version v8
python scripts/train_yolo.py --version v10
python scripts/train_yolo.py --version both --batch 8 --workers 8
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--version {v8,v10,both}` | obrigatória | Qual modelo treinar |
| `--epochs N` | `100` | Número de épocas |
| `--imgsz N` | `640` | Tamanho da imagem de entrada |
| `--batch N` | `16` | Batch size (padrão pensado para RTX 4050; **diferente** do `train.py`) |
| `--workers N` | `8` | Processos do dataloader |
| `--save-period N` | `5` | Salva checkpoint a cada N épocas |

> [!WARNING]
> Este script dá `AttributeError` em qualquer máquina com GPU por causa de `total_mem` ([problema 4](05-problemas-conhecidos.md#4-total_mem)). Até a correção, use o `train.py`.

### `scripts/inference_compare.py` — comparação

```bash
python scripts/inference_compare.py
python scripts/inference_compare.py --num-images 30
python scripts/inference_compare.py --skip-val --source caminho/para/imagens
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--num-images N` | `10` | Imagens na comparação visual |
| `--source PASTA` | `data/bdd100k/images/val` | Pasta com imagens para a comparação visual |
| `--conf X` | `0.25` | Limiar de confiança (⚠️ aceito, mas não usado no `predict`: [problema 12](05-problemas-conhecidos.md#12-reprodutibilidade)) |
| `--skip-val` | desligado | Pula a validação formal e só gera as imagens |

### `scripts/test_convert_bbox.py` — teste da conversão

```bash
python scripts/test_convert_bbox.py
```

Roda 6 verificações da fórmula `x1,y1,x2,y2 → x_centro,y_centro,largura,altura`. Saída esperada: `✅ TODOS OS 6 TESTES PASSARAM`. O teste usa uma **cópia** da função, não a do código real ([problema 7](05-problemas-conhecidos.md#7-duplicação)).

---

## Onde ficam as saídas

```
codigo/
├── data/bdd100k/
│   ├── data.yaml                 ← configuração do dataset para o Ultralytics
│   ├── images/train  → link para .../images/100k/train
│   ├── images/val    → link para .../images/100k/val
│   └── labels/{train,val}/*.txt  ← uma linha por objeto: "classe xc yc w h"
├── runs/detect/
│   ├── yolov8m_bdd100k/
│   │   ├── weights/best.pt       ← melhor época (maior fitness na validação)
│   │   ├── weights/last.pt       ← última época (usado no resume)
│   │   ├── weights/epochN.pt     ← a cada 5 épocas
│   │   ├── args.yaml             ← todos os parâmetros usados
│   │   ├── results.csv           ← métricas por época
│   │   └── *.png                 ← curvas, matriz de confusão etc.
│   └── yolov10m_bdd100k/         ← mesma estrutura
└── results/comparison/<AAAAMMDD_HHMMSS>/
    ├── metrics_comparison.png    ← gráfico de barras (só se os 2 modelos existirem)
    ├── metrics_comparison.csv    ← Model, mAP50, mAP50-95, Precision, Recall
    ├── training_summary.csv      ← status de cada treino (completed/skipped/...)
    └── visual/compare_NNN_<imagem>.jpg  ← YOLOv8m | YOLOv10m lado a lado
```

### Como ler os resultados

- **`results.csv`**: uma linha por época. As colunas `metrics/mAP50(B)` e `metrics/recall(B)` são as mais importantes. As perdas (`*_loss`) devem cair ao longo do tempo. A coluna `time` é o tempo acumulado em segundos.
- **`training_summary.csv`**: `completed` = terminou; `interrupted` = Ctrl+C; `skipped` = cancelado no menu; `failed` = erro (veja o terminal).
- **Imagens `visual/`**: cada caixa traz a classe e a confiança. Imagem sem caixas = o modelo não detectou nada acima de 0,25 de confiança.

Os valores de referência de um treino saudável e a análise das execuções anteriores estão em [resultados/README.md](../resultados/README.md).

---

## Usando o VS Code

O repositório inclui [`.vscode/launch.json`](../.vscode/launch.json) com a configuração **"Train BDD100K"**, que roda `codigo/train.py --source codigo --batch 4` usando o Python de `.venv/`. Abra a **raiz do repositório** no VS Code e aperte `F5`.

O caminho do intérprete (`.venv/bin/python3`) é o do Linux. No Windows, troque por `${workspaceFolder}/.venv/Scripts/python.exe` em `launch.json` e `settings.json`.

---

## Solução de problemas

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `torch.cuda.is_available()` é `False` | PyTorch sem CUDA ou sem driver | Veja [Instalação → GPU](02-instalacao.md#conferir-se-a-gpu-está-funcionando) |
| `CUDA out of memory` | Batch grande demais | `--batch 4` |
| Computador trava / swap no início do treino | `cache="ram"` com o dataset inteiro | [Problema 5](05-problemas-conhecidos.md#5-cache-em-ram) |
| `AttributeError: ... total_mem` | Bug no `train_yolo.py` | Use o `train.py` ou aplique o [prompt 4](06-prompts-de-correcao.md#prompt-4--total_mem--total_memory) |
| Treino aparece como `skipped` sem você cancelar | `input()` sem terminal | [Problema 8](05-problemas-conhecidos.md#8-input-interativo) |
| `data.yaml não encontrado` / caminhos de outra máquina | Caminhos absolutos | Apague `data/bdd100k/data.yaml` e rode com `--source` ([problema 1](05-problemas-conhecidos.md#1-caminhos-absolutos)) |
| Symlink falha no Windows | Falta permissão | O código tenta *junction* e depois copia as imagens (lento, ocupa mais disco) |
| mAP e recall perto de zero | Dataset incompleto ou errado | Confira a contagem de imagens ([problema 2](05-problemas-conhecidos.md#2-dataset-incompleto)) |

---

[← Instalação](02-instalacao.md) · [Índice](../README.md) · [Próximo: Arquitetura do código →](04-arquitetura-do-codigo.md)
