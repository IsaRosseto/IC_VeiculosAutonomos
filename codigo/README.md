# `codigo/` — código-fonte do pipeline

[← Índice do repositório](../README.md)

Código da **fase preliminar** do projeto (YOLOv8m vs YOLOv10m, 10 classes originais do BDD100K). O código dos treinos finais do [artigo](../docs/08-artigo-sicfei.md) (YOLOv11m, YOLO26, classes em português) e o do módulo de faixa ainda não estão aqui.

Esta pasta era chamada `IA-teste-main/` na versão original. Todos os comandos abaixo devem ser executados **de dentro desta pasta**.

```
codigo/
├── train.py                  Pipeline completo, 5 etapas em sequência
├── requirements.txt          Dependências Python
└── scripts/
    ├── convert_bdd100k.py    Etapa 2 isolada: JSON do BDD100K → labels YOLO + data.yaml
    ├── train_yolo.py         Etapas 3–4 isoladas: treino do v8, do v10 ou dos dois
    ├── inference_compare.py  Etapa 5 isolada: métricas + imagens lado a lado
    └── test_convert_bbox.py  Teste da fórmula de conversão de caixas
```

## Qual arquivo usar?

| Situação | Comando |
|---|---|
| Primeira vez, fazer tudo | `python train.py --source ./bdd100k` |
| Já converti, quero treinar e comparar | `python train.py --skip-convert` |
| Só quero (re)gerar a comparação | `python train.py --skip-convert --skip-train` |
| Só converter o dataset | `python scripts/convert_bdd100k.py --source ./bdd100k` |
| Treinar só um modelo | `python scripts/train_yolo.py --version v8` ⚠️ [problema 4](../docs/05-problemas-conhecidos.md#4-total_mem) |
| Comparar com imagens próprias | `python scripts/inference_compare.py --source pasta/ --skip-val` |
| Testar a conversão de caixas | `python scripts/test_convert_bbox.py` |

Todas as opções de cada script estão em [docs/03-como-executar.md](../docs/03-como-executar.md). O funcionamento interno, função por função, está em [docs/04-arquitetura-do-codigo.md](../docs/04-arquitetura-do-codigo.md).

## Pastas criadas aqui durante a execução

Nenhuma delas vai para o Git (veja o [`.gitignore`](../.gitignore)).

| Pasta | Criada por | Conteúdo |
|---|---|---|
| `bdd100k/` | você | Dataset original extraído do Kaggle |
| `data/bdd100k/` | etapa 2 | `data.yaml`, `labels/{train,val}/*.txt`, links `images/{train,val}` |
| `runs/detect/yolov8m_bdd100k/` | etapa 3 | Pesos, `results.csv`, `args.yaml`, gráficos do Ultralytics |
| `runs/detect/yolov10m_bdd100k/` | etapa 4 | Idem |
| `results/comparison/<data_hora>/` | etapa 5 | Gráfico, CSVs e `visual/` com as imagens lado a lado |
| `wandb/` | etapas 3–4 | Logs do Weights & Biases |
| `yolov8m.pt`, `yolov10m.pt` | etapas 3–4 | Pesos pré-treinados baixados pelo Ultralytics |

> Os resultados que valem a pena guardar (CSVs, imagens, `args.yaml`) podem ser copiados para [`../resultados/`](../resultados/) com uma pasta nova e uma linha na tabela do [resultados/README.md](../resultados/README.md).
