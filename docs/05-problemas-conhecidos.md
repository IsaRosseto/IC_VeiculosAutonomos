# 5. Problemas conhecidos

[← Arquitetura do código](04-arquitetura-do-codigo.md) · [Índice](../README.md) · [Próximo: Prompts de correção →](06-prompts-de-correcao.md)

---

O código roda, mas tem os problemas abaixo. Cada um tem um prompt pronto para correção em [06-prompts-de-correcao.md](06-prompts-de-correcao.md) com o mesmo número.

**Legenda:** 🔴 Crítico (invalida resultados ou impede a execução) · 🟠 Alto · 🟡 Médio · 🟢 Baixo

## Resumo

| # | Gravidade | Problema | Onde | Status |
|---|---|---|---|---|
| [1](#1-caminhos-absolutos) | 🔴 Crítico | Caminhos absolutos em `data.yaml`, nos symlinks e nos checkpoints | `train.py` → `create_data_yaml`, `link_images` | Aberto |
| [2](#2-dataset-incompleto) | 🔴 Crítico | Dataset incompleto: 1.160 de 70.000 imagens de treino | dados locais | Aberto |
| [3](#3-resultados-inválidos) | 🔴 Crítico | Resultados inválidos: mAP50 0,17, recall 0,003, v10 não treinado | `runs/`, `check_and_resume` | Aberto |
| [4](#4-total_mem) | 🔴 Crítico | `total_mem` não existe (o certo é `total_memory`): crash em máquina com GPU | `scripts/train_yolo.py:83-84` | Aberto |
| [5](#5-cache-em-ram) | 🟠 Alto | `cache="ram"` e `workers=12` fixos: falta de RAM com o dataset completo | `train.py:550,553,589,633,643` / `scripts/train_yolo.py:308` | Aberto |
| [6](#6-detecção-de-pastas) | 🟠 Alto | A busca de pastas pode escolher `10k/train` em vez de `100k/train` | `find_source_paths` (nos 2 arquivos) | Aberto |
| [7](#7-duplicação) | 🟡 Médio | Código triplicado; o teste copia a função em vez de importar | `train.py`, `scripts/*` | Aberto |
| [8](#8-input-interativo) | 🟡 Médio | `input()` interativo trava ou pula treinos sem ninguém olhando | `check_and_resume`, `step1_check_gpu` | Aberto |
| [9](#9-overwrite) | 🟡 Médio | "Overwrite" não apaga o run anterior | `train_single_model`, `train_model` | Aberto |
| [10](#10-torchload) | 🟡 Médio | Patch global `torch.load(weights_only=False)` é inseguro e desnecessário | `step1_check_gpu`, `patch_torch_load` | Aberto |
| [11](#11-wandb) | 🟡 Médio | WandB inicializado na mão e também pelo Ultralytics; `resume="allow"` sem `id` | `init_wandb_safe`, `init_wandb` | Aberto |
| [12](#12-reprodutibilidade) | 🟢 Baixo | Versões não fixadas, sem seed, validação repetida, `--conf` ignorado | `requirements.txt`, `step5_compare` | Aberto |
| [13](#13-bbox) | 🟢 Baixo | O clamp da bbox é feito depois da conversão, e a caixa pode ficar deslocada | `convert_bbox_to_yolo` | Aberto |
| [14](#14-documentação) | 🟢 Baixo | Faltava documentação | — | ✅ Resolvido (esta pasta `docs/`) |

> Ao corrigir um problema, mude a coluna **Status** para `✅ Resolvido` e registre a mudança no fim deste arquivo, em [Histórico de correções](#histórico-de-correções).

---

## Detalhes

### 1. Caminhos absolutos

O `data.yaml` gerado grava `path: /home/<usuario>/Documents/...`. As pastas `images/train` e `images/val` são symlinks absolutos, e os checkpoints `last.pt` guardam os caminhos da máquina original. Em outro PC, ou depois de mover a pasta, tudo quebra. Na máquina original esses links já estão quebrados, porque os dados foram movidos.

Exemplo real em [`resultados/anteriores/treino_yolov8m/args.yaml`](../resultados/anteriores/treino_yolov8m/args.yaml): `data: /home/gugzica3/Documents/IC_isabella/IA-teste-main/data/bdd100k/data.yaml`.

> A pasta `IA-teste-main/` foi renomeada para `codigo/` nesta organização. Como o código calcula os caminhos a partir da própria localização, a renomeação não muda o comportamento — mas qualquer `data.yaml` ou `last.pt` antigo continuará apontando para o caminho velho.

### 2. Dataset incompleto

Só 1.160 imagens de treino foram extraídas, contra cerca de 70.000 esperadas. A validação tem as 10.000. O modelo treinou com cerca de **1,6%** dos dados. As labels `.txt` foram geradas para as 70 mil imagens (a partir do JSON), então a conversão não acusou nada.

Como conferir: veja [Instalação → Extração](02-instalacao.md#extração).

### 3. Resultados inválidos

O `results.csv` tem 13 épocas, mAP50 ≈ 0,16 e recall ≈ 0,003, o que indica um modelo praticamente inútil (na comparação visual, as imagens não têm nenhuma caixa desenhada). Segundo o registro da máquina original, um `args.yaml` mais recente (não incluído aqui) mostrava `data: coco8.yaml` e um `resume` apontando para outro caminho (`/home/.../Documents/IA-teste-main/...`), ou seja, alguma execução rodou no dataset de demonstração do Ultralytics. O YOLOv10m aparece como `skipped` em todas as execuções. Cada época levou cerca de 55 minutos.

Análise completa em [resultados/README.md](../resultados/README.md).

> Esses registros são das primeiras execuções. Os treinos finais do projeto, com o dataset completo, estão no [artigo SICFEI](08-artigo-sicfei.md) e não têm esse problema. Os problemas de código desta lista continuam valendo para o código de `codigo/`.

### 4. `total_mem`

O atributo de `torch.cuda.get_device_properties()` se chama `total_memory`. Do jeito que está, `scripts/train_yolo.py` dá `AttributeError` assim que encontra uma GPU. O `train.py` já usa o nome certo.

### 5. Cache em RAM

Cerca de 70 mil imagens em 640×360×3 bytes dão cerca de 48 GB. O valor está fixo e não pode ser mudado pela linha de comando. `workers=12` também está fixo no `train.py`, e o WandB registra esses valores fixos em vez dos reais.

Curiosidade: o `args.yaml` guardado mostra `cache: false`, `workers: 8` e `cos_lr: false`, valores diferentes do código atual. O treino registrado foi feito com uma versão anterior do código.

### 6. Detecção de pastas

O `os.walk` percorre `images/10k/train`, `images/100k/train` e as pastas de `bdd100k_seg`, e o código fica com a **última** que encontrar. A ordem não é garantida. Com os JSONs acontece o mesmo: pode pegar um JSON de segmentação ou de *drivable area*.

### 7. Duplicação

`convert_bbox_to_yolo`, `find_source_paths`, `draw_detections`, `MODEL_CONFIGS` e a lista de classes existem em 2 ou 3 lugares (mapa completo em [Arquitetura do código](04-arquitetura-do-codigo.md#scripts-em-scripts)). Se um for corrigido, os outros continuam errados. O `test_convert_bbox.py` testa uma **cópia** da função.

### 8. `input()` interativo

Rodar com `nohup`, `tmux` desanexado, SLURM ou Colab trava ou cancela o treino. Com `EOFError` o código cancela em silêncio, e o modelo aparece como `skipped`.

### 9. Overwrite

O código usa `exist_ok=True` sem apagar a pasta, então `epoch*.pt`, `results.csv` e as imagens do run antigo se misturam com o novo. Um `best.pt` antigo pode sobreviver e ser usado na comparação.

### 10. `torch.load`

Forçar `weights_only=False` em todo o processo permite que qualquer `.pt` execute código arbitrário. O Ultralytics 8.3 ou mais novo já trata isso internamente. Enquanto não for corrigido: **só carregue arquivos `.pt` de fontes confiáveis.**

### 11. WandB

O Ultralytics tem um callback próprio de WandB, ativado com `yolo settings wandb=True`. Somado ao `wandb.init` manual, isso pode gerar runs duplicados ou misturados. `resume="allow"` sem um `id` fixo cria um run novo a cada execução.

### 12. Reprodutibilidade

O `requirements.txt` só usa `>=`. O comentário diz "Windows", mas o projeto rodou no Linux. O `random.sample` não tem seed, então as imagens comparadas mudam a cada execução. A etapa 5 valida de novo modelos que já foram validados no fim do treino. `conf` e `iou` estão fixos na comparação visual (o `inference_compare.py` aceita `--conf`, mas não usa o valor).

### 13. Bbox

Se `x1 < 0` ou `x2 > 1280`, limitar o centro e a largura separadamente gera uma caixa deslocada. O correto é cortar x1, y1, x2, y2 em `[0, W]` e `[0, H]` primeiro e só depois converter. Além disso, `IMG_WIDTH` e `IMG_HEIGHT` são fixos, sem conferir o tamanho real da imagem.

### 14. Documentação

Resolvido com o `README.md` da raiz e a pasta `docs/`. Falta manter a documentação atualizada conforme as correções forem feitas ([prompt 14](06-prompts-de-correcao.md#prompt-14--manter-o-readme-e-a-documentação)).

---

## Histórico de correções

| Data | Problema | O que mudou | Commit |
|---|---|---|---|
| 2026-10 | 14 | Repositório organizado: código em `codigo/`, documentação em `docs/`, resultados em `resultados/` | #2 |
| 2026-10 | — | README atualizado com o artigo SICFEI (resultados finais) em `docs/08-artigo-sicfei.md` | — |

---

[← Arquitetura do código](04-arquitetura-do-codigo.md) · [Índice](../README.md) · [Próximo: Prompts de correção →](06-prompts-de-correcao.md)
