# 6. Prompts de correção

[← Problemas conhecidos](05-problemas-conhecidos.md) · [Índice](../README.md) · [Próximo: Glossário →](07-glossario.md)

---

Cada prompt corrige um dos [problemas conhecidos](05-problemas-conhecidos.md) (mesma numeração). Eles são independentes e foram escritos para colar num assistente de código (Claude Code, Cursor, ChatGPT etc.) **aberto na pasta `codigo/`**. Todos seguem a mesma estrutura: **papel → contexto → tarefa → restrições → critério de aceite → formato da resposta**.

## Ordem recomendada

**7 → 4 → 1 → 6 → 13 → 5 → 8 → 9 → 10 → 11 → 12 → 2 → 3 → 14**

Fazer o 7 primeiro (unificar o código) evita corrigir a mesma coisa em três lugares.

| Prompt | Problema | Gravidade |
|---|---|---|
| [1](#prompt-1--caminhos-absolutos--portáveis) | Caminhos absolutos | 🔴 |
| [2](#prompt-2--dataset-incompleto-validação-de-integridade) | Dataset incompleto | 🔴 |
| [3](#prompt-3--resultados-inválidos-re-treino-controlado-e-sanity-check) | Resultados inválidos | 🔴 |
| [4](#prompt-4--total_mem--total_memory) | `total_mem` | 🔴 |
| [5](#prompt-5--cache-e-workers-configuráveis-com-proteção-de-ram) | Cache em RAM | 🟠 |
| [6](#prompt-6--detecção-determinística-das-pastas-do-dataset) | Detecção de pastas | 🟠 |
| [7](#prompt-7--eliminar-a-duplicação-de-código) | Duplicação | 🟡 |
| [8](#prompt-8--modo-não-interativo) | `input()` interativo | 🟡 |
| [9](#prompt-9--overwrite-que-de-fato-recomeça-do-zero) | Overwrite | 🟡 |
| [10](#prompt-10--remover-o-patch-inseguro-de-torchload) | `torch.load` | 🟡 |
| [11](#prompt-11--integração-correta-com-o-wandb) | WandB | 🟡 |
| [12](#prompt-12--reprodutibilidade) | Reprodutibilidade | 🟢 |
| [13](#prompt-13--conversão-de-bbox-correta) | Bbox | 🟢 |
| [14](#prompt-14--manter-o-readme-e-a-documentação) | Documentação | 🟢 |

> Depois de aplicar uma correção: rode `python scripts/test_convert_bbox.py` (ou `pytest`, se o prompt 7 já tiver sido aplicado), atualize o status em [05-problemas-conhecidos.md](05-problemas-conhecidos.md) e faça um commit com a mensagem `fix: problema N — <resumo>`.

---

### Prompt 1 — Caminhos absolutos → portáveis

```
Você é um engenheiro de ML especialista em Ultralytics YOLO e em portabilidade de projetos Python entre Linux e Windows.

CONTEXTO
Projeto: pipeline YOLOv8m vs YOLOv10m no BDD100K. Arquivos: train.py, scripts/convert_bdd100k.py, scripts/train_yolo.py, scripts/inference_compare.py.
Problema: o projeto não funciona em outra máquina nem depois de mover a pasta, porque:
- create_data_yaml() grava `path:` com caminho ABSOLUTO (ex.: /home/usuario/...).
- link_images() / setup_directory_structure() criam symlinks ABSOLUTOS de data/bdd100k/images/{train,val} para a pasta original do dataset.
- O resume usa last.pt, que guarda nos train_args os caminhos absolutos da máquina original.

TAREFA
1. Faça o data.yaml ser sempre (re)gerado em tempo de execução a partir de PROJECT_ROOT, para que nunca dependa de um caminho salvo antes. Antes de treinar, se o `path` do data.yaml existente não bater com o DATA_DIR atual, regenere o arquivo.
2. Crie os symlinks com caminho RELATIVO (os.path.relpath(src, dst.parent)). No Windows, mantenha o fallback para junction (mklink /J), que exige absoluto, e depois cópia.
3. Antes de criar o link, detecte e remova symlinks QUEBRADOS (dst.is_symlink() and not dst.exists()).
4. No resume, antes de model.train(resume=True), carregue o checkpoint, confira se ckpt["train_args"]["data"] existe. Se não existir, avise o usuário e sobrescreva com o DATA_YAML atual, usando a API do Ultralytics (model.train(resume=True, data=...) se suportado na versão instalada; senão, edite os train_args do checkpoint e salve uma cópia).
5. Adicione o arquivo data/bdd100k/data.yaml ao .gitignore (crie o .gitignore se não existir).

RESTRIÇÕES
- Não mude a interface de linha de comando existente.
- Use pathlib. Compatível com Python 3.10+ no Linux e no Windows.
- Não use bibliotecas novas.

CRITÉRIO DE ACEITE
- Copiar o projeto inteiro para outra pasta e rodar `python train.py --skip-convert` funciona sem editar nada.
- `readlink data/bdd100k/images/train` mostra um caminho relativo.

FORMATO DA RESPOSTA
Mostre um diff por arquivo, depois explique em 3-5 bullets o que mudou e por quê, e por fim os comandos para testar.
```

---

### Prompt 2 — Dataset incompleto: validação de integridade

```
Você é um engenheiro de dados para visão computacional.

CONTEXTO
Na última execução, o pipeline (train.py) treinou com apenas 1.160 das ~70.000 imagens de treino do BDD100K (images/100k/train), porque a extração do zip ficou incompleta. Nenhum aviso foi exibido. As labels .txt foram geradas para as 70k imagens (a partir do JSON), mas só 1.160 imagens existiam.

TAREFA
Crie uma função `validate_dataset(data_dir: Path, min_train: int, min_val: int) -> dict` em train.py, chamada no fim da conversão e no início do treino, que:
1. Conte as imagens em images/train e images/val (seguindo symlinks).
2. Conte as labels .txt correspondentes e calcule:
   - imagens sem label;
   - labels sem imagem (órfãs);
   - labels vazias;
   - porcentagem de cobertura em relação ao JSON (imagens esperadas = número de frames no JSON, quando disponível).
3. Valores esperados do BDD100K 100k: train = 70.000, val = 10.000. Se a cobertura for menor que 95%, mostre um AVISO VERMELHO grande, explicando que a extração está incompleta e como conferir (`ls images/100k/train | wc -l`). Peça confirmação, ou aborte se `--strict` for passado.
4. Adicione os argumentos `--strict` (aborta) e `--subset N` (permite treinar de propósito com um subconjunto, sem aviso).
5. Salve o relatório em data/bdd100k/dataset_report.json.

RESTRIÇÕES
- Não leia o conteúdo das imagens (só os nomes); precisa ser rápido com 80k arquivos.
- Use os.scandir para performance.

CRITÉRIO DE ACEITE
- Com 1.160/70.000 imagens, o script mostra o aviso e para com --strict.
- Com o dataset completo, imprime "✅ Dataset íntegro" e segue.

FORMATO DA RESPOSTA
Código da função, os pontos de chamada (diff) e um exemplo da saída no terminal nos dois cenários.
```

---

### Prompt 3 — Resultados inválidos: re-treino controlado e sanity check

```
Você é um pesquisador de ML que revisa experimentos de detecção de objetos.

CONTEXTO
Resultados atuais do YOLOv8m no BDD100K (runs/detect/yolov8m_bdd100k/results.csv): 13 épocas, mAP50=0.16, mAP50-95=0.10, precision=0.32, recall=0.003, ~53 min/época. O YOLOv10m nunca foi treinado (status "skipped"). O args.yaml mais recente mostra `data: coco8.yaml` e `resume:` apontando para um caminho de outra pasta, ou seja, uma execução de resume rodou no dataset de demonstração do Ultralytics. Recall de 0.003 com precision de 0.32 é anormal.

TAREFA
1. Adicione ao train.py um modo `--sanity` que:
   a. treina 3 épocas com fraction=0.05 (5% do dataset), imgsz=640, batch do usuário;
   b. confere no fim se mAP50 > 0.05 e recall > 0.05; se não, imprime um diagnóstico com as causas prováveis (labels desalinhadas com imagens, classes erradas, data.yaml errado, dataset errado);
   c. imprime o data.yaml EFETIVAMENTE usado pelo trainer (trainer.args.data) e aborta se não for o BDD100K.
2. Antes de qualquer resume, leia ckpt["train_args"]["data"] e recuse continuar se for diferente do DATA_YAML atual (ex.: coco8.yaml), oferecendo recomeçar do zero.
3. Adicione uma função que plota 4 imagens de treino com as labels desenhadas (a partir dos .txt), salvas em results/sanity/, para inspeção visual do alinhamento.
4. Documente em docs/03-como-executar.md (na raiz do repositório) o procedimento: sanity → treino completo v8 → treino completo v10 → comparação.

RESTRIÇÕES
- Use só a API pública do Ultralytics (model.train, model.val, trainer.args).
- Não mude os hiperparâmetros do treino completo.

CRITÉRIO DE ACEITE
- `python train.py --sanity` termina em menos de 30 min numa GPU de 6 GB e mostra PASS/FAIL claro.
- Resume com um checkpoint de coco8 é bloqueado com mensagem explicativa.

FORMATO DA RESPOSTA
Diff, explicação de cada verificação e checklist de como interpretar um FAIL.
```

---

### Prompt 4 — `total_mem` → `total_memory`

```
Você é um desenvolvedor Python revisando código PyTorch.

CONTEXTO
Em scripts/train_yolo.py, na função print_gpu_info() (linhas ~83-84), o código usa `torch.cuda.get_device_properties(0).total_mem`. Esse atributo NÃO existe; o correto é `total_memory`. Resultado: AttributeError em qualquer máquina COM GPU, antes do treino começar. Além disso, a "VRAM livre" é calculada com memory_allocated, que só mostra a memória alocada por este processo, e não a livre de verdade.

TAREFA
1. Troque `total_mem` por `total_memory`.
2. Calcule a VRAM livre real com `torch.cuda.mem_get_info(0)`, que retorna (free, total) em bytes, e mostre as duas.
3. Chame get_device_properties uma vez só e guarde numa variável.
4. Mostre um aviso se a VRAM total for menor que 6 GB, sugerindo batch=4 (mesmo comportamento do train.py).
5. Procure no projeto inteiro (grep) outros usos de `total_mem` e corrija todos.

RESTRIÇÕES
- Mudança mínima, sem refatorar o resto do arquivo.

CRITÉRIO DE ACEITE
- `python -c "from scripts.train_yolo import print_gpu_info; print_gpu_info()"` roda sem erro com e sem GPU.

FORMATO DA RESPOSTA
Só o diff e uma linha de explicação.
```

---

### Prompt 5 — `cache` e `workers` configuráveis com proteção de RAM

```
Você é um engenheiro de MLOps especialista em treinamento com Ultralytics em hardware limitado.

CONTEXTO
Em train.py (train_single_model) e scripts/train_yolo.py (train_model), model.train() é chamado com `cache="ram"` e (no train.py) `workers=12` FIXOS no código. O BDD100K tem ~70k imagens; em 640 px o cache em RAM precisa de ~48 GB, o que causa falta de memória ou swap infinito. O init_wandb_safe também registra "workers": 12 e "cache": "ram" fixos, não os valores reais.

TAREFA
1. Adicione os argumentos em ambos os scripts:
   --cache {ram,disk,none} (padrão: auto)
   --workers INT (padrão: min(8, os.cpu_count()))
   --imgsz INT (padrão 640, no train.py)
2. Implemente o modo "auto": estime a RAM necessária = n_imagens × imgsz × (imgsz×720/1280) × 3 bytes × 1.2. Compare com a RAM disponível (psutil.virtual_memory().available; se o psutil não estiver instalado, leia /proc/meminfo no Linux; se não der, use "disk"). Use "ram" se couber em 70% da RAM disponível; senão, "disk".
3. Imprima a decisão: "Cache: disk (precisaria de 48.3 GB, disponível 15.2 GB)".
4. Passe para o WandB os valores REAIS usados.
5. Converta "none" para cache=False no Ultralytics.

RESTRIÇÕES
- O psutil é opcional (try/except import). Se você usá-lo, adicione ao requirements.txt.
- Não mude os outros hiperparâmetros.

CRITÉRIO DE ACEITE
- Numa máquina com 16 GB de RAM e o dataset completo, auto escolhe "disk".
- `--cache ram` força ram, mostrando um aviso se não couber.

FORMATO DA RESPOSTA
Diff dos dois arquivos, a função de estimativa isolada e testável, e um exemplo da saída.
```

---

### Prompt 6 — Detecção determinística das pastas do dataset

```
Você é um engenheiro de software focado em robustez de pipelines de dados.

CONTEXTO
find_source_paths() (em train.py e scripts/convert_bdd100k.py) percorre o dataset com os.walk e guarda a ÚLTIMA pasta chamada "train"/"val" que tenha "image" no caminho e contenha .jpg. O BDD100K extraído tem várias pastas assim:
- bdd100k/images/100k/{train,val,test}  ← CORRETA (detecção, 70k/10k)
- bdd100k/images/10k/{train,val,test}   ← segmentação (7k/1k)
- bdd100k_seg/bdd100k/seg/images/{train,val}
A ordem do os.walk não é garantida, então o pipeline pode treinar com as imagens erradas. Com os JSONs acontece o mesmo: o critério "train" in name AND ("label" in name OR "det" in name) pode pegar JSONs de outras tarefas.

TAREFA
1. Reescreva find_source_paths para:
   a. Procurar primeiro caminhos conhecidos, em ordem de prioridade:
      - */images/100k/{split}
      - */images/{split}
   b. Para os JSONs, preferir os nomes exatos: bdd100k_labels_images_{split}.json, depois det_{split}.json, depois det_v2_{split}.json.
   c. Excluir explicitamente caminhos que contenham "10k", "seg", "drivable", "lane".
   d. Se houver mais de um candidato depois dos filtros, escolher o que tem MAIS imagens e listar todos os candidatos no log.
2. Adicione os argumentos opcionais --train-images, --val-images, --train-json e --val-json, que pulam a detecção automática.
3. Mova a função para um único lugar (se o Prompt 7 já foi aplicado, em bdd_common.py).
4. Escreva testes pytest com uma árvore falsa em tmp_path reproduzindo a estrutura acima, conferindo que 100k é sempre a escolhida.

RESTRIÇÕES
- Não leia o conteúdo dos JSONs para decidir (são grandes, ~1 GB).

CRITÉRIO DE ACEITE
- O teste passa em qualquer ordem de diretórios (use random.shuffle num mock de os.walk ou crie as pastas em ordens diferentes).

FORMATO DA RESPOSTA
Código novo da função, os testes e o diff dos argparse.
```

---

### Prompt 7 — Eliminar a duplicação de código

```
Você é um engenheiro de software sênior fazendo refatoração sem mudar o comportamento.

CONTEXTO
O projeto tem 4 scripts com muito código duplicado:
- train.py (orquestrador completo, ~980 linhas)
- scripts/convert_bdd100k.py, scripts/train_yolo.py, scripts/inference_compare.py
Duplicados em 2 ou 3 lugares: BDD100K_CLASSES, CLASS_TO_ID, IMG_WIDTH/HEIGHT, CLASS_COLORS, MODEL_CONFIGS, convert_bbox_to_yolo, find_source_paths, process_json(_labels), link_images/setup_directory_structure, create_data_yaml, init_wandb, finish_wandb, check/resume, draw_detections, find_best_weights, validação de métricas e gráfico/CSV.
scripts/test_convert_bbox.py REDEFINE convert_bbox_to_yolo em vez de importar, então não testa o código real.

TAREFA
1. Crie um pacote `bdd_yolo/` com os módulos:
   - config.py      (constantes, classes, cores, MODEL_CONFIGS, caminhos baseados em PROJECT_ROOT)
   - dataset.py     (convert_bbox_to_yolo, find_source_paths, process_json, link_images, create_data_yaml)
   - training.py    (resume, wandb, train_single_model)
   - compare.py     (find_best_weights, draw_detections, validate, chart, csv, visual)
   - env.py         (checagem de GPU)
2. train.py vira um orquestrador fino (<150 linhas) que importa do pacote. Os scripts/*.py viram wrappers finos que chamam as mesmas funções. Mantenha os mesmos argumentos de CLI e o mesmo texto de saída (banners incluídos).
3. Quando duas versões de uma função forem diferentes, liste as diferenças antes de unificar e escolha a mais correta, justificando (ex.: train.py tem junction→symlink→cópia; convert_bdd100k.py tem symlink→junction→cópia).
4. Converta test_convert_bbox.py para pytest em tests/test_dataset.py, IMPORTANDO de bdd_yolo.dataset. Mantenha os 6 casos atuais.
5. Adicione um pyproject.toml mínimo ou instrua `pip install -e .` para os imports funcionarem.

RESTRIÇÕES
- Zero mudança de comportamento nesta etapa (as correções de bugs ficam para outros prompts).
- Sem dependências novas além de pytest (dev).

CRITÉRIO DE ACEITE
- `pytest` passa.
- `python train.py --help` e `python scripts/train_yolo.py --help` mostram os mesmos argumentos de antes.
- `grep -rn "def convert_bbox_to_yolo"` retorna exatamente 1 resultado.

FORMATO DA RESPOSTA
1) Tabela com as diferenças encontradas entre as versões duplicadas e a decisão tomada. 2) A árvore de arquivos nova. 3) O código de cada arquivo. 4) Os comandos de verificação.
```

---

### Prompt 8 — Modo não interativo

```
Você é um engenheiro de MLOps acostumado com treinos longos em servidores (SSH, tmux, SLURM, Colab).

CONTEXTO
train.py e scripts/train_yolo.py usam input() em:
- step1_check_gpu: "Continuar sem GPU? [s/N]"
- check_and_resume / get_resume_decision: menu Resume/Overwrite/Cancelar
Rodando com nohup, SLURM ou num pipeline, o input() gera EOFError, que o código trata como "cancel", e o treino é PULADO em silêncio. Em tmux desanexado, ele fica esperando para sempre.

TAREFA
1. Adicione os argumentos:
   --on-existing {ask,resume,overwrite,skip} (padrão: ask)
   --allow-cpu   (aceita CPU sem perguntar)
   -y / --yes    (atalho: on-existing=resume se houver last.pt, senão overwrite; e allow-cpu)
2. Se sys.stdin.isatty() for False e on-existing=ask, NÃO use input(): mostre um erro claro dizendo qual flag usar e saia com código 2.
3. Remova o tratamento silencioso de EOFError → cancel.
4. Registre no log a decisão tomada e o motivo ("resume automático: --on-existing resume").
5. Atualize o ../.vscode/launch.json e o ../docs/03-como-executar.md com exemplos (`nohup python train.py --skip-convert -y > train.log 2>&1 &`).

RESTRIÇÕES
- O comportamento padrão no terminal interativo continua o mesmo de hoje.

CRITÉRIO DE ACEITE
- `echo | python train.py --skip-convert` termina com uma mensagem explicativa e código 2, sem pular o treino em silêncio.
- `python train.py --skip-convert -y < /dev/null` treina ou resume sem perguntar nada.

FORMATO DA RESPOSTA
Diff, tabela "flag → comportamento" e os comandos de teste.
```

---

### Prompt 9 — Overwrite que de fato recomeça do zero

```
Você é um desenvolvedor Python cuidadoso com operações destrutivas de arquivos.

CONTEXTO
Em train.py (train_single_model) e scripts/train_yolo.py (train_model), escolher "Overwrite" só segue com model.train(..., exist_ok=True). A pasta runs/detect/<run_name> NÃO é limpa: epoch*.pt, results.csv, best.pt e as imagens do run antigo se misturam com o novo, e um best.pt antigo pode sobreviver e ser usado na comparação.

TAREFA
1. No overwrite, ANTES de treinar, mova a pasta antiga para runs/detect/_archive/<run_name>_<timestamp>/ (não apague; é mais seguro). Adicione a flag --delete-old para apagar de verdade com shutil.rmtree.
2. Confirme que o destino do arquivamento não existe e que a pasta movida está dentro de RUNS_DIR (proteção contra apagar o caminho errado: use Path.resolve() e is_relative_to).
3. Depois de mover, treine com exist_ok=False para garantir que a pasta é nova.
4. Mostre o espaço liberado/arquivado em GB.
5. Atualize o texto do menu: "[2] Overwrite — arquiva o run anterior em _archive/ e recomeça".

RESTRIÇÕES
- Nunca apague nada fora de RUNS_DIR.
- Compatível com Windows (no Windows, shutil.move entre volumes copia; trate PermissionError com mensagem clara).

CRITÉRIO DE ACEITE
- Depois de um overwrite, runs/detect/yolov8m_bdd100k/ só contém arquivos do novo treino, e o antigo está em _archive/.

FORMATO DA RESPOSTA
Diff e explicação das salvaguardas.
```

---

### Prompt 10 — Remover o patch inseguro de `torch.load`

```
Você é um engenheiro de segurança em ML (supply chain de modelos).

CONTEXTO
train.py (step1_check_gpu) e scripts/train_yolo.py (patch_torch_load) substituem torch.load globalmente para forçar weights_only=False. Isso desativa a proteção do PyTorch 2.6+ contra pickle malicioso para QUALQUER .pt carregado no processo. O Ultralytics >= 8.3 (requirements já exige) carrega os próprios checkpoints com segurança sem esse patch. Versões instaladas: torch 2.11, ultralytics 8.4.

TAREFA
1. Remova o monkeypatch dos dois arquivos.
2. Teste mentalmente/confira na documentação se YOLO("yolov8m.pt"), YOLO("yolov10m.pt"), YOLO(last.pt) e model.train(resume=True) funcionam sem o patch nas versões citadas. Se algum caso precisar, use a solução pontual: torch.serialization.add_safe_globals([...]) com as classes necessárias, ou um context manager que aplica weights_only=False SÓ nessa chamada e SÓ para arquivos dentro de RUNS_DIR ou dos pesos oficiais.
3. Aumente a versão mínima no requirements.txt se for preciso (ex.: ultralytics>=8.3.50) e justifique.
4. Adicione em ../docs/05-problemas-conhecidos.md (problema 10) uma nota: "só carregue .pt de fontes confiáveis".

RESTRIÇÕES
- Não quebre o resume de checkpoints existentes.

CRITÉRIO DE ACEITE
- `grep -rn "_original_load\|weights_only" .` não encontra patch global.
- Treino novo e resume funcionam.

FORMATO DA RESPOSTA
Diff, explicação do risco removido e passos de teste.
```

---

### Prompt 11 — Integração correta com o WandB

```
Você é um especialista em rastreamento de experimentos com Weights & Biases e Ultralytics.

CONTEXTO
train.py (init_wandb_safe) e scripts/train_yolo.py (init_wandb) chamam wandb.init() manualmente antes de model.train(). O Ultralytics também tem um callback de WandB (ativado por `yolo settings wandb=True`) que cria o próprio run. Resultados: runs duplicados ou métricas em runs separados. Além disso:
- resume="allow" sem `id` fixo cria um run NOVO a cada execução (o resume não continua o mesmo gráfico);
- o config registra valores fixos (workers=12, cache=ram) e não os reais;
- existem várias pastas wandb/offline-run-* que nunca foram sincronizadas.

TAREFA
1. Escolha UMA abordagem e justifique. Recomendado: usar o callback nativo do Ultralytics (settings.update({"wandb": True})), passando project e name pelo model.train(project=..., name=...), e remover o wandb.init manual. Se mantiver o manual, desative o nativo e registre as métricas por callback (model.add_callback("on_fit_epoch_end", ...)).
2. Para o resume: gere um run_id determinístico por run_name (ex.: salvo em runs/detect/<run_name>/wandb_id.txt) e use wandb.init(id=run_id, resume="allow").
3. Adicione a flag --wandb {online,offline,disabled} (padrão: offline se não houver login, detectado por wandb.api.api_key).
4. Registre no config os args REAIS (vars(args) + o cache e workers resolvidos).
5. No fim, se estiver offline, mostre o comando `wandb sync wandb/offline-run-*`.
6. Coloque wandb/ no .gitignore.

RESTRIÇÕES
- O treino nunca pode falhar por causa do WandB.

CRITÉRIO DE ACEITE
- Um treino interrompido e resumido aparece como UM run contínuo no WandB.
- Com --wandb disabled, nenhuma pasta wandb/ é criada.

FORMATO DA RESPOSTA
Decisão justificada (2-3 frases), diff e como verificar no painel.
```

---

### Prompt 12 — Reprodutibilidade

```
Você é um pesquisador preocupado com reprodutibilidade em experimentos de ML (contexto: iniciação científica, resultados vão para um relatório).

CONTEXTO
- requirements.txt só usa ">=" (torch>=2.1, ultralytics>=8.3...) e o comentário diz "Testado em Windows", mas o projeto rodou no Linux com torch 2.11.0+cu126 e ultralytics 8.4.34.
- step5_compare (train.py) e inference_compare.py usam random.sample sem seed: as imagens comparadas mudam a cada execução.
- O treino não fixa seed/deterministic explicitamente.
- step5 valida de novo os modelos que train_single_model já validou.
- conf=0.25 e iou=0.45 fixos na comparação visual (o inference_compare tem --conf, mas não usa o valor no predict).

TAREFA
1. Adicione --seed (padrão 42) e use em random.seed, numpy, e em model.train(seed=..., deterministic=True).
2. Salve as imagens sorteadas em results/comparison/<ts>/selected_images.txt, e adicione --images-list para reusar a mesma lista.
3. Faça o predict usar args.conf e um novo args.iou.
4. Guarde as métricas da validação final em runs/detect/<run>/final_metrics.json e, no step5, reaproveite se o best.pt não mudou (compare mtime/hash). Adicione --revalidate para forçar.
5. Gere requirements-lock.txt com as versões exatas (pip freeze filtrado para os pacotes usados) e explique em ../docs/02-instalacao.md a diferença entre os dois arquivos. Corrija o comentário "Windows".
6. Salve em results/comparison/<ts>/environment.json: versões de Python, torch, CUDA e ultralytics, nome da GPU e os args usados.

RESTRIÇÕES
- deterministic=True pode deixar o treino mais lento: documente isso e permita --no-deterministic.

CRITÉRIO DE ACEITE
- Rodar a comparação duas vezes com o mesmo seed gera as mesmas imagens e as mesmas métricas.

FORMATO DA RESPOSTA
Diff, o conteúdo do requirements-lock.txt e uma seção nova pronta para ../docs/02-instalacao.md.
```

---

### Prompt 13 — Conversão de bbox correta

```
Você é um especialista em formatos de anotação (COCO, YOLO, BDD100K).

CONTEXTO
convert_bbox_to_yolo(x1, y1, x2, y2, img_w=1280, img_h=720) converte para (xc, yc, w, h) normalizados e SÓ DEPOIS limita cada valor a [0,1] separadamente. Se a caixa sai da imagem (ex.: x1=-50, x2=100), o centro e a largura limitados geram uma caixa deslocada que não corresponde à parte visível. O tamanho da imagem também é fixo (1280x720) sem verificação. Caixas minúsculas (<2 px) viram ruído.

TAREFA
1. Reescreva: primeiro faça clip de x1,x2 em [0,img_w] e de y1,y2 em [0,img_h]; descarte se a largura ou a altura resultante for menor que min_size (padrão 2 px); depois converta. Não é mais preciso limitar depois.
2. Em process_json, conte quantas caixas foram cortadas e quantas descartadas por tamanho, e mostre nas estatísticas.
3. Opcional, controlado pela flag --check-sizes: para uma amostra de 200 imagens, leia o tamanho real com PIL (Image.open(p).size, sem decodificar) e avise se alguma não for 1280x720.
4. Atualize os testes (pytest): mantenha os 6 casos atuais e adicione:
   - caixa parcialmente fora à esquerda;
   - caixa totalmente fora (deve retornar None);
   - caixa de 1 px (None);
   - round-trip: yolo → xyxy → yolo com erro < 1e-6.

RESTRIÇÕES
- Mesma assinatura de função (adicione min_size como kwarg com padrão).
- Atenção: mudar a conversão invalida os labels já gerados; adicione um aviso e a flag --reconvert.

CRITÉRIO DE ACEITE
- Todos os testes passam.
- A caixa (-50, 100, 100, 200) vira xc=50/1280, w=100/1280 (só a parte visível).

FORMATO DA RESPOSTA
Função nova, testes e diff das estatísticas.
```

---

### Prompt 14 — Manter o README e a documentação

> Este prompt deve ser usado com o assistente aberto na **raiz do repositório** (não em `codigo/`), porque a documentação fica em `README.md` e `docs/`.

```
Você é um redator técnico de projetos de pesquisa em ML.

CONTEXTO
A documentação do projeto fica no README.md da raiz (índice) e em docs/ (01-visao-geral, 02-instalacao, 03-como-executar, 04-arquitetura-do-codigo, 05-problemas-conhecidos, 06-prompts-de-correcao, 07-glossario). O código fica em codigo/. Depois das correções (Prompts 1-13), essa documentação ficou desatualizada: docs/05-problemas-conhecidos.md lista itens já resolvidos, os números de linha mudaram e os comandos têm flags novas (--on-existing, -y, --cache, --workers, --seed, --sanity, --strict, --wandb etc.).

TAREFA
1. Rode `--help` em cada script de codigo/ e atualize as tabelas de opções em docs/03-como-executar.md e codigo/README.md com TODAS as flags reais.
2. Atualize docs/04-arquitetura-do-codigo.md com a nova organização dos arquivos e funções (números de linha incluídos).
3. Em docs/05-problemas-conhecidos.md, marque cada problema corrigido como "✅ Resolvido" e registre a correção na tabela "Histórico de correções" (data, problema, mudança, hash do commit).
4. Atualize o "Início rápido" e a tabela "Situação do projeto" do README.md da raiz (incluindo o --sanity, se existir).
5. Adicione em docs/03-como-executar.md uma seção "Rodando em servidor" (nohup/tmux/-y).
6. Atualize a seção de solução de problemas: CUDA indisponível, falta de memória, symlink no Windows, WandB offline.
7. Confira se todos os links relativos entre os arquivos .md continuam funcionando.

RESTRIÇÕES
- Português, direto, sem marketing. Todos os comandos precisam ter sido testados ou estar marcados como "não testado".
- Mantenha a estrutura de arquivos e a navegação (links "← anterior · Índice · próximo →") existentes.

CRITÉRIO DE ACEITE
- Uma pessoa que nunca viu o projeto consegue, só com a documentação, ir do zero até a primeira imagem comparativa.

FORMATO DA RESPOSTA
O diff de cada arquivo .md alterado.
```

---

[← Problemas conhecidos](05-problemas-conhecidos.md) · [Índice](../README.md) · [Próximo: Glossário →](07-glossario.md)
