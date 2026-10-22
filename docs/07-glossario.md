# 7. Glossário

[← Prompts de correção](06-prompts-de-correcao.md) · [Índice](../README.md)

---

Termos usados no projeto, em ordem alfabética.

| Termo | Significado |
|---|---|
| **AMP** (*Automatic Mixed Precision*) | Treinar usando números de 16 bits (FP16) onde for seguro. Gasta menos VRAM e deixa o treino até ~2× mais rápido |
| **Anchor-free** | Detector que prevê a caixa diretamente a partir de cada ponto da imagem, sem caixas-modelo pré-definidas (*anchors*). YOLOv8 e YOLOv10 são *anchor-free* |
| **Batch** | Quantas imagens o modelo processa de uma vez antes de atualizar os pesos. Batch maior usa mais VRAM |
| **BDD100K** | *Berkeley DeepDrive 100K*: dataset de direção com 100 mil vídeos. Veja a [visão geral](01-visao-geral.md#dataset-bdd100k) |
| **`best.pt`** | Checkpoint da época com o melhor *fitness* na validação (no Ultralytics, 0,1·mAP50 + 0,9·mAP50-95) |
| **Bounding box (bbox)** | Retângulo que delimita um objeto na imagem |
| **Cache** | Guardar as imagens já decodificadas na RAM (`"ram"`) ou em disco (`"disk"`) para não ler o `.jpg` de novo em cada época |
| **Checkpoint** | Arquivo `.pt` com os pesos do modelo e o estado do treino, usado para continuar depois |
| **COCO** | Dataset genérico de 80 classes em que os pesos `yolov8m.pt` e `yolov10m.pt` foram pré-treinados |
| **`coco8.yaml`** | Mini-dataset de demonstração do Ultralytics (8 imagens). Aparecer no treino indica erro de configuração |
| **Confiança (`conf`)** | Probabilidade que o modelo atribui a uma detecção. Detecções abaixo do limiar (0,25 no projeto) são descartadas |
| **`close_mosaic`** | Desligar o aumento de dados *mosaic* nas últimas N épocas para o modelo se ajustar a imagens normais |
| **`cos_lr`** | Taxa de aprendizado que decresce seguindo uma curva de cosseno |
| **CUDA** | Plataforma da NVIDIA que permite ao PyTorch usar a GPU |
| **`data.yaml`** | Arquivo que diz ao Ultralytics onde estão as imagens e os labels e quais são as classes |
| **Época (*epoch*)** | Uma passada completa por todas as imagens de treino |
| **Fine-tuning** | Partir de pesos pré-treinados (COCO) e continuar o treino em outro dataset (BDD100K) |
| **IoU** (*Intersection over Union*) | Área da interseção dividida pela área da união entre a caixa prevista e a real. 1 = sobreposição perfeita |
| **Junction** | Tipo de link de pasta do Windows que não exige permissão de administrador |
| **`last.pt`** | Checkpoint da última época concluída. Usado no *resume* |
| **mAP** (*mean Average Precision*) | Média, entre as classes, da área sob a curva precisão × recall |
| **mAP50** | mAP considerando acerto quando IoU ≥ 0,50 |
| **mAP50-95** | Média do mAP com limiares de IoU de 0,50 a 0,95 (passo 0,05). Métrica principal do COCO, mais exigente |
| **Mosaic** | Aumento de dados que junta 4 imagens em uma só durante o treino |
| **NMS** (*Non-Maximum Suppression*) | Pós-processamento que remove caixas duplicadas do mesmo objeto. O YOLOv10 foi projetado para dispensá-lo |
| **`patience`** | Número de épocas sem melhora antes de parar o treino (*early stopping*). `0` desativa |
| **Precision (precisão)** | Verdadeiros positivos ÷ todas as detecções. "Das caixas que o modelo desenhou, quantas estavam certas?" |
| **Recall (revocação)** | Verdadeiros positivos ÷ todos os objetos reais. "Dos objetos que existiam, quantos o modelo achou?" |
| **Resume** | Continuar um treino interrompido a partir do `last.pt` |
| **Run** | Uma execução de treino, com sua pasta em `runs/detect/<nome>/` |
| **Symlink** | Link simbólico: um "atalho" de pasta que aponta para outro lugar sem copiar os arquivos |
| **VRAM** | Memória da placa de vídeo. Limita o batch e o tamanho da imagem |
| **WandB** (*Weights & Biases*) | Serviço online para acompanhar gráficos de treino |
| **`weights_only`** | Parâmetro do `torch.load` que impede um `.pt` de executar código ao ser carregado |
| **Workers** | Processos paralelos que leem e preparam as imagens enquanto a GPU treina |
| **YOLO** (*You Only Look Once*) | Família de detectores de objetos em um único estágio, rápidos o bastante para tempo real |
| **Formato YOLO** | Um `.txt` por imagem, uma linha por objeto: `classe x_centro y_centro largura altura`, tudo normalizado entre 0 e 1 |

---

[← Prompts de correção](06-prompts-de-correcao.md) · [Índice](../README.md)
