# Plano GPU — RTX 5090 (Colega de Time)

> Este documento detalha o plano de experimentos para rodar na GPU RTX 5090 do colega.
> Cada experimento tem: objetivo, código de referência, métrica esperada, e como salvar resultados.

---

## Filosofia

1. **Sempre salvar checkpoints** — interrupções custam horas
2. **Logar tudo no results.tsv** — commit hash, score, VRAM, tempo, notas
3. **Branch por experimento** — `exp/deberta-v3`, `exp/birads-bert`, etc.
4. **Não commitar dados** — só código e resultados agregados
5. **Aprender com falhas** — anotar por que cada experimento caiu

---

## Setup na Máquina do Colega

```bash
# Clone do repo
git clone <repo_url>
cd SPR2026

# Instalar dependências pesadas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate scikit-learn pandas numpy
pip install sentencepiece protobuf

# Verificar GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM')"

# Baixar dados
kaggle competitions download -c spr-2026-mammography-report-classification -p data/
unzip data/spr-2026-mammography-report-classification.zip -d data/
```

---

## Experimentos em Ordem de Prioridade

### EXP-01: BERTimbau Base Fine-Tuning ⭐ (começar aqui)

**Modelo**: `neuralmind/bert-base-portuguese-cased`
**Por quê**: BERT treinado em português — alinhado com os laudos em PT-BR
**Tempo estimado**: ~30 min na 5090
**VRAM estimada**: ~6 GB

```python
# exp01_bertimbau.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import hashlib

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_LABELS = 7
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5

# ... (ver experimentos/exp01_bertimbau.py quando criado)
```

**Hiperparâmetros a testar**:
- LR: 1e-5, 2e-5, 3e-5
- Epochs: 3, 5, 7
- Max length: 128, 256, 512

---

### EXP-02: BERTimbau Large Fine-Tuning

**Modelo**: `neuralmind/bert-large-portuguese-cased`
**Por quê**: Versão maior — mais capacidade para texto clínico
**Tempo estimado**: ~90 min na 5090
**VRAM estimada**: ~18 GB

**Diferença do EXP-01**: trocar apenas o model name, batch_size para 16, gradient_accumulation_steps=2

---

### EXP-03: BioBERTpt Fine-Tuning ⭐⭐ (alta prioridade)

**Modelo**: `pucpr/biobertpt-all`
**Por quê**: BERT biomédico em português — treinado em PubMed + SciELO em PT
**Tempo estimado**: ~35 min na 5090
**VRAM estimada**: ~6 GB

```python
MODEL_NAME = "pucpr/biobertpt-all"
# Mesma arquitetura do EXP-01, só trocar model name
```

---

### EXP-04: DeBERTa-v3 Multilingual

**Modelo**: `microsoft/mdeberta-v3-base`
**Por quê**: Estado da arte em classificação de texto — multilingual
**Tempo estimado**: ~45 min na 5090
**VRAM estimada**: ~8 GB

```python
MODEL_NAME = "microsoft/mdeberta-v3-base"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
```

---

### EXP-05: Ensemble BERT + TF-IDF ⭐⭐⭐ (maior potencial)

**Ideia**: Combinar probabilidades do melhor BERT com TF-IDF + LinearSVC
**Por quê**: BERT captura semântica; TF-IDF captura padrões léxicos raros (classes 5 e 6)
**Tempo estimado**: ~60 min (treinar ambos)

```python
# Pesos do ensemble a otimizar via optuna no val set
alpha = 0.7  # peso BERT
beta  = 0.3  # peso TF-IDF

proba_final = alpha * proba_bert + beta * proba_tfidf
```

---

### EXP-06: Autoresearch Loop — Fase 1 (Overnight Base) 🤖

**Ferramenta**: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
**Objetivo**: Deixar o agente AI otimizar `train.py` overnight a partir do melhor resultado encontrado (~100 experimentos)
**Pré-requisito**: Rodar após EXP-01, EXP-03 ou EXP-04 — usar o melhor `train.py` como ponto de partida

**Configuração**:

```bash
# Na máquina do colega
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# Copiar nossos arquivos
cp ../SPR2026/prepare.py .
cp ../SPR2026/train.py .       # <-- usar o train.py do melhor exp até agora
cp ../SPR2026/program.md .

# Inicializar tracking
echo -e "commit\tcv_f1\tvram_gb\tstatus\tnotes" > results.tsv

# Rodar overnight (budget = minutos por experimento)
python autoresearch.py --branch autoresearch/spr2026 --budget 10
```

**O que o agente explora automaticamente na Fase 1**:
- Learning rates, warmup, weight decay
- Focal loss gamma (0.5 → 3.0) e alpha por classe
- Batch size e gradient accumulation
- Dropout rates
- Max token length (128, 256, 512)
- Scheduler: linear vs cosine vs cosine_with_restarts

---

### EXP-06b: Autoresearch Loop — Fase 2 (Refinamento Inteligente) 🤖🔁

**Objetivo**: Após a Fase 1, o agente analisa o histórico de `results.tsv`, identifica os padrões dos experimentos bem-sucedidos e propõe uma nova rodada de refinamento focada.

**Como funciona**:

```bash
# 1. Após a Fase 1 terminar, checar os top-5 experimentos:
sort -t$'\t' -k2 -rn results.tsv | head -6

# 2. Atualizar o program.md com o aprendizado da Fase 1
# (fazer isso manualmente ou deixar o agente fazer via prompt abaixo)

# 3. Rodar Fase 2 com budget menor e foco em refinamento
python autoresearch.py --branch autoresearch/spr2026-phase2 --budget 5
```

**Atualizar o `program.md` antes da Fase 2** com este template:

```markdown
## Resultados da Fase 1 (preencher antes de rodar Fase 2)

Melhores configurações encontradas:
- Melhor cv_f1: [X.XXX] — commit [hash]
- LR ótimo: [valor]
- Gamma focal loss ótimo: [valor]
- Max length ótimo: [valor]

## Foco da Fase 2

Com base nos resultados acima, explorar:
1. Variações finas ao redor do melhor LR (±20%)
2. Combinações de data augmentation não testadas
3. Arquiteturas de cabeça classificadora (linear vs MLP vs attention pooling)
4. Label smoothing: 0.05, 0.1, 0.15
5. Mixup de embeddings entre classes minoritárias
```

---

### EXP-06c: Geração do Report de Melhorias 📄

**Objetivo**: Após as Fases 1 e 2, gerar automaticamente um relatório estruturado com os aprendizados, salvar no GitHub, e manter os checkpoints localmente.

**Script para gerar o report** (rodar na máquina do Raul após os loops):

```python
# generate_autoresearch_report.py
import pandas as pd
import json
from datetime import datetime

def generate_report(results_tsv="results.tsv", output_md="experiments/autoresearch_report.md"):
    df = pd.read_csv(results_tsv, sep="\t")

    top5 = df[df["status"] == "keep"].nlargest(5, "cv_f1")
    discarded = df[df["status"] == "discard"]
    crashed = df[df["status"] == "crash"]

    report = f"""# Autoresearch Report — SPR2026
Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Total de experimentos: {len(df)}
Experimentos bem-sucedidos (keep): {len(df[df['status']=='keep'])}
Descartados: {len(discarded)}
Crashes: {len(crashed)}

---

## Top 5 Experimentos

{top5[['exp_id','model','cv_f1','vram_gb','time_min','notes']].to_markdown(index=False)}

---

## Melhor Configuração

- **cv_f1**: {top5.iloc[0]['cv_f1']:.5f}
- **Notas**: {top5.iloc[0]['notes']}
- **Commit**: {top5.iloc[0].get('commit', 'N/A')}

---

## O Que Funcionou

"""
    # Extrair padrões dos experimentos keep vs discard
    keep_notes = df[df['status']=='keep']['notes'].tolist()
    discard_notes = df[df['status']=='discard']['notes'].tolist()

    report += "### Padrões nos experimentos que melhoraram:\n"
    for n in keep_notes[:10]:
        report += f"- {n}\n"

    report += "\n### Padrões nos experimentos descartados:\n"
    for n in discard_notes[:10]:
        report += f"- {n}\n"

    report += f"""
---

## Próximos Passos Recomendados

1. Submeter o melhor checkpoint para o Kaggle e verificar score público
2. Rodar Fase 2 com foco nos hiperparâmetros do top-1
3. Ensemble do top-3 e comparar com modelo único
4. Investigar F1 por classe — onde ainda estamos perdendo?

---

## Checkpoints (ficam na máquina local)

Os checkpoints dos top-5 experimentos estão em:
`autoresearch/checkpoints/` — NÃO commitar no GitHub (arquivos grandes)

Para compartilhar: usar `scp` ou Google Drive para transferir apenas o best_model.pt
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report salvo em: {output_md}")
    return output_md

if __name__ == "__main__":
    generate_report()
```

**Após gerar o report, commitar apenas o markdown**:

```bash
# Na máquina do Raul — após rodar generate_autoresearch_report.py:
cd ../SPR2026

# Copiar o report gerado (só o .md, não os checkpoints)
cp ../autoresearch/experiments/autoresearch_report.md experiments/

# Commitar no GitHub
git add experiments/autoresearch_report.md experiments/results.tsv
git commit -m "autoresearch: phase1+2 report - best cv_f1=0.XXX"
git push

# Checkpoints ficam LOCALMENTE na máquina do Raul:
# autoresearch/checkpoints/ — não subir (podem ter GB)
```

**O que vai para o GitHub** (leve):
- `experiments/results.tsv` — tabela de todos os experimentos
- `experiments/autoresearch_report.md` — relatório com análise e próximos passos

**O que fica local** (pesado):
- Checkpoints `.pt` / `.safetensors`
- Cache do HuggingFace
- Logs detalhados de treino

---

### EXP-07: BiomedBERT English (para comparação)

**Modelo**: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
**Nota**: Os laudos são em PT, então pode ser inferior ao BERTimbau
**Valor**: Entender se o pré-treino em domínio (biomédico) > idioma (português)

---

### EXP-08: Class Imbalance Strategies

**Foco**: Classes minoritárias (5 = 29 amostras, 6 = 45 amostras)

```python
# Estratégias a testar:

# A) Focal Loss (penaliza mais os erros nas classes raras)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        ...

# B) Oversampling SMOTE-like (duplicar exemplos das classes raras)
from sklearn.utils import resample
# Resample class 5 para 200 exemplos

# C) Two-stage training
# Stage 1: treinar com todos os dados
# Stage 2: fine-tune apenas nas classes minoritárias

# D) Threshold adjustment
# Após treino, ajustar threshold de classificação para cada classe
# via otimização no val set
```

---

## Sistema de Tracking de Resultados

### results.tsv (manter atualizado)

```tsv
exp_id  model                   cv_f1   vram_gb  time_min  status  notes
EXP-00  TF-IDF+LinearSVC        0.748   0        2         keep    baseline local
EXP-01  BERTimbau-base          ???     6        30        ???     aguardando
```

### Após cada experimento, rodar:

```bash
# Adicionar linha ao results.tsv
echo -e "EXP-01\tBERTimbau-base\t0.XXX\t6.2\t28\tkeep\t lr=2e-5, ep=5" >> experiments/results.tsv
git add experiments/results.tsv
git commit -m "EXP-01: BERTimbau-base cv_f1=0.XXX"
```

---

## Protocolo de Erros — O Que Fazer Quando um Experimento Falha

| Erro | Causa Provável | Solução |
|---|---|---|
| CUDA OOM | Batch size grande | Reduzir batch_size, aumentar gradient_accumulation_steps |
| F1 < 0.60 | Imbalance sem compensação | Verificar class_weight, focal loss |
| Convergência lenta | LR muito baixo | Aumentar LR ou usar warmup schedule |
| Overfitting (train >> val) | Modelo muito grande | Aumentar dropout, reduzir epochs |
| Classes 5/6 com F1=0 | Poucos exemplos | Oversampling ou two-stage training |

---

## Salvando e Compartilhando Resultados

```bash
# O colega envia apenas:
# 1. results.tsv atualizado
# 2. O melhor checkpoint (só os pesos, não os dados)
# 3. O script de inferência correspondente

# Estrutura para compartilhar:
experiments/
├── results.tsv              # todos os experimentos
├── exp01_bertimbau/
│   ├── config.json
│   ├── best_model.pt        # checkpoint (~430 MB para base)
│   └── inference.py
└── exp03_biobertpt/
    ├── config.json
    ├── best_model.pt
    └── inference.py
```

---

## Cronograma Sugerido

| Dia | Ação | Responsável |
|---|---|---|
| Dia 1 manhã | EXP-01 BERTimbau base | GPU 5090 |
| Dia 1 tarde | EXP-03 BioBERTpt | GPU 5090 |
| Dia 1 noite | EXP-06 Autoresearch overnight | GPU 5090 |
| Dia 2 | Analisar resultados, EXP-04 DeBERTa | GPU 5090 |
| Dia 2 tarde | EXP-05 Ensemble best models | GPU 5090 |
| Dia 3 | EXP-08 Imbalance strategies | GPU 5090 |
| Contínuo | Submissões no Kaggle | Qualquer |

---

## Dicas para a 5090

- **bf16 em vez de fp16**: RTX 5090 (Blackwell) suporta bf16 nativo — use `bf16=True` no TrainingArguments
- **torch.compile**: acelera até 20% com `model = torch.compile(model)`
- **Flash Attention 2**: instalar `flash-attn` para modelos grandes
- **Batch size grande**: com 32GB VRAM pode usar batch_size=64 para modelos base

```python
training_args = TrainingArguments(
    bf16=True,           # Blackwell nativo
    tf32=True,           # TensorFloat32 para ops intermediárias
    dataloader_num_workers=4,
    per_device_train_batch_size=64,    # aproveitar os 32GB
    gradient_accumulation_steps=1,
    ...
)
```
