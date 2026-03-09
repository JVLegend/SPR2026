# Relatório de Pesquisa — SPR 2026 Mammography

> Gerado em: 2026-03-09
> Fonte: Pesquisa em competições Kaggle similares + literatura NLP clínico + papers BI-RADS

---

## TL;DR — O que fazer primeiro

1. **BERTimbau-base** ou **BioBERTpt(all)** como backbone → +15-25% vs TF-IDF
2. **Focal Loss** para classes raras (5 = 29 amostras, 6 = 45 amostras) → +15-25% nessas classes
3. **Two-stage fine-tuning** → +8-12% macro F1 em datasets extremamente desbalanceados
4. **Ensemble 3-5 modelos** (sementes diferentes) → +3-5% adicional
5. **EDA / data augmentation** nas classes minoritárias → +5-10%

**Meta realista**: 0.773 → **0.82-0.87** com pipeline completo.

---

## 1. O que Competições Similares Ensinaram

### NBME — Score Clinical Patient Notes (2022)
- **Task**: NER em notas clínicas de pacientes (médico)
- **1º lugar**: Ensemble de modelos **DeBERTa** — 900h de trabalho
- **Aprendizado**: DeBERTa domina texto clínico; diversidade de modelos no ensemble é chave
- **Link**: [github.com/alexBDG/Score-Clinical-Patient-Notes](https://github.com/alexBDG/Score-Clinical-Patient-Notes)

### PII Detection (2023)
- **1º lugar**: 5x **DeBERTa-v3-large** com arquiteturas distintas
- **Aprendizado**: Usar arquiteturas ligeiramente diferentes (não só seeds) para diversidade de ensemble
- **Link**: [github.com/bogoconic1/pii-detection-1st-place](https://github.com/bogoconic1/pii-detection-1st-place)

### Otto Group Product Classification (referência ensemble)
- **Trick vencedor**: Stacking em 3 camadas (33 modelos base → 3 meta-learners → média ponderada)
- **Aprendizado**: Para classificação multiclasse extremamente desbalanceada, meta-ensembles valem mais que um único modelo grande

### TECRR — Benchmark BI-RADS de Relatórios (paper 2024, BMC)
- Dataset: 5.046 relatórios únicos de mamografia
- **Melhores resultados**: XLM-RoBERTa, BETO, BioBERTpt — 74-77% accuracy, 88-91% em priorização binária
- **BioBERTpt superou BETO espanhol** em textos portugueses
- **Link**: [bmcmedinformdecismak.biomedcentral.com](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02717-7)

---

## 2. Modelos Recomendados (por prioridade)

### Tier A — Alta Prioridade (rodar primeiro na 5090)

| Modelo | HuggingFace | Por quê |
|---|---|---|
| `neuralmind/bert-large-portuguese-cased` | [link](https://huggingface.co/neuralmind/bert-large-portuguese-cased) | BERTimbau Large — BERT em PT-BR, estado da arte local |
| `pucpr/biobertpt-all` | [link](https://huggingface.co/pucpr/biobertpt-all) | Biomédico PT-BR (PubMed + SciELO + notas clínicas) |
| `microsoft/mdeberta-v3-base` | [link](https://huggingface.co/microsoft/mdeberta-v3-base) | Estado da arte multilingual, domina NLP competitions |

### Tier B — Média Prioridade

| Modelo | HuggingFace | Por quê |
|---|---|---|
| `neuralmind/bert-base-portuguese-cased` | [link](https://huggingface.co/neuralmind/bert-base-portuguese-cased) | BERTimbau Base — mais leve que Large |
| `xlm-roberta-large` | [link](https://huggingface.co/xlm-roberta-large) | RoBERTa multilingual grande |
| `microsoft/deberta-v3-large` | [link](https://huggingface.co/microsoft/deberta-v3-large) | DeBERTa-v3-large inglês (pode ajudar mesmo em PT) |

### Tier C — Experimental

| Modelo | Por quê |
|---|---|
| `medicalai/ClinicalBERT` | BERT em notas clínicas MIMIC-III (inglês, mas domínio clínico forte) |
| GPT-2 para augmentation | Gerar exemplos sintéticos de classes 5 e 6 |

---

## 3. Técnicas de Imbalance — Implementação Concreta

### 3.1 Focal Loss ⭐⭐⭐ (maior impacto esperado)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para classificação multiclasse desbalanceada.
    Penaliza mais os erros nas classes raras.

    alpha: peso por classe (inverso da frequência)
    gamma: fator de foco (2.0 é o padrão da literatura)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # tensor de shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

# Como calcular alpha a partir dos dados:
# from sklearn.utils.class_weight import compute_class_weight
# weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# alpha = torch.tensor(weights, dtype=torch.float32).to(device)
```

**Hiperparâmetros a testar**: gamma ∈ {0.5, 1.0, 2.0, 3.0}

---

### 3.2 Two-Stage Fine-Tuning ⭐⭐ (crítico para imbalance extremo)

```python
# Stage 1: apenas a cabeça classificadora (backbone congelado)
# - Epochs: 2-3
# - LR: 1e-3 (maior porque só treina a cabeça)
# - Loss: focal loss com alpha pesado nas classes raras

# Stage 2: modelo completo com LR diferenciado
# - Epochs: 3-5
# - LR backbone: 1e-5 (menor para não destruir representações)
# - LR cabeça: 1e-4
# - Loss: focal loss

# Implementação com HuggingFace Trainer:
# Stage 1:
for param in model.base_model.parameters():
    param.requires_grad = False
trainer_stage1 = Trainer(model=model, args=args_stage1, ...)
trainer_stage1.train()

# Stage 2:
for param in model.base_model.parameters():
    param.requires_grad = True
trainer_stage2 = Trainer(model=model, args=args_stage2, ...)
trainer_stage2.train()
```

**Paper de referência**: "Two-Stage Fine-Tuning: A Novel Strategy for Learning Class-Imbalanced Data" (arxiv 2207.10858)

---

### 3.3 EDA — Easy Data Augmentation para classes raras

```python
# Aplicar apenas nas classes com < 200 amostras:
# classes 0 (610), 1 (693), 3 (713), 4 (214), 5 (29), 6 (45)
# Focar em: 5 (29 → ~200), 6 (45 → ~200), 4 (214 → ~500)

# Operações EDA:
# 1. Synonym replacement: trocar palavras por sinônimos médicos
#    Ex: "nodulo" → "lesao nodular", "calcificacao" → "deposito calcario"
# 2. Random swap: trocar ordem de frases (laudos têm seções modulares)
# 3. Random deletion: remover frases menos informativas (p=0.1)
# 4. Back-translation: PT → EN → PT via DeepL API (custo baixo para 200 amostras)

# Implementação simples com nlpaug:
# pip install nlpaug
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet', lang='por')
augmented = aug.augment(text, n=3)  # 3 versões por texto
```

---

### 3.4 Oversampling Estratégico

```python
# Não usar SMOTE no espaço de features TF-IDF (dimensionalidade muito alta)
# Usar oversampling direto dos textos originais antes do tokenizer

from sklearn.utils import resample

def oversample_rare_classes(df, target_col='target', min_samples=200):
    """Duplica amostras das classes raras até atingir min_samples."""
    parts = [df]
    for cls, count in df[target_col].value_counts().items():
        if count < min_samples:
            cls_df = df[df[target_col] == cls]
            n_needed = min_samples - count
            oversampled = resample(cls_df, n_samples=n_needed,
                                   replace=True, random_state=42)
            parts.append(oversampled)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

# Resultado esperado: classe 5: 29 → 200, classe 6: 45 → 200
```

---

## 4. Hiperparâmetros Recomendados (DeBERTa / BERTimbau)

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Modelo
    per_device_train_batch_size=32,     # 5090 com 32GB: pode usar 64
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,

    # Otimização
    learning_rate=2e-5,                 # testar: 1e-5, 2e-5, 3e-5
    warmup_ratio=0.1,                   # 10% dos steps
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Treino
    num_train_epochs=5,                 # testar: 3, 5, 7
    lr_scheduler_type="cosine",

    # Eficiência (5090 Blackwell)
    bf16=True,                          # melhor que fp16 em Blackwell
    tf32=True,
    dataloader_num_workers=4,

    # Avaliação
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    # Logging
    logging_steps=50,
    report_to="none",                   # mudar para "wandb" se quiser tracking
)
```

---

## 5. Estratégia de Ensemble Final

```python
# Pesos sugeridos (a otimizar via optuna no val set):
# Modelos com melhor macro F1 recebem peso maior

def weighted_ensemble(probas_list, weights=None):
    """
    probas_list: lista de arrays (n_samples, n_classes)
    weights: lista de floats (soma = 1.0)
    """
    if weights is None:
        weights = [1/len(probas_list)] * len(probas_list)

    weighted = sum(w * p for w, p in zip(weights, probas_list))
    return weighted.argmax(axis=1)

# Exemplo:
# proba_bertimbau = model_bertimbau.predict_proba(X_test)
# proba_biobertpt = model_biobertpt.predict_proba(X_test)
# proba_deberta   = model_deberta.predict_proba(X_test)
# proba_tfidf     = model_tfidf.predict_proba(X_test)

# Pesos iniciais:
# weights = [0.35, 0.30, 0.25, 0.10]  # bertimbau, biobertpt, deberta, tfidf

# Otimizar pesos com Nelder-Mead no val set:
from scipy.optimize import minimize
def neg_f1(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()
    preds = weighted_ensemble(probas_val, weights)
    return -f1_score(y_val, preds, average='macro')

result = minimize(neg_f1, x0=[0.25]*4, method='Nelder-Mead')
best_weights = result.x / result.x.sum()
```

---

## 6. Pipeline Completo Sugerido (para a GPU 5090)

```
Dia 1:
  [A] EXP-01: BERTimbau-base, focal loss, 5 épocas
      → salvar checkpoint, logar cv_f1 em results.tsv

  [B] EXP-03: BioBERTpt(all), focal loss, 5 épocas
      → salvar checkpoint, logar cv_f1

Dia 1 noite (overnight):
  [C] EXP-06: Autoresearch loop — explorar LR, gamma, epochs, dropout
      → ~100 experimentos automáticos em EXP-01 como base

Dia 2:
  [D] EXP-04: mDeBERTa-v3-base, focal loss, two-stage fine-tuning
      → comparar com EXP-01 e EXP-03

  [E] EXP-08a: Oversampling classes 5/6 → re-treinar melhor modelo
  [F] EXP-08b: EDA augmentation → re-treinar melhor modelo

Dia 3:
  [G] EXP-05: Ensemble dos 3 melhores modelos
      → otimizar pesos com Nelder-Mead no val set
      → submeter para Kaggle

  [H] EXP-09 (se tempo): BERTimbau-large (maior, mais demorado)
```

---

## 7. Métricas Esperadas por Etapa

| Etapa | Modelo | Macro F1 Esperado |
|---|---|---|
| Baseline atual | TF-IDF + LinearSVC | 0.748 local / 0.773 LB |
| EXP-01 | BERTimbau-base + focal | 0.80 - 0.83 |
| EXP-03 | BioBERTpt(all) + focal | 0.81 - 0.84 |
| EXP-04 | mDeBERTa + two-stage | 0.82 - 0.85 |
| EXP-05 | Ensemble 3 modelos | **0.84 - 0.88** |
| Autoresearch overnight | Melhor config auto | 0.85 - 0.89 |

---

## 8. Fontes e Papers

| Técnica | Paper / Link |
|---|---|
| BioBERTpt | [ACL 2020 — clinicalnlp-1.7](https://aclanthology.org/2020.clinicalnlp-1.7/) |
| Two-Stage Fine-Tuning | [arxiv 2207.10858](https://arxiv.org/pdf/2207.10858) |
| TECRR BI-RADS benchmark | [BMC Medical Informatics 2024](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02717-7) |
| Automatic BI-RADS NLP | [Clinical Radiology 2023](https://www.clinicalradiologyonline.net/article/S0009-9260(23)00423-3/abstract) |
| Focal Loss | [Lin et al. 2017 (RetinaNet)](https://arxiv.org/abs/1708.02002) |
| DeBERTa-v3 | [He et al. 2021](https://arxiv.org/abs/2111.09543) |
| NBME 1st place | [github.com/alexBDG](https://github.com/alexBDG/Score-Clinical-Patient-Notes) |
| PII Detection 1st | [github.com/bogoconic1](https://github.com/bogoconic1/pii-detection-1st-place) |
| Autoresearch | [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) |
| Oversampling BERT médico | [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S0933365724001313) |
