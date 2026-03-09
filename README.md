# SPR 2026 — Mammography Report Classification

## Time: RUMO AO TOPO

| Membro | Papel |
|---|---|
| João Victor Dias | ML / NLP / Coordenação |
| Dr. Gustavo Sakuno | Especialista Clínico / Radiologia |
| Raul Primo | GPU / Engenharia / Infraestrutura |

---

**Competição**: [SPR 2026 Mammography Report Classification](https://www.kaggle.com/competitions/spr-2026-mammography-report-classification)
**Task**: Prever categoria BI-RADS (0-6) a partir de laudos de mamografia em **Português**
**Métrica**: Macro F1
**Baseline público**: 0.773 (TF-IDF + LinearSVC)

---

## Estrutura do Repositório

```
SPR2026/
├── data/                          # ← NÃO versionar (gitignore)
│   ├── train.csv                  # 18.272 amostras, colunas: ID, report, target
│   ├── test.csv                   # 4 amostras (code competition)
│   └── submission.csv             # template de submissão
│
├── experiments/                   # resultados dos experimentos
│   └── results.tsv                # commit | cv_score | notes
│
├── submissions/                   # arquivos de submissão gerados
│   └── *.csv
│
├── prepare.py                     # FIXO — data loading, métricas, submissão
├── train.py                       # MODIFICÁVEL — modelo atual
├── program.md                     # instruções para loop autoresearch
├── plan_gpu.md                    # plano de experimentos para GPU 5090
├── colab_baseline.ipynb           # notebook para rodar no Colab (CPU/T4)
└── README.md                      # este arquivo
```

---

## Setup Rápido

```bash
# 1. Clonar repo
git clone <repo_url>
cd SPR2026

# 2. Instalar dependências leves (Colab / CPU)
pip install scikit-learn pandas numpy kaggle

# 3. Baixar dados
kaggle competitions download -c spr-2026-mammography-report-classification -p data/
unzip data/spr-2026-mammography-report-classification.zip -d data/

# 4. Rodar baseline
python train.py
```

---

## Sobre os Dados

| Atributo | Valor |
|---|---|
| Train size | 18.272 amostras |
| Test size | 4 (code competition) |
| Idioma | Português (BR) |
| Texto médio | ~400 caracteres |
| Duplicatas | 9.141 relatórios duplicados |
| Classes | 7 (BI-RADS 0, 1, 2, 3, 4, 5, 6) |
| Imbalance ratio | 550x (classe 2 domina) |

**Distribuição das classes:**
```
BI-RADS 0:     610  (3.3%)   → "Incompleto"
BI-RADS 1:     693  (3.8%)   → "Negativo"
BI-RADS 2:  15.968 (87.4%)  → "Benigno" ← domina!
BI-RADS 3:     713  (3.9%)   → "Provavelmente Benigno"
BI-RADS 4:     214  (1.2%)   → "Suspeito"
BI-RADS 5:      29  (0.2%)   → "Altamente Suspeito" ← rarísssimo
BI-RADS 6:      45  (0.2%)   → "Maligno Conhecido"
```

**⚠️ Atenção ao imbalance:** Classe 5 tem apenas 29 exemplos. Usar `class_weight="balanced"` em todos os modelos.

---

## Experimentos Realizados (CPU/Local)

| # | Modelo | CV Macro F1 | Notas |
|---|---|---|---|
| 0 | Baseline (TF-IDF + LinearSVC) | ~0.748 local | Score público: 0.773 |
| 1 | TF-IDF + LinearSVC + LR Ensemble | em andamento | v3 train.py |

> **Nota**: O CV local difere do score público por causa do GroupKFold + distribuição de test.

---

## Estratégia de Experimentação

### Tier 1 — CPU / Colab (rápido, sem GPU)
- [x] TF-IDF word+char + LinearSVC (baseline)
- [ ] TF-IDF + LogisticRegression ensemble
- [ ] TF-IDF + múltiplos C values + voting
- [ ] Feature engineering: comprimento do texto, seções detectadas, keywords BI-RADS

### Tier 2 — GPU Colab T4 (médio)
- [ ] `neuralmind/bert-base-portuguese-cased` fine-tuning (BERTimbau)
- [ ] `pucpr/biobertpt-all` — BioMedical BERT em Português
- [ ] Inference rápida com quantização INT8

### Tier 3 — GPU 5090 (pesado, overnight)
Ver `plan_gpu.md` para plano detalhado.

---

## Como Submeter no Kaggle

Esta é uma **code competition** — a submissão é um notebook, não um CSV:

1. Fazer upload do notebook para Kaggle
2. O notebook deve ler de `/kaggle/input/spr-2026-mammography-report-classification/`
3. Gerar `submission.csv` com colunas `ID` e `target`
4. Submeter via "Submit Prediction"

**Limite**: verificar na página da competição (geralmente 5 submissões/dia).

---

## .gitignore

```
data/
*.zip
submissions/*.csv
__pycache__/
*.pyc
.env
```
