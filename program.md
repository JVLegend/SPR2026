# Autoresearch Program — SPR 2026 Mammography Report Classification

## Objetivo

Melhorar o score da competição Kaggle **SPR 2026 Mammography Report Classification** partindo do baseline ~0.773.

**Competição**: https://www.kaggle.com/competitions/spr-2026-mammography-report-classification
**Métrica**: (verificar na competição — provavelmente macro F1 ou accuracy)
**Baseline kernel**: `mirkoferretti/spr-2026-mammography-baseline-0-77321`

---

## Contexto da Task

Classificação de relatórios de mamografia em texto clínico em inglês (radiologia). A task envolve:
- Textos curtos a médios (laudos radiológicos)
- Labels categóricas (possivelmente BI-RADS ou categorias diagnósticas)
- Dados clínicos estruturados + texto livre

---

## O que o Agente Pode Modificar

Apenas `train.py` — o arquivo de treinamento. O agente NÃO pode:
- Mudar o pipeline de dados (prepare.py)
- Adicionar dependências externas além das listadas
- Modificar o script de avaliação

---

## Estratégias a Explorar (em ordem de prioridade)

### 1. Troca de Backbone / Embeddings (Alto impacto)
- Trocar TF-IDF/CountVectorizer por embeddings pré-treinados
- Testar: `sentence-transformers/all-MiniLM-L6-v2` (leve, rápido)
- Testar: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (domínio médico)
- Testar: `medicalai/ClinicalBERT` (domínio clínico)

### 2. Fine-tuning de Modelo Pré-Treinado
- Usar `transformers` (HuggingFace) para fine-tuning
- Arquitetura: BERT/RoBERTa/DeBERTa-v3-base
- Treinamento: AdamW + cosine annealing, 3-5 épocas
- Label smoothing: 0.1

### 3. Data Augmentation de Texto
- Synonym replacement usando WordNet
- Back-translation (EN → PT → EN) para relatórios
- Paraphrase via modelo T5

### 4. Ensemble / Stacking
- Combinar predições de múltiplos modelos leves
- Weighted average por performance no val set
- Stacking com meta-learner logístico

### 5. Pré-processamento Específico de Domínio
- Normalização de abreviações médicas (BI-RADS, ACR, etc.)
- Remover seções irrelevantes (cabeçalho/rodapé dos laudos)
- Extrair features estruturadas: menção a calcificações, densidade, nódulos

### 6. Regularização e Otimização
- Testar: dropout rates 0.1, 0.2, 0.3
- Testar: weight decay 0.01, 0.1
- Testar: learning rates 1e-5, 2e-5, 3e-5, 5e-5
- Gradient clipping: 1.0

### 7. Cross-Validation Estratificada
- 5-fold estratificado por label
- Treinar múltiplos folds e ensemblar predições

---

## Métrica de Progresso

- **Baseline**: ~0.773
- **Meta intermediária**: 0.80
- **Meta ambiciosa**: 0.85+

---

## Como Rodar o Loop

```bash
# 1. Instalar autoresearch
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# 2. Configurar o projeto
# Copiar train.py e prepare.py deste diretório para autoresearch/

# 3. Inicializar results
echo -e "commit\tmetric\tvram\tstatus\tnotes" > results.tsv

# 4. Rodar loop overnight
python autoresearch.py --branch autoresearch/spr2026 --budget 5
```

---

## Próximos Passos Imediatos

1. [ ] Configurar `kaggle.json` em `C:\Users\jvict\.kaggle\`
2. [ ] Baixar baseline: `kaggle kernels pull mirkoferretti/spr-2026-mammography-baseline-0-77321`
3. [ ] Baixar dados: `kaggle competitions download -c spr-2026-mammography-report-classification`
4. [ ] Examinar baseline notebook para entender estrutura dos dados
5. [ ] Adaptar `prepare.py` e `train.py` para o formato autoresearch
6. [ ] Rodar primeiro experimento manual (substituir baseline por BiomedBERT)
7. [ ] Submeter resultado para Kaggle e verificar score público
8. [ ] Iniciar loop autoresearch overnight
