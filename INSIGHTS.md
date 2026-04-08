# SPR 2026 — Insights & Histórico de Experimentos

## Competição
- **Task:** Classificar laudos mamográficos (PT-BR) em BI-RADS 0-6
- **Métrica:** Macro F1
- **Aviso do organizador:** Classes raras impactam muito o score. Esperar shake-up no private LB. Usar validação local para escolher as 2 melhores submissions.
- **Label cleaning permitido** (confirmado pelo host em 2026-04-08)
- Uma competidora (53rd) confirmou que o dataset tem inconsistências

---

## Scoreboard (Public LB) — Atualizado 2026-04-08

| # | Notebook | Score | Approach |
|---|----------|-------|----------|
| 1 | **EXP-25 label cleaning** | **0.80473** | V12 + remove mislabeled (P<10%, all models disagree) |
| 2 | V12 (base, externo) | 0.80403 | 3 SVC + 1 LGB, pesos fixos, sem CV |
| 3 | EXP-14 (2lvl stacking) | 0.80328 | 7 modelos, meta-learner LR, GroupKFold 5-fold |
| 4 | EXP-15 (multiC+semente) | 0.79699 | Multi-C SVC, pesos OOF Dirichlet, thresholds 0-6 |
| 5 | EXP-17 (creative) | 0.78362 | — |
| 6 | EXP-20 (v14 melhorado) | 0.78391 | V14 + mais dense features |
| 7 | EXP-08 (BERTimbau focal) | 0.77784 | BERTimbau Base + Focal Loss (GPU) |
| 8 | EXP-16 (2level stacking) | 0.77620 | 9 modelos, meta-learner, thresholds 0-6 |
| 9 | EXP-22 (um xtudo) | 0.77120 | — |
| 10 | EXP-12 (hybrid stacking) | 0.76839 | TF-IDF + stacking |
| 11 | exp-01-gs (Gustavo) | 0.76522 | — |
| 12 | EXP-11 (tfidf turbo) | 0.76449 | TF-IDF multi-section + 5 models |
| 13 | EXP-18 (classicml) | 0.75889 | — |
| 14 | EXP-23 (minimal SVC) | 0.75446 | Apenas 2 SVCs (sem LGB) |
| 15 | EXP-21 (catboost) | 0.74732 | V14 + CatBoost |
| 16 | EXP-19 (lightgpu) | 0.74280 | LightGBM GPU |
| 17 | exp_gs_04 (Gustavo) | 0.66310 | — |
| 18 | EXP-10 (purebert) | 0.62250 | BERT only |
| 19 | EXP-02 (bertimbau large) | 0.53317 | BERTimbau large (pouco folds) |

### Pendentes
- **EXP-24** — specialist binary SVCs para classes raras + motor V12
- **EXP-26** — label correction iterativo (corrige em vez de remover, 3 rounds)

---

## O que FUNCIONA
1. **TF-IDF + LinearSVC** é o melhor approach para texto clínico PT-BR
2. **3 SVCs** (achados, full, full2) com word+char ngrams é o ponto ótimo
3. **1 LGB** no achados + 5 dense features complementa os SVCs
4. **Pesos fixos V12:** SVC blend (0.25A + 0.40F + 0.35F2), depois 0.70SVC + 0.30LGB
5. **Thresholds classes 3-6** com busca greedy OOF (6→5→4→3)
6. **Label cleaning conservador** — remover P(label)<10% onde todos modelos discordam (+0.0007)
7. **5 dense features:** report_length, has_measurement, has_spiculation, has_distortion, has_biopsy
8. **Guardrails:** carcinoma/CDIS→6, espiculado+distorção→5

## O que NÃO FUNCIONA
1. **Mais modelos/diversidade** piora (EXP-16: 9 modelos < EXP-14: 7)
2. **Multi-C SVC** piora (EXP-15: 0.797 vs V12: 0.804)
3. **Busca de pesos OOF Dirichlet** provavelmente overfita
4. **Thresholds para classes 0 e 1** piora
5. **Mais dense features** piora (EXP-20: 0.784)
6. **CatBoost** não compete (EXP-21: 0.747)
7. **Menos que 3 SVCs** perde informação (EXP-23: 0.754)
8. **BERT** sem GPU suficiente não compete
9. **LGB no full text** causa miscalibração

**Regra geral:** O ponto ótimo é 3-4 modelos simples. Menos perde info, mais adiciona ruído. Label cleaning é a fronteira mais promissora.

---

## Melhores 2 submissions (final)
1. **EXP-25** (0.80473) — melhor score + dados limpos
2. **EXP-14** (0.80328) — CV robusto, estratégia diferente

---

## Arquitetura V12 (referência)

```
TF-IDF:
  Achados: word(1,3) + char_wb(3,5) 80k
  Full v1: word(1,3) + char_wb(3,5) 80k
  Full v2: word(1,3) + char_wb(3,6) 100k
  LGB input: achados TF-IDF + 5 dense features

Models:
  SVC-A: CalibratedClassifierCV(LinearSVC balanced) on achados
  SVC-F: same on full v1
  SVC-F2: same on full v2
  LGB: LGBMClassifier balanced, 300 trees, lr=0.05, depth=6

Blend: 0.70 * (0.25*A + 0.40*F + 0.35*F2) + 0.30 * LGB
Thresholds: cls6>0.15, cls5>0.20, cls4>0.25, cls3>0.33
Guardrails: carcinoma/CDIS→6, espiculado+distorção→5
```

Notebook V12 base: `C:\Users\jvict\Downloads\improved-mammography-classifier.ipynb`
