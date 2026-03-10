#!/usr/bin/env bash
# ============================================================
# setup_gpu.sh — SPR2026 GPU Setup (RTX 5090 / Blackwell)
# Roda todos os experimentos do plan_gpu.md em sequência
#
# Uso:
#   chmod +x setup_gpu.sh
#   KAGGLE_USERNAME=seu_user KAGGLE_KEY=sua_key bash setup_gpu.sh
#
# Ou coloque ~/.kaggle/kaggle.json antes de rodar.
# ============================================================

set -e  # Para se qualquer comando falhar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " SPR2026 — Setup GPU RTX 5090"
echo " Dir: $SCRIPT_DIR"
echo "=================================================="

# ── 1. Python e GPU check ──────────────────────────────────
echo ""
echo "[1/6] Verificando ambiente..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch ainda não instalado — será instalado abaixo"

# ── 2. Instalar Kaggle CLI e dependências ──────────────────
echo ""
echo "[2/6] Instalando dependências..."

# PyTorch cu128 para Blackwell (RTX 5090)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128

# HuggingFace stack
pip install --quiet transformers datasets accelerate sentencepiece protobuf

# Flash Attention 2 (acelera modelos grandes na 5090)
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn falhou (opcional) — continuando sem ela"

# Kaggle CLI + utilitários
pip install --quiet kaggle scikit-learn pandas numpy scipy optuna tabulate

echo "Dependências instaladas."

# ── 3. Configurar Kaggle credenciais ──────────────────────
echo ""
echo "[3/6] Configurando Kaggle..."

mkdir -p ~/.kaggle

if [ -f ~/.kaggle/kaggle.json ]; then
    echo "kaggle.json já existe — usando credenciais existentes."
elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "kaggle.json criado a partir das variáveis de ambiente."
else
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│              CREDENCIAIS DO KAGGLE NECESSÁRIAS              │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│  Como obter:                                                │"
    echo "│  1. Acesse https://www.kaggle.com                          │"
    echo "│  2. Clique na sua foto (canto superior direito)            │"
    echo "│  3. Settings → API → Create New Token                      │"
    echo "│  4. Vai baixar um kaggle.json — abra e copie os valores    │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""

    # Lê username sem eco
    read -r -p "  Digite seu Kaggle username: " KAGGLE_USERNAME
    if [ -z "$KAGGLE_USERNAME" ]; then
        echo "ERRO: username não pode ser vazio."
        exit 1
    fi

    # Lê API key sem mostrar no terminal
    read -r -s -p "  Digite sua Kaggle API key (não aparece na tela): " KAGGLE_KEY
    echo ""  # newline após a leitura silenciosa
    if [ -z "$KAGGLE_KEY" ]; then
        echo "ERRO: API key não pode ser vazia."
        exit 1
    fi

    printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "  ✓ Credenciais salvas em ~/.kaggle/kaggle.json"
fi

# ── 4. Baixar dados ───────────────────────────────────────
echo ""
echo "[4/6] Baixando dados do Kaggle..."

mkdir -p data

if [ -f "data/train.csv" ] && [ -f "data/test.csv" ]; then
    echo "Dados já existem em data/ — pulando download."
else
    kaggle competitions download \
        -c spr-2026-mammography-report-classification \
        -p data/

    # Descompactar (pode ser .zip com subpastas)
    if [ -f "data/spr-2026-mammography-report-classification.zip" ]; then
        unzip -o "data/spr-2026-mammography-report-classification.zip" -d data/
        # Se descompactou em subpasta, mover para data/
        if [ ! -f "data/train.csv" ]; then
            find data/ -name "train.csv" -exec mv {} data/ \;
            find data/ -name "test.csv"  -exec mv {} data/ \;
            find data/ -name "submission.csv" -exec mv {} data/ \;
        fi
    fi

    echo "Verificando arquivos..."
    [ -f "data/train.csv" ] || { echo "ERRO: data/train.csv não encontrado após extração!"; exit 1; }
    [ -f "data/test.csv"  ] || { echo "ERRO: data/test.csv não encontrado após extração!"; exit 1; }
    echo "OK — train.csv e test.csv prontos."
fi

# Verificar GPU após instalação do PyTorch
echo ""
echo "[5/6] Verificando GPU..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}')
    print(f'VRAM: {vram:.1f} GB')
    print(f'CUDA: {torch.version.cuda}')
    print(f'bf16 suportado: {torch.cuda.is_bf16_supported()}')
else:
    print('AVISO: CUDA não disponível — rodando em CPU')
"

# ── 5. EXP-00 / baseline: train.py atual ─────────────────
echo ""
echo "[6/6] Rodando experimentos..."
echo ""
echo "=================================================="
echo " EXP BASELINE (train.py atual — TF-IDF Ensemble)"
echo "=================================================="

mkdir -p experiments

START=$(date +%s)
python train.py | tee experiments/baseline_run.log
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

# Extrair score do log
SCORE=$(grep "CV macro F1" experiments/baseline_run.log | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "???")
echo ""
echo "Baseline concluído em ${ELAPSED} min — CV F1: $SCORE"

# Atualizar results.tsv
if ! grep -q "BASELINE-GPU" experiments/results.tsv 2>/dev/null; then
    echo -e "BASELINE-GPU\tTF-IDF+LR Ensemble\t${SCORE}\t0\t${ELAPSED}\tkeep\tPrimeiro run na 5090" >> experiments/results.tsv
fi

# ── 6. EXP-01: BERTimbau Base ─────────────────────────────
echo ""
echo "=================================================="
echo " EXP-01: BERTimbau Base (neuralmind/bert-base-portuguese-cased)"
echo "=================================================="

python - <<'PYEOF' | tee experiments/exp01_bertimbau.log
import torch, numpy as np, time
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from prepare import load_data, make_groups, RANDOM_STATE, N_SPLITS

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_LABELS = 7
MAX_LEN    = 256
BATCH_SIZE = 64   # 5090 tem 32GB, aproveitar
EPOCHS     = 5
LR         = 2e-5

print(f"Modelo: {MODEL_NAME}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

train_df, test_df = load_data()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["report"], truncation=True, max_length=MAX_LEN)

groups = make_groups(train_df)
y = train_df["target"].astype(int).values
gkf = GroupKFold(n_splits=N_SPLITS)

oof_preds = np.zeros(len(train_df), dtype=int)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, y, groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    tr_ds = Dataset.from_dict({
        "report": train_df.iloc[tr_idx]["report"].tolist(),
        "label":  y[tr_idx].tolist()
    }).map(tokenize, batched=True)

    val_ds = Dataset.from_dict({
        "report": train_df.iloc[val_idx]["report"].tolist(),
        "label":  y[val_idx].tolist()
    }).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=f"experiments/exp01_fold{fold+1}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    preds = trainer.predict(val_ds).predictions.argmax(-1)
    oof_preds[val_idx] = preds
    score = f1_score(y[val_idx], preds, average="macro")
    fold_scores.append(score)
    print(f"  Fold {fold+1} F1: {score:.5f}")

cv = f1_score(y, oof_preds, average="macro")
print(f"\nEXP-01 CV macro F1 (OOF): {cv:.5f}")
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
PYEOF

SCORE_01=$(grep "EXP-01 CV macro F1" experiments/exp01_bertimbau.log | grep -oP '[0-9]+\.[0-9]+' || echo "???")
echo -e "EXP-01\tBERTimbau-base\t${SCORE_01}\t6\t30\tkeep\tlr=2e-5 ep=5 max_len=256 bf16 batch=64" >> experiments/results.tsv
echo "EXP-01 concluído — CV F1: $SCORE_01"

# ── 7. EXP-03: BioBERTpt ──────────────────────────────────
echo ""
echo "=================================================="
echo " EXP-03: BioBERTpt (pucpr/biobertpt-all)"
echo "=================================================="

python - <<'PYEOF' | tee experiments/exp03_biobertpt.log
import torch, numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from prepare import load_data, make_groups, RANDOM_STATE, N_SPLITS

MODEL_NAME = "pucpr/biobertpt-all"
NUM_LABELS = 7
MAX_LEN    = 256
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 2e-5

print(f"Modelo: {MODEL_NAME}")
train_df, test_df = load_data()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["report"], truncation=True, max_length=MAX_LEN)

groups = make_groups(train_df)
y = train_df["target"].astype(int).values
gkf = GroupKFold(n_splits=N_SPLITS)
oof_preds = np.zeros(len(train_df), dtype=int)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, y, groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    tr_ds = Dataset.from_dict({
        "report": train_df.iloc[tr_idx]["report"].tolist(),
        "label":  y[tr_idx].tolist()
    }).map(tokenize, batched=True)

    val_ds = Dataset.from_dict({
        "report": train_df.iloc[val_idx]["report"].tolist(),
        "label":  y[val_idx].tolist()
    }).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=f"experiments/exp03_fold{fold+1}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tr_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    preds = trainer.predict(val_ds).predictions.argmax(-1)
    oof_preds[val_idx] = preds
    score = f1_score(y[val_idx], preds, average="macro")
    fold_scores.append(score)
    print(f"  Fold {fold+1} F1: {score:.5f}")

cv = f1_score(y, oof_preds, average="macro")
print(f"\nEXP-03 CV macro F1 (OOF): {cv:.5f}")
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
PYEOF

SCORE_03=$(grep "EXP-03 CV macro F1" experiments/exp03_biobertpt.log | grep -oP '[0-9]+\.[0-9]+' || echo "???")
echo -e "EXP-03\tBioBERTpt-all\t${SCORE_03}\t6\t35\tkeep\tlr=2e-5 ep=5 max_len=256 bf16 batch=64" >> experiments/results.tsv
echo "EXP-03 concluído — CV F1: $SCORE_03"

# ── 8. EXP-04: mDeBERTa-v3 ────────────────────────────────
echo ""
echo "=================================================="
echo " EXP-04: mDeBERTa-v3 (microsoft/mdeberta-v3-base)"
echo "=================================================="

python - <<'PYEOF' | tee experiments/exp04_mdeberta.log
import torch, numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from prepare import load_data, make_groups, RANDOM_STATE, N_SPLITS

MODEL_NAME = "microsoft/mdeberta-v3-base"
NUM_LABELS = 7
MAX_LEN    = 512
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 1e-5

print(f"Modelo: {MODEL_NAME}")
train_df, test_df = load_data()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["report"], truncation=True, max_length=MAX_LEN)

groups = make_groups(train_df)
y = train_df["target"].astype(int).values
gkf = GroupKFold(n_splits=N_SPLITS)
oof_preds = np.zeros(len(train_df), dtype=int)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, y, groups)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    tr_ds = Dataset.from_dict({
        "report": train_df.iloc[tr_idx]["report"].tolist(),
        "label":  y[tr_idx].tolist()
    }).map(tokenize, batched=True)

    val_ds = Dataset.from_dict({
        "report": train_df.iloc[val_idx]["report"].tolist(),
        "label":  y[val_idx].tolist()
    }).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=f"experiments/exp04_fold{fold+1}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=4,  # simula batch=64
        dataloader_num_workers=4,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tr_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    preds = trainer.predict(val_ds).predictions.argmax(-1)
    oof_preds[val_idx] = preds
    score = f1_score(y[val_idx], preds, average="macro")
    fold_scores.append(score)
    print(f"  Fold {fold+1} F1: {score:.5f}")

cv = f1_score(y, oof_preds, average="macro")
print(f"\nEXP-04 CV macro F1 (OOF): {cv:.5f}")
print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
PYEOF

SCORE_04=$(grep "EXP-04 CV macro F1" experiments/exp04_mdeberta.log | grep -oP '[0-9]+\.[0-9]+' || echo "???")
echo -e "EXP-04\tmdeberta-v3-base\t${SCORE_04}\t8\t45\tkeep\tlr=1e-5 ep=5 max_len=512 bf16 batch=16 grad_acc=4" >> experiments/results.tsv
echo "EXP-04 concluído — CV F1: $SCORE_04"

# ── 9. Resumo final ───────────────────────────────────────
echo ""
echo "=================================================="
echo " RESUMO DOS EXPERIMENTOS"
echo "=================================================="
echo ""
echo "Arquivo results.tsv atualizado:"
cat experiments/results.tsv
echo ""
echo "Top experimentos por CV F1:"
sort -t$'\t' -k3 -rn experiments/results.tsv | head -6
echo ""
echo "=================================================="
echo " PRÓXIMOS PASSOS (ver plan_gpu.md)"
echo "=================================================="
echo " EXP-02: BERTimbau Large (~90 min, ~18GB VRAM)"
echo " EXP-05: Ensemble BERT + TF-IDF (melhor potencial)"
echo " EXP-06: Autoresearch overnight"
echo " EXP-08: Focal Loss + Oversampling para classes 5/6"
echo ""
echo "Para commitar resultados:"
echo "  git add experiments/results.tsv experiments/*.log"
echo "  git commit -m 'GPU runs: EXP-01,03,04 concluídos'"
echo ""
echo "Setup completo!"
