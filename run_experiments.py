"""
run_experiments.py — Otimizado para RTX 5090 (32 GB VRAM, Blackwell)
======================================================================
Roda sequencialmente:
  EXP-01  BERTimbau Base
  EXP-02  BERTimbau Large
  EXP-03  BioBERTpt
  EXP-04  mDeBERTa-v3-base
  EXP-07  BiomedBERT-EN
  EXP-08  BERTimbau Base + Focal Loss
  EXP-05  Ensemble dos melhores OOFs

Uso:
    python run_experiments.py                        # todos
    python run_experiments.py --exp EXP-01           # so um
    python run_experiments.py --exp EXP-01,EXP-03    # lista
"""

import argparse
import gc
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EXP_DIR     = os.path.join(BASE_DIR, "experiments")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")
RESULTS_TSV = os.path.join(EXP_DIR, "results.tsv")
SUB_DIR     = os.path.join(BASE_DIR, "submissions")

os.makedirs(EXP_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SUB_DIR,  exist_ok=True)

sys.path.insert(0, BASE_DIR)
from prepare import load_data, make_groups, evaluate, save_submission, N_SPLITS

# ── GPU info ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    VRAM_GB = props.total_memory / 1e9
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {VRAM_GB:.1f} GB")
    IS_5090 = VRAM_GB > 20          # 5090 tem ~32 GB; 2080 tem 8 GB
else:
    VRAM_GB  = 0
    IS_5090  = False

NUM_LABELS = 7

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO DOS EXPERIMENTOS
# Batch sizes e flags calibrados para RTX 5090 (32 GB, bf16 nativo).
# Fallback automatico para 8 GB (fp16, batch menor) caso GPU diferente.
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(model_name, max_len, batch_5090, batch_8gb, grad_accum_8gb,
         epochs, lr, warmup, wd, focal, notes, focal_gamma=2.0):
    if IS_5090:
        batch      = batch_5090
        grad_accum = 1
        use_bf16   = True
        use_fp16   = False
    else:
        batch      = batch_8gb
        grad_accum = grad_accum_8gb
        use_bf16   = False
        use_fp16   = True
    return dict(
        model_name=model_name, max_len=max_len,
        batch_size=batch, grad_accum=grad_accum,
        epochs=epochs, lr=lr, warmup=warmup, weight_decay=wd,
        focal_loss=focal, focal_gamma=focal_gamma,
        bf16=use_bf16, fp16=use_fp16,
        notes=notes,
    )

EXPERIMENTS = {
    # ── Prioridade ⭐ ──────────────────────────────────────────────────────────
    "EXP-01": _cfg(
        model_name   = "neuralmind/bert-base-portuguese-cased",
        max_len=256,  batch_5090=64, batch_8gb=16, grad_accum_8gb=2,
        epochs=5, lr=2e-5, warmup=0.1, wd=0.01, focal=False,
        notes="BERTimbau-base lr=2e-5 ep=5 max_len=256",
    ),
    "EXP-02": _cfg(
        model_name   = "neuralmind/bert-large-portuguese-cased",
        max_len=256,  batch_5090=32, batch_8gb=4, grad_accum_8gb=8,
        epochs=5, lr=1e-5, warmup=0.1, wd=0.01, focal=False,
        notes="BERTimbau-large lr=1e-5 ep=5 max_len=256",
    ),
    "EXP-03": _cfg(
        model_name   = "pucpr/biobertpt-all",
        max_len=256,  batch_5090=64, batch_8gb=16, grad_accum_8gb=2,
        epochs=5, lr=2e-5, warmup=0.1, wd=0.01, focal=False,
        notes="BioBERTpt lr=2e-5 ep=5 max_len=256",
    ),
    "EXP-04": _cfg(
        model_name   = "microsoft/mdeberta-v3-base",
        max_len=512,  batch_5090=32, batch_8gb=8, grad_accum_8gb=4,
        epochs=5, lr=1e-5, warmup=0.1, wd=0.01, focal=False,
        notes="mDeBERTa-v3-base lr=1e-5 ep=5 max_len=512",
    ),
    "EXP-07": _cfg(
        model_name   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        max_len=256,  batch_5090=64, batch_8gb=16, grad_accum_8gb=2,
        epochs=5, lr=2e-5, warmup=0.1, wd=0.01, focal=False,
        notes="BiomedBERT-EN lr=2e-5 ep=5 max_len=256",
    ),
    "EXP-08": _cfg(
        model_name   = "neuralmind/bert-base-portuguese-cased",
        max_len=256,  batch_5090=64, batch_8gb=16, grad_accum_8gb=2,
        epochs=5, lr=2e-5, warmup=0.1, wd=0.01, focal=True, focal_gamma=2.0,
        notes="BERTimbau-base FocalLoss gamma=2 lr=2e-5 ep=5",
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ReportDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FocalTrainer(Trainer):
    def __init__(self, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self._focal = FocalLoss(gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self._focal(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_results():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write("exp_id\tmodel\tcv_f1\tvram_gb\ttime_min\tstatus\tnotes\n")
            f.write("EXP-00\tTF-IDF+LinearSVC\t0.748\t0\t2\tkeep\tbaseline local\n")
        print(f"Criado: {RESULTS_TSV}")


def log_result(exp_id, model, cv_f1, vram_gb, time_min, status, notes):
    row = f"{exp_id}\t{model}\t{cv_f1:.5f}\t{vram_gb}\t{time_min}\t{status}\t{notes}\n"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(row)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {exp_id} | cv_f1={cv_f1:.5f} | {status.upper()}")
    print(f"  Tempo: {time_min:.1f} min | VRAM pico: {vram_gb:.1f} GB")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TREINO DE UM EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_id, cfg, train_df, test_df):
    print(f"\n{'#' * 60}")
    print(f"  {exp_id}: {cfg['model_name']}")
    print(f"  batch={cfg['batch_size']} | grad_accum={cfg['grad_accum']}")
    print(f"  max_len={cfg['max_len']} | epochs={cfg['epochs']}")
    print(f"  bf16={cfg['bf16']} | fp16={cfg['fp16']} | focal={cfg['focal_loss']}")
    print(f"{'#' * 60}\n")

    model_name = cfg["model_name"]
    max_len    = cfg["max_len"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    epochs     = cfg["epochs"]
    lr         = cfg["lr"]
    wd         = cfg["weight_decay"]
    warmup_r   = cfg["warmup"]

    X_text  = train_df["report"].fillna("").values
    y       = train_df["target"].astype(int).values
    groups  = make_groups(train_df)
    classes = sorted(np.unique(y))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gkf       = GroupKFold(n_splits=N_SPLITS)
    oof_proba = np.zeros((len(train_df), NUM_LABELS))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.array(classes)[np.argmax(logits, axis=1)]
        return {"macro_f1": evaluate(labels, preds)}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_text, y, groups)):
        print(f"\n  Fold {fold + 1}/{N_SPLITS}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
        )

        total_steps  = (len(tr_idx) // (batch_size * grad_accum)) * epochs
        warmup_steps = max(1, int(total_steps * warmup_r))
        out_dir      = os.path.join(CKPT_DIR, f"{exp_id}_fold{fold + 1}")

        train_args = TrainingArguments(
            output_dir                  = out_dir,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = 64,
            gradient_accumulation_steps = grad_accum,
            learning_rate               = lr,
            weight_decay                = wd,
            warmup_steps                = warmup_steps,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "macro_f1",
            greater_is_better           = True,
            bf16                        = cfg["bf16"],
            fp16                        = cfg["fp16"],
            tf32                        = cfg["bf16"],   # TF32 quando bf16
            dataloader_num_workers      = 4,
            report_to                   = "none",
            logging_steps               = 50,
        )

        train_ds = ReportDataset(X_text[tr_idx], tokenizer, max_len, y[tr_idx])
        val_ds   = ReportDataset(X_text[val_idx], tokenizer, max_len, y[val_idx])

        if cfg["focal_loss"]:
            trainer = FocalTrainer(
                focal_gamma=cfg.get("focal_gamma", 2.0),
                model=model, args=train_args,
                train_dataset=train_ds, eval_dataset=val_ds,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = Trainer(
                model=model, args=train_args,
                train_dataset=train_ds, eval_dataset=val_ds,
                compute_metrics=compute_metrics,
            )

        trainer.train()

        pred_out       = trainer.predict(val_ds)
        proba          = torch.softmax(torch.tensor(pred_out.predictions), dim=1).numpy()
        oof_proba[val_idx] = proba

        fold_f1 = evaluate(y[val_idx], np.array(classes)[np.argmax(proba, axis=1)])
        print(f"  Fold {fold + 1} macro F1: {fold_f1:.5f}")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ── OOF ───────────────────────────────────────────────────────────────────
    oof_preds = np.array(classes)[np.argmax(oof_proba, axis=1)]
    cv_score  = evaluate(y, oof_preds)
    elapsed   = (time.time() - t0) / 60
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    oof_path  = os.path.join(EXP_DIR, f"oof_proba_{exp_id}.npy")
    np.save(oof_path, oof_proba)
    print(f"  OOF salvo: {oof_path}")

    # ── Retrain full + submissao ───────────────────────────────────────────────
    print(f"\n  Retreinando no dataset completo para submissao...")
    model_full = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    full_ds  = ReportDataset(X_text, tokenizer, max_len, y)
    test_ds  = ReportDataset(test_df["report"].fillna("").values, tokenizer, max_len)

    total_steps_full  = (len(X_text) // (batch_size * grad_accum)) * epochs
    warmup_steps_full = max(1, int(total_steps_full * warmup_r))

    args_full = TrainingArguments(
        output_dir                  = os.path.join(CKPT_DIR, f"{exp_id}_full"),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        weight_decay                = wd,
        warmup_steps                = warmup_steps_full,
        eval_strategy               = "no",
        save_strategy               = "no",
        bf16                        = cfg["bf16"],
        fp16                        = cfg["fp16"],
        tf32                        = cfg["bf16"],
        dataloader_num_workers      = 4,
        report_to                   = "none",
        logging_steps               = 100,
    )

    if cfg["focal_loss"]:
        trainer_full = FocalTrainer(
            focal_gamma=cfg.get("focal_gamma", 2.0),
            model=model_full, args=args_full, train_dataset=full_ds,
        )
    else:
        trainer_full = Trainer(model=model_full, args=args_full, train_dataset=full_ds)

    trainer_full.train()

    pred_test  = trainer_full.predict(test_ds)
    proba_test = torch.softmax(torch.tensor(pred_test.predictions), dim=1).numpy()
    test_preds = np.array(classes)[np.argmax(proba_test, axis=1)]

    sub_path = os.path.join(SUB_DIR, f"submission_{exp_id}.csv")
    save_submission(test_df, test_preds, path=sub_path)

    del model_full, trainer_full
    gc.collect()
    torch.cuda.empty_cache()

    # ── Log ───────────────────────────────────────────────────────────────────
    status      = "keep" if cv_score > 0.748 else "review"
    model_short = model_name.split("/")[-1]
    log_result(exp_id, model_short, cv_score, round(vram_peak, 1), round(elapsed, 1), status, cfg["notes"])

    return cv_score, oof_proba


# ══════════════════════════════════════════════════════════════════════════════
# EXP-05: ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble(train_df, test_df, completed_ids):
    print(f"\n{'#' * 60}")
    print(f"  EXP-05: Ensemble de {completed_ids}")
    print(f"{'#' * 60}\n")

    y       = train_df["target"].astype(int).values
    classes = sorted(np.unique(y))

    oofs, names = [], []
    for eid in completed_ids:
        path = os.path.join(EXP_DIR, f"oof_proba_{eid}.npy")
        if os.path.exists(path):
            oofs.append(np.load(path))
            names.append(eid)
            print(f"  Carregado: {path}")
        else:
            print(f"  AVISO: OOF nao encontrado para {eid}")

    if len(oofs) < 2:
        print("  Ensemble requer >= 2 OOFs. Pulando.")
        return None

    # Media simples
    oof_mean  = np.mean(oofs, axis=0)
    score_eq  = evaluate(y, np.array(classes)[np.argmax(oof_mean, axis=1)])
    print(f"  Media igualitaria: {score_eq:.5f}")

    # Grid search de pesos (para 2 modelos)
    best_score, best_w = score_eq, None
    n = len(oofs)
    if n == 2:
        for w in np.arange(0.0, 1.01, 0.05):
            blend  = w * oofs[0] + (1 - w) * oofs[1]
            preds  = np.array(classes)[np.argmax(blend, axis=1)]
            s      = evaluate(y, preds)
            if s > best_score:
                best_score = s
                best_w = [round(w, 2), round(1 - w, 2)]
    elif n >= 3:
        # Busca simples: iterar sobre vizinhanca da media
        from itertools import product
        grid = np.arange(0.0, 1.01, 0.1)
        for combo in product(grid, repeat=n - 1):
            last = 1.0 - sum(combo)
            if last < 0 or last > 1:
                continue
            weights = list(combo) + [last]
            blend   = sum(w * o for w, o in zip(weights, oofs))
            preds   = np.array(classes)[np.argmax(blend, axis=1)]
            s       = evaluate(y, preds)
            if s > best_score:
                best_score = s
                best_w = [round(ww, 2) for ww in weights]

    if best_w is None:
        best_w = [round(1.0 / n, 2)] * n

    print(f"  Pesos otimizados: {dict(zip(names, best_w))}")
    print(f"  CV macro F1 ensemble: {best_score:.5f}")

    oof_path = os.path.join(EXP_DIR, "oof_proba_EXP-05.npy")
    np.save(oof_path, oof_mean)

    status = "keep" if best_score > 0.748 else "review"
    log_result(
        "EXP-05",
        "+".join(names),
        best_score,
        0.0, 0.0, status,
        f"ensemble pesos={best_w}",
    )
    return best_score


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", default=None,
        help="Experimentos (ex: EXP-01,EXP-03). Padrao: todos."
    )
    args = parser.parse_args()

    init_results()

    train_df, test_df = load_data()
    print(f"\nTrain: {train_df.shape} | Test: {test_df.shape}")
    print(f"Distribuicao de classes:\n{train_df['target'].value_counts().sort_index()}\n")

    all_exp_order = ["EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-07", "EXP-08", "EXP-05"]

    if args.exp:
        exps_to_run = [e.strip() for e in args.exp.split(",")]
    else:
        exps_to_run = all_exp_order

    completed = []
    for exp_id in exps_to_run:
        if exp_id == "EXP-05":
            run_ensemble(train_df, test_df, completed)
            continue

        if exp_id not in EXPERIMENTS:
            print(f"Experimento {exp_id} nao configurado. Pulando.")
            continue

        try:
            run_experiment(exp_id, EXPERIMENTS[exp_id], train_df, test_df)
            completed.append(exp_id)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] {exp_id}: GPU sem memoria. Pulando.")
                torch.cuda.empty_cache()
                gc.collect()
                log_result(exp_id, EXPERIMENTS[exp_id]["model_name"].split("/")[-1],
                           0.0, VRAM_GB, 0.0, "crash", "OOM — reduzir batch_size")
            else:
                print(f"\n[ERRO] {exp_id}: {e}")
                import traceback; traceback.print_exc()
                log_result(exp_id, EXPERIMENTS[exp_id]["model_name"].split("/")[-1],
                           0.0, 0.0, 0.0, "crash", str(e)[:120])
        except Exception as e:
            print(f"\n[ERRO] {exp_id}: {e}")
            import traceback; traceback.print_exc()
            log_result(exp_id, EXPERIMENTS[exp_id]["model_name"].split("/")[-1],
                       0.0, 0.0, 0.0, "crash", str(e)[:120])

    # ── Relatorio final ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TODOS OS EXPERIMENTOS CONCLUIDOS")
    print("=" * 60)
    if os.path.exists(RESULTS_TSV):
        df_res = pd.read_csv(RESULTS_TSV, sep="\t")
        print("\n")
        print(df_res.sort_values("cv_f1", ascending=False).to_string(index=False))

    print(f"\nResultados em : {RESULTS_TSV}")
    print(f"Submissoes em : {SUB_DIR}")
    print(f"Checkpoints em: {CKPT_DIR}  (nao subir no GitHub)")


if __name__ == "__main__":
    main()
