"""
train.py — AGENT MAY MODIFY THIS FILE
Experiment: v3 — Ensemble TF-IDF (word+char) + LogisticRegression com class weights

Competition: SPR 2026 Mammography Report Classification
Metric: macro F1
Baseline public LB: 0.773 (TF-IDF + LinearSVC)

Key facts:
- 18,272 train samples, 7 classes (BI-RADS 0-6), Portuguese text
- SEVERE imbalance: class 2 = 15,968 vs class 5 = 29 samples
- 9,141 duplicate reports → GroupKFold prevents leakage
- ~400 chars avg report length
"""

import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

from prepare import (
    load_data, make_groups, evaluate, save_submission,
    RANDOM_STATE, N_SPLITS
)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_ws_re = re.compile(r"[ \t]+")
_nl_re = re.compile(r"\n{2,}")

_ABBREV_MAP = {
    r"\bbi-?rads\b": "birads",
    r"\bcalc\.?\b": "calcificacao",
    r"\bnod\.?\b": "nodulo",
    r"\bdx\.?\b": "diagnostico",
    r"\blt\.?\b": "lateral",
    r"\bcc\b": "craniocaudal",
    r"\bmlo\b": "mediolateral",
}

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _ws_re.sub(" ", s)
    s = _nl_re.sub("\n", s)
    for pattern, replacement in _ABBREV_MAP.items():
        s = re.sub(pattern, replacement, s)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorizer():
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=80000,
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_svc():
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000,
    )

def build_lr():
    return LogisticRegression(
        C=4.0,
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run():
    train, test = load_data()

    train["text"] = train["report"].apply(clean_text)
    test["text"]  = test["report"].apply(clean_text)

    X_text = train["text"].values
    y      = train["target"].astype(int).values
    groups = make_groups(train)
    classes = sorted(np.unique(y))

    gkf = GroupKFold(n_splits=N_SPLITS)

    # Collect OOF probabilities from both models
    oof_proba_svc = np.zeros((len(train), len(classes)))
    oof_proba_lr  = np.zeros((len(train), len(classes)))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_text, y, groups)):
        vec = build_vectorizer()
        X_tr = vec.fit_transform(X_text[tr_idx])
        X_val = vec.transform(X_text[val_idx])

        # LinearSVC → decision function as soft scores
        svc = build_svc()
        svc.fit(X_tr, y[tr_idx])
        dec = svc.decision_function(X_val)
        # Softmax-normalize decision function for pseudo-probabilities
        dec_exp = np.exp(dec - dec.max(axis=1, keepdims=True))
        oof_proba_svc[val_idx] = dec_exp / dec_exp.sum(axis=1, keepdims=True)

        # LogisticRegression → true probabilities
        lr = build_lr()
        lr.fit(X_tr, y[tr_idx])
        oof_proba_lr[val_idx] = lr.predict_proba(X_val)

        # Per-fold score (ensemble)
        ensemble = 0.5 * oof_proba_svc[val_idx] + 0.5 * oof_proba_lr[val_idx]
        fold_preds = np.argmax(ensemble, axis=1)
        # Map back to original class labels
        fold_preds_mapped = np.array(classes)[fold_preds]
        score = evaluate(y[val_idx], fold_preds_mapped)
        print(f"  Fold {fold+1}: macro F1 = {score:.5f}")

    # OOF ensemble
    ensemble_oof = 0.5 * oof_proba_svc + 0.5 * oof_proba_lr
    oof_preds = np.array(classes)[np.argmax(ensemble_oof, axis=1)]
    cv_score = evaluate(y, oof_preds)
    print(f"\n  CV macro F1 (OOF ensemble): {cv_score:.5f}")

    # ── Retrain on full data ────────────────────────────────────────────────
    vec_final = build_vectorizer()
    X_full = vec_final.fit_transform(X_text)
    X_test_vec = vec_final.transform(test["text"].values)

    svc_final = build_svc()
    svc_final.fit(X_full, y)

    lr_final = build_lr()
    lr_final.fit(X_full, y)

    # Predict test
    dec_test = svc_final.decision_function(X_test_vec)
    dec_exp  = np.exp(dec_test - dec_test.max(axis=1, keepdims=True))
    proba_svc_test = dec_exp / dec_exp.sum(axis=1, keepdims=True)
    proba_lr_test  = lr_final.predict_proba(X_test_vec)

    ensemble_test = 0.5 * proba_svc_test + 0.5 * proba_lr_test
    test_preds = np.array(classes)[np.argmax(ensemble_test, axis=1)]

    save_submission(test, test_preds, path="submission.csv")

    return cv_score


if __name__ == "__main__":
    score = run()
    print(f"\nFINAL SCORE: {score:.5f}")
