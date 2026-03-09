"""
prepare.py — FIXED, DO NOT MODIFY
Data loading, evaluation, and submission logic for SPR 2026.

Competition: SPR 2026 Mammography Report Classification
Task: Predict BI-RADS category (0-6) from mammography report text (Portuguese)
Metric: Macro F1
"""

import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SUB_PATH   = os.path.join(DATA_DIR, "submission.csv")

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS     = 5
TARGET_COL   = "target"
TEXT_COL     = "report"
ID_COL       = "ID"

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    assert {ID_COL, TEXT_COL, TARGET_COL}.issubset(train.columns)
    assert {ID_COL, TEXT_COL}.issubset(test.columns)
    return train, test

def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def make_groups(df: pd.DataFrame) -> np.ndarray:
    """GroupKFold groups — prevents leakage from duplicate reports."""
    return df[TEXT_COL].apply(stable_hash).values

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred) -> float:
    """Primary metric: macro F1."""
    return f1_score(y_true, y_pred, average="macro")

# ── Submission ─────────────────────────────────────────────────────────────────
def save_submission(test_df: pd.DataFrame, preds: np.ndarray, path: str = "submission.csv"):
    sub = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: preds.astype(int)})
    sub.to_csv(path, index=False)
    print(f"Submission saved to {path} ({len(sub)} rows)")
    return sub
