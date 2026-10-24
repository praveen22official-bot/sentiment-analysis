"""End-to-end churn pipeline: data -> cleaning -> features -> XGBoost -> saved model.

Uses a synthetic telecom-style dataset so the repo runs anywhere. To use real data,
pass --csv path/to/data.csv with a binary `churn` column.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUM = ["tenure_months", "monthly_charges", "support_calls", "late_payments"]
CAT = ["contract", "internet_service", "payment_method"]


def make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.normal(65, 25, n).clip(15, 150).round(2),
            "support_calls": rng.poisson(1.5, n),
            "late_payments": rng.poisson(0.6, n),
            "contract": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]),
            "internet_service": rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.4, 0.15]),
            "payment_method": rng.choice(["card", "bank", "cheque"], n),
        }
    )
    logit = (
        -1.2
        - 0.035 * df.tenure_months
        + 0.015 * df.monthly_charges
        + 0.30 * df.support_calls
        + 0.45 * df.late_payments
        + np.where(df.contract == "month-to-month", 1.0, np.where(df.contract == "one-year", -0.2, -0.9))
        + np.where(df.internet_service == "fiber", 0.4, 0)
    )
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for c in CAT:
        df[c] = df[c].fillna("unknown").astype(str)
    return df


def build_model() -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, eval_metric="logloss", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def train(df: pd.DataFrame, out: str = "model.joblib") -> dict:
    df = clean(df)
    X, y = df[NUM + CAT], df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 3),
        "accuracy": round(float(accuracy_score(yte, pred)), 3),
        "f1": round(float(f1_score(yte, pred)), 3),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 3),
    }
    joblib.dump(model, out)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="model.joblib")
    a = ap.parse_args()
    data = pd.read_csv(a.csv) if a.csv else make_synthetic()
    print(json.dumps(train(data, a.out), indent=2))
