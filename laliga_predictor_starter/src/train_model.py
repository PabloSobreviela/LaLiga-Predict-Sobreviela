# src/train_model.py
"""
Train a multinomial logistic regression with a robust, imputed pipeline.
Saves bundle to CONFIG['model_path'] containing:
- pipeline
- feature_order
- feature_means (for app vector defaults)
- meta
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG

DATA_PROC = Path(CONFIG["processed_data_path"])
MODEL_PATH = Path(CONFIG["model_path"])

# Features used by the app when building the single-row vector
FEATURE_ORDER = [
    "elo_home","elo_away","elo_diff",
    "home_last_gf","away_last_gf",
    "home_last_ga","away_last_ga",
    "home_last_gd","away_last_gd",
    "home_last_sot_for","away_last_sot_for",
    "home_last_sot_against","away_last_sot_against",
    "rest_days_home","rest_days_away",
    "form_gd_diff","form_sot_for_diff","form_sot_against_diff","rest_days_diff",
]

def _ensure_labels(df: pd.DataFrame) -> pd.Series:
    """
    Ensure we have a categorical label in H/D/A; accept FTR or derive from goals.
    """
    if "FTR" in df.columns and df["FTR"].notna().any():
        y = df["FTR"].map({"H":0,"D":1,"A":2})
        if y.notna().any():
            return y.astype(int)
    # derive if needed
    if "FTHG" in df.columns and "FTAG" in df.columns:
        y = np.where(df["FTHG"]>df["FTAG"], 0, np.where(df["FTHG"]<df["FTAG"], 2, 1))
        return pd.Series(y, index=df.index)
    raise KeyError("No label found. Need FTR or (FTHG/FTAG).")

def train() -> Tuple[float, float, Dict]:
    """
    Train the model and write a bundle to models/model.joblib
    Returns: (accuracy, logloss, meta)
    """
    feats_path = DATA_PROC / "features_train.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(f"{feats_path} not found. Build features first.")

    df = pd.read_parquet(feats_path)

    # Keep only rows with targets available
    y = _ensure_labels(df)
    X = df[FEATURE_ORDER].copy()

    # Build pipeline: impute -> scale -> multinomial logistic regression
    numeric = FEATURE_ORDER
    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric)],
        remainder="drop"
    )
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=None
    )
    pipe = Pipeline([
        ("pre", pre),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logreg", clf),
    ])

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    # Evaluate
    yhat = pipe.predict(X_val)
    yproba = pipe.predict_proba(X_val)
    acc = float(accuracy_score(y_val, yhat))
    ll = float(log_loss(y_val, yproba))

    # Save bundle
    feature_means = {c: float(pd.to_numeric(X[c], errors="coerce").dropna().mean()) for c in FEATURE_ORDER}
    bundle = {
        "pipeline": pipe,
        "feature_order": FEATURE_ORDER,
        "feature_means": feature_means,
        "meta": {"acc_val": acc, "logloss_val": ll},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    return acc, ll, bundle["meta"]
