# src/train_model.py — trainer used by the working app (labels with fallback)

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG


def _expected_feature_order() -> List[str]:
    return [
        "elo_home", "elo_away", "elo_diff",
        "home_gf_roll", "away_gf_roll",
        "home_ga_roll", "away_ga_roll",
        "home_gd_roll", "away_gd_roll",
        "home_sotf_roll", "away_sotf_roll",
        "home_sota_roll", "away_sota_roll",
        "home_rest_days", "away_rest_days",
        "form_gd_diff", "form_sot_for_diff",
        "form_sot_against_diff", "rest_days_diff",
    ]


def _ensure_features_exist() -> Path:
    proc_dir = Path(CONFIG["processed_data_path"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    feats_path = proc_dir / "features.parquet"
    if feats_path.exists():
        return feats_path

    raw_dir = Path(CONFIG["raw_data_path"])
    csvs = sorted(raw_dir.glob(f"{CONFIG['league']}_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{feats_path} not found and no raw CSVs were found in {raw_dir}.\n"
            "Use **Train / Data → Update dataset & retrain** in the app to fetch CSVs first."
        )

    # Try to auto-build if features are missing
    try:
        from . import features as F
        F.build_features(seasons=None)
    except Exception as e:
        raise FileNotFoundError(
            f"{feats_path} not found and auto-build failed: {e}"
        )

    if not feats_path.exists():
        raise FileNotFoundError(
            f"Feature build completed but {feats_path} is still missing. "
            "Please re-run **Update dataset & retrain**."
        )
    return feats_path


def _build_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Return (labels_0_1_2, mask_valid_rows).
    Prefer numeric 'target' → map 'FTR' → derive from 'FTHG'/'FTAG'.
    Drop rows that still lack a valid label.
    """
    y = pd.Series(np.nan, index=df.index, dtype="float")

    # 1) numeric 'target'
    if "target" in df.columns:
        y = pd.to_numeric(df["target"], errors="coerce")

    # 2) map FTR where y is NaN
    if "FTR" in df.columns:
        mapped = df["FTR"].map({"H": 0, "D": 1, "A": 2})
        y = y.where(~y.isna(), mapped)

    # 3) derive from goals
    if ("FTHG" in df.columns) and ("FTAG" in df.columns):
        need = y.isna()
        if need.any():
            hg = pd.to_numeric(df.loc[need, "FTHG"], errors="coerce")
            ag = pd.to_numeric(df.loc[need, "FTAG"], errors="coerce")
            y.loc[need] = np.where(hg > ag, 0, np.where(hg < ag, 2, 1))

    valid = y.isin([0, 1, 2])
    if valid.sum() == 0:
        raise KeyError(
            "No usable labels found in features.parquet (need one of: numeric 'target', 'FTR', or 'FTHG' & 'FTAG')."
        )
    return y[valid].astype(int), valid


def _select_features(df: pd.DataFrame) -> List[str]:
    expected = _expected_feature_order()
    cols = [c for c in expected if c in df.columns]
    if len(cols) < 6:
        raise KeyError(
            "Too few training features found. "
            f"Expected a subset of {expected}, got {cols}. Rebuild features from the app."
        )
    return cols


def train() -> Tuple[float, float, dict]:
    feats_path = _ensure_features_exist()
    df = pd.read_parquet(feats_path)

    # Labels + mask
    y, mask = _build_labels(df)

    # Features
    feature_order = _select_features(df)
    X = df.loc[mask, feature_order].copy().astype(float)
    X = X.fillna(X.mean(numeric_only=True))  # LR can't handle NaN

    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            C=1.0,
        )),
    ])

    pipe.fit(X_train, y_train)

    proba_val = pipe.predict_proba(X_val)
    acc = accuracy_score(y_val, proba_val.argmax(axis=1))
    ll = log_loss(y_val, proba_val, labels=[0, 1, 2])

    feature_means = pd.Series(X.mean(numeric_only=True), index=feature_order).to_dict()

    bundle = {
        "pipeline": pipe,
        "feature_order": feature_order,
        "feature_means": feature_means,
        "meta": {
            "classes": [0, 1, 2],
            "n_samples": int(X.shape[0]),
            "val_acc": float(acc),
            "val_logloss": float(ll),
        },
    }

    model_path = Path(CONFIG["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    return float(acc), float(ll), bundle["meta"]
