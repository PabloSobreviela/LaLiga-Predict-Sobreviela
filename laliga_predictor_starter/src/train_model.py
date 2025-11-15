# src/train_model.py — robust trainer with auto feature build and label fallbackhttps://github.com/PabloSobreviela/LaLiga-Predict-Sobreviela/blob/main/laliga_predictor_starter/src/train_model.py

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
    """
    Canonical feature order we aim for. The trainer will intersect this list
    with whatever is actually available in features.parquet to avoid crashes
    if new features are added or some are missing.
    """
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
    """
    Ensure data/processed/features.parquet exists.
    If not, try to auto-build by importing src.features and calling build_features()
    (which scans data/raw). If no raw CSVs are present, raise a clear error.
    """
    proc_dir = Path(CONFIG["processed_data_path"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    feats_path = proc_dir / "features.parquet"
    if feats_path.exists():
        return feats_path

    # Try auto-build only if there are raw CSVs available.
    raw_dir = Path(CONFIG["raw_data_path"])
    csvs = sorted(raw_dir.glob(f"{CONFIG['league']}_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{feats_path} not found and no raw CSVs were found in {raw_dir}.\n"
            "Use the app's **Train / Data → Update dataset & retrain** to fetch CSVs first."
        )

    try:
        from . import features as F
        # Let features.py scan data/raw itself; seasons=None means "use what's there".
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


def _ensure_target(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric target series 0/1/2 where 0=H,1=D,2=A.

    Priority:
    1) Use 'target' if present and numeric.
    2) Map 'FTR' (H/D/A) if present.
    3) Derive from 'FTHG'/'FTAG' if present.

    Raise a clear error if none of the above are available.
    """
    if "target" in df.columns:
        s = df["target"]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_numeric_dtype(s):
            # Force to int and valid range if possible
            s = s.astype(float).round().astype(int)
            if set(np.unique(s)).issubset({0, 1, 2}):
                return s
        # Fall through if weird type/values.

    if "FTR" in df.columns:
        mapped = df["FTR"].map({"H": 0, "D": 1, "A": 2})
        if mapped.notna().any():
            # If some rows NaN (missing FTR), try fallback from goals where possible
            if "FTHG" in df.columns and "FTAG" in df.columns:
                need = mapped.isna()
                if need.any():
                    hg = df.loc[need, "FTHG"]
                    ag = df.loc[need, "FTAG"]
                    mapped.loc[need] = np.where(hg > ag, 0, np.where(hg < ag, 2, 1))
            mapped = mapped.fillna(1).astype(int)  # default draws if totally missing
            return mapped

    if "FTHG" in df.columns and "FTAG" in df.columns:
        return np.where(df["FTHG"] > df["FTAG"], 0,
                        np.where(df["FTHG"] < df["FTAG"], 2, 1)).astype(int)

    raise KeyError(
        "No label found.\n"
        "Need one of: numeric 'target' (0/1/2), or categorical 'FTR' (H/D/A), "
        "or goals 'FTHG' & 'FTAG' to derive it. Rebuild features from the app."
    )


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Intersect expected features with df columns. Enforce that we still have a
    reasonable number (>5) to train a stable model; otherwise raise a helpful error.
    """
    expected = _expected_feature_order()
    cols = [c for c in expected if c in df.columns]
    if len(cols) < 6:
        raise KeyError(
            "Too few training features found in features.parquet. "
            f"Expected a subset of {expected}, got only {cols}. "
            "Please re-run **Train / Data → Update dataset & retrain** to rebuild features."
        )
    return cols


def train() -> Tuple[float, float, dict]:
    """
    Train a multinomial logistic regression model.
    Returns (accuracy, logloss, meta_dict).
    Saves bundle to models/model.joblib for the app to load.
    """
    feats_path = _ensure_features_exist()
    df = pd.read_parquet(feats_path)

    # Build labels (0/1/2)
    y = _ensure_target(df).values

    # Choose features that exist
    feature_order = _select_features(df)
    X = df[feature_order].astype(float).copy()

    # Fill NaNs defensively (LR can't handle NaN)
    X = X.fillna(X.mean())

    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            C=1.0,
            n_jobs=None,           # Streamlit Cloud often limits n_jobs; keep None
        )),
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)
    acc = accuracy_score(y_val, proba.argmax(axis=1))
    ll = log_loss(y_val, proba, labels=[0, 1, 2])

    # Persist bundle
    feature_means = pd.Series(X.mean(), index=feature_order).to_dict()
    bundle = {
        "pipeline": pipe,
        "feature_order": feature_order,
        "feature_means": feature_means,
        "meta": {
            "classes": [0, 1, 2],           # H/D/A order used by predict_proba_HDA
            "n_samples": int(len(df)),
            "val_acc": float(acc),
            "val_logloss": float(ll),
        },
    }

    model_path = Path(CONFIG["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    return float(acc), float(ll), bundle["meta"]
