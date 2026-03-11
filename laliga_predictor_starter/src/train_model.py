import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.config import CONFIG

DATA_PROC = Path(CONFIG["processed_data_path"])
MODEL_PATH = Path(CONFIG["model_path"])
TARGET_LABELS = [0, 1, 2]

# Boosting rounds for iterative learning (more = better fit, slower)
DEFAULT_MAX_ITER = 300


def _ensure_labels(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of integer labels {H:0, D:1, A:2}.
    Prefer 'FTR' if present; otherwise derive from goals.
    """
    if "FTR" in df.columns:
        y = df["FTR"].map({"H": 0, "D": 1, "A": 2})
        if y.isna().any():
            raise KeyError("FTR has unexpected values; expected only H/D/A.")
        return y.astype(int)

    if {"FTHG", "FTAG"}.issubset(df.columns):
        res = np.where(
            df["FTHG"].values > df["FTAG"].values, 0,
            np.where(df["FTHG"].values < df["FTAG"].values, 2, 1)
        )
        return pd.Series(res, index=df.index, name="FTR_derived").astype(int)

    raise KeyError("No label found. Need 'FTR' (H/D/A), or 'FTHG' & 'FTAG' to derive it.")


def _select_features(df: pd.DataFrame) -> List[str]:
    """
    Choose model features from features.parquet.
    Exclude identifiers / labels; keep numeric columns only.
    """
    exclude = {"FTR", "FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    if not feats:
        raise ValueError("No numeric feature columns found after exclusion.")
    return feats


def _build_pipeline(max_iter: int = DEFAULT_MAX_ITER) -> Pipeline:
    """
    Impute NaNs (median) -> scale -> HistGradientBoosting.

    Gradient boosting learns iteratively over many rounds, typically
    achieving better accuracy than logistic regression on tabular data.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.12,
            random_state=42,
        )),
    ])


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split to simulate forward-in-time evaluation.
    """
    if "Date" not in df.columns:
        raise KeyError("Expected 'Date' column to do time split.")
    dfx = df.sort_values("Date").reset_index(drop=True)
    n = len(dfx)
    if n < 10:
        raise ValueError("Need at least 10 matches to train a stable model.")
    cut = min(max(1, int(n * (1 - test_frac))), n - 1)
    return dfx.iloc[:cut].copy(), dfx.iloc[cut:].copy()


def train(max_iter: int = DEFAULT_MAX_ITER) -> Tuple[float, float, Dict]:
    """
    Trains the model (gradient boosting over max_iter rounds), evaluates on a
    time split, and saves the bundle.

    Returns:
        (accuracy, logloss, meta_dict)
    """
    feats_fp = DATA_PROC / "features.parquet"
    if not feats_fp.exists():
        raise FileNotFoundError(f"{feats_fp} not found. Run feature builder first.")

    df = pd.read_parquet(feats_fp)

    y_series = _ensure_labels(df)
    if y_series.nunique() < 2:
        raise ValueError("Training data must contain at least two outcome classes.")

    feature_cols = _select_features(df)

    train_df, test_df = _time_split(df, test_frac=0.2)
    y_train = _ensure_labels(train_df)
    y_test = _ensure_labels(test_df)

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    pipe = _build_pipeline(max_iter=max_iter)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    try:
        ll = float(log_loss(y_test, y_proba))
    except ValueError:
        ll = float(log_loss(y_test, y_proba, labels=TARGET_LABELS))

    feature_means = X_train.mean(numeric_only=True).to_dict()

    meta = {
        "accuracy_time_split": acc,
        "logloss_time_split": ll,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features": feature_cols,
        "model": f"SimpleImputer(median) -> StandardScaler -> HistGradientBoosting(max_iter={max_iter})",
        "class_labels": TARGET_LABELS,
        "train_date_range": [
            str(train_df["Date"].min().date()),
            str(train_df["Date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["Date"].min().date()),
            str(test_df["Date"].max().date()),
        ],
    }

    bundle = {
        "pipeline": pipe,
        "feature_order": feature_cols,
        "feature_means": feature_means,
        "meta": meta,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"Time-split Accuracy: {acc:.3f} | LogLoss: {ll:.3f} | Features: {len(feature_cols)}")
    return acc, ll, meta


if __name__ == "__main__":
    train()
