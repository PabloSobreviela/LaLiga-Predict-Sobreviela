import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, List, Any

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss, precision_score
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.class_weight import compute_sample_weight

from src.config import CONFIG


class _HGBMultiSeedEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble of HistGradientBoostingClassifier with different random seeds."""

    def __init__(self, n_models: int = 5, seeds: List[int] = None, **hgb_params):
        self.n_models = n_models
        self.seeds = seeds or list(range(42, 42 + n_models))
        self.hgb_params = {k: v for k, v in hgb_params.items() if k != "random_state"}
        self.models_: List[HistGradientBoostingClassifier] = []

    def fit(self, X, y, sample_weight=None):
        check_classification_targets(y)
        self.models_ = []
        for s in self.seeds[: self.n_models]:
            m = HistGradientBoostingClassifier(**self.hgb_params, random_state=s)
            if sample_weight is not None:
                m.fit(X, y, sample_weight=sample_weight)
            else:
                m.fit(X, y)
            self.models_.append(m)
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X) for m in self.models_]).mean(axis=0)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

DATA_PROC = Path(CONFIG["processed_data_path"])
MODEL_PATH = Path(CONFIG["model_path"])
TARGET_LABELS = [0, 1, 2]

DEFAULT_TEST_FRAC = 0.17  # 83% train (best so far: 0.639)
_MODEL_CFG = CONFIG.get("model", {})
DEFAULT_MAX_ITER = _MODEL_CFG.get("max_iter", 700)


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


def _select_features(df: pd.DataFrame, max_nan_frac: float = 0.5) -> List[str]:
    """
    Choose model features from features.parquet.
    Exclude identifiers, labels, and columns with >max_nan_frac missing.
    """
    exclude = {"FTR", "FTHG", "FTAG", "HomeTeam", "AwayTeam", "Date", "Season"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    nan_frac = df[feats].isna().mean()
    feats = [c for c in feats if nan_frac.get(c, 0) <= max_nan_frac]
    if not feats:
        raise ValueError("No numeric feature columns found after exclusion.")
    return feats


def _build_pipeline(max_iter: int = DEFAULT_MAX_ITER, **clf_overrides) -> Pipeline:
    """
    Impute NaNs (median) -> scale -> classifier.
    classifier can be: hgb, xgb, or ensemble (HGB + XGB voting).
    """
    clf_type = _MODEL_CFG.get("classifier", "hgb")
    hgb_params = {
        "max_iter": max_iter,
        "max_depth": _MODEL_CFG.get("max_depth", 6),
        "learning_rate": _MODEL_CFG.get("learning_rate", 0.07),
        "min_samples_leaf": _MODEL_CFG.get("min_samples_leaf", 18),
        "l2_regularization": _MODEL_CFG.get("l2_regularization", 0.09),
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": 0.1,
        "random_state": 42,
    }
    if clf_type == "hgb_multi":
        clf = _HGBMultiSeedEnsemble(
            n_models=10,
            seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
            **hgb_params,
        )
    elif clf_type == "ensemble":
        try:
            import xgboost as xgb
        except ImportError:
            clf_type = "hgb"
        if clf_type == "ensemble":
            xgb_params = {
                "n_estimators": _MODEL_CFG.get("n_estimators", 400),
                "max_depth": _MODEL_CFG.get("max_depth", 6),
                "learning_rate": _MODEL_CFG.get("learning_rate", 0.07),
                "colsample_bytree": _MODEL_CFG.get("colsample_bytree", 0.8),
                "subsample": _MODEL_CFG.get("subsample", 0.9),
                "random_state": 43,
                "objective": "multi:softprob",
                "num_class": 3,
            }
            clf = VotingClassifier(
                estimators=[
                    ("hgb", HistGradientBoostingClassifier(**hgb_params)),
                    ("xgb", xgb.XGBClassifier(**xgb_params)),
                ],
                voting="soft",
                weights=[1.0, 1.0],
            )
        else:
            clf = HistGradientBoostingClassifier(**hgb_params)
    elif clf_type == "xgb":
        try:
            import xgboost as xgb
        except ImportError:
            clf_type = "hgb"
        if clf_type == "xgb":
            xgb_params = {
                "n_estimators": _MODEL_CFG.get("n_estimators", 400),
                "max_depth": _MODEL_CFG.get("max_depth", 6),
                "learning_rate": _MODEL_CFG.get("learning_rate", 0.07),
                "colsample_bytree": _MODEL_CFG.get("colsample_bytree", 0.8),
                "subsample": _MODEL_CFG.get("subsample", 0.9),
                "random_state": 42,
                "objective": "multi:softprob",
                "num_class": 3,
            }
            clf = xgb.XGBClassifier(**xgb_params)
        else:
            clf = HistGradientBoostingClassifier(**hgb_params)
    else:
        clf = HistGradientBoostingClassifier(**hgb_params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", clf),
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


def train(max_iter: int = DEFAULT_MAX_ITER, test_frac: float = DEFAULT_TEST_FRAC) -> Tuple[float, float, Dict]:
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

    train_df, test_df = _time_split(df, test_frac=test_frac)
    y_train = _ensure_labels(train_df)
    y_test = _ensure_labels(test_df)

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Sample weights to counteract class imbalance (Home ~45%, Draw ~26%, Away ~29%)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    pipe = _build_pipeline(max_iter=max_iter)
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    try:
        ll = float(log_loss(y_test, y_proba))
    except ValueError:
        ll = float(log_loss(y_test, y_proba, labels=TARGET_LABELS))

    feature_means = X_train.mean(numeric_only=True).to_dict()

    meta = {
        "accuracy_time_split": acc,
        "precision_time_split": prec,
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

    print(f"Time-split Accuracy: {acc:.3f} | Precision: {prec:.3f} | LogLoss: {ll:.3f} | Features: {len(feature_cols)}")
    return acc, ll, meta


if __name__ == "__main__":
    train()
