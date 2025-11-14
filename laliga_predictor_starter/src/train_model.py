# src/train_model.py
# Trains the LaLiga predictor model. If features are missing, it will
# build them automatically via src.features.build_features().

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, List

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
import src.features as F  # for auto-build fallback


def _feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Take normalized match_features.parquet and return:
      X (numeric features only), y (0/1/2 for H/D/A), feature_order (column order).
    """
    # Columns to drop (ids/labels)
    drop_cols = {"Date", "home_team", "away_team", "FTR"}
    # Keep numeric feature columns only
    feature_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feature_cols].copy()

    # Target from FTR
    y = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    if y.isna().any():
        # If anything slipped through, drop those rows
        keep = ~y.isna()
        X = X.loc[keep]
        y = y.loc[keep]

    return X, y.values.astype(int), feature_cols


def _pipeline_for(X: pd.DataFrame) -> Pipeline:
    """
    Build a numeric pipeline robust to NaNs:
      - median impute
      - standardize
      - multinomial logistic regression
    """
    num_features = list(X.columns)

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # dense numeric
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", numeric, num_features)],
        remainder="drop",
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=400,
        n_jobs=None,  # use default
        solver="lbfgs",
        C=1.0,
        verbose=0,
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf),
        ]
    )
    return pipe


def train() -> Tuple[float, float, Dict[str, Any]]:
    """
    Train the model from processed features.
    Returns: (accuracy, log_loss, meta)
    Saves bundle to CONFIG["model_path"] with keys:
        pipeline, feature_order, feature_means, meta
    """
    proc_dir = Path(CONFIG["processed_data_path"])
    feats_path = proc_dir / "match_features.parquet"

    # --- Auto-build if missing ---
    if not feats_path.exists():
        # Attempt to build using any CSVs we can find (the app usually downloads first)
        try:
            F.build_features(seasons=None)
        except Exception as e:
            raise FileNotFoundError(f"{feats_path} not found and auto-build failed: {e}")

    df = pd.read_parquet(feats_path)

    # Split feature/target
    X, y, feature_order = _feature_target_split(df)
    if X.empty:
        raise ValueError("Feature table has no usable rows/columns after filtering.")

    # Train/validation split (stratified to keep H/D/A proportions)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build/train pipeline
    pipe = _pipeline_for(X_train)
    pipe.fit(X_train, y_train)

    # Evaluate
    yhat = pipe.predict(X_val)
    yproba = pipe.predict_proba(X_val)

    acc = float(accuracy_score(y_val, yhat))
    try:
        ll = float(log_loss(y_val, yproba, labels=[0, 1, 2]))
    except Exception:
        # In case of a degenerate class missing, still compute logloss on present labels
        present = sorted(np.unique(y_val).tolist())
        ll = float(log_loss(y_val, yproba[:, present], labels=present))

    # Feature means (useful default row when feature is missing at predict time)
    feature_means = X_train.mean(numeric_only=True).to_dict()

    # Persist bundle
    out_path = Path(CONFIG["model_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(
        trainer="multinomial_logistic",
        acc=acc,
        logloss=ll,
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
        version="1.0.0",
    )

    bundle = dict(
        pipeline=pipe,
        feature_order=feature_order,
        feature_means=feature_means,
        meta=meta,
    )
    joblib.dump(bundle, out_path)

    return acc, ll, meta
