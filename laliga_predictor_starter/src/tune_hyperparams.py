# src/tune_hyperparams.py
# Hyperparameter tuning for the La Liga predictor using time-series CV.
# Run: python -m src.tune_hyperparams
# Best params are printed and applied to train_model.
#
# Use --refine for a focused grid around current best (faster, often better).

from pathlib import Path
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer

from src.train_model import (
    DATA_PROC,
    _ensure_labels,
    _select_features,
    _time_split,
)

RANDOM_STATE = 42
TEST_FRAC = 0.2
N_ITER = 80  # random samples (for RandomizedSearch)
CV_SPLITS = 5

# Baseline params (current best ~0.622) — refine around these
BASELINE_PARAMS = {
    "max_iter": 700,
    "max_depth": 6,
    "learning_rate": 0.07,
    "min_samples_leaf": 18,
    "l2_regularization": 0.09,
}


def _build_tunable_pipeline(**clf_params) -> Pipeline:
    base_params = {
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": 0.1,
        "random_state": RANDOM_STATE,
    }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", HistGradientBoostingClassifier(**{**base_params, **clf_params})),
    ])


def run_tuning() -> Dict[str, Any]:
    feats_fp = DATA_PROC / "features.parquet"
    if not feats_fp.exists():
        raise FileNotFoundError(f"{feats_fp} not found. Run: python -m src.features")

    df = pd.read_parquet(feats_fp)
    y = _ensure_labels(df)
    feature_cols = _select_features(df)
    X = df[feature_cols].copy()

    train_df, test_df = _time_split(df, test_frac=TEST_FRAC)
    X_train = train_df[feature_cols].copy()
    y_train = _ensure_labels(train_df)
    X_test = test_df[feature_cols].copy()
    y_test = _ensure_labels(test_df)

    use_refine = "--refine" in sys.argv
    if use_refine:
        param_grid = {
            "clf__max_iter": [600, 700, 900],
            "clf__max_depth": [5, 6, 7],
            "clf__learning_rate": [0.05, 0.07, 0.09],
            "clf__min_samples_leaf": [14, 18, 24],
            "clf__l2_regularization": [0.06, 0.09, 0.15],
        }
        search = GridSearchCV(
            _build_tunable_pipeline(),
            param_grid=param_grid,
            scoring=make_scorer(accuracy_score),
            cv=TimeSeriesSplit(n_splits=CV_SPLITS),
            verbose=1,
            n_jobs=-1,
        )
    else:
        param_dist = {
            "clf__max_iter": [400, 600, 800, 1000, 1200, 1500],
            "clf__max_depth": [4, 5, 6, 7, 8, 9],
            "clf__learning_rate": [0.03, 0.05, 0.07, 0.09, 0.1],
            "clf__min_samples_leaf": [8, 12, 16, 20, 24, 30],
            "clf__l2_regularization": [0.03, 0.06, 0.09, 0.12, 0.18],
        }
        search = RandomizedSearchCV(
            _build_tunable_pipeline(),
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring=make_scorer(accuracy_score),
            cv=TimeSeriesSplit(n_splits=CV_SPLITS),
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_acc_cv = search.best_score_

    # Evaluate on held-out test (final 20%)
    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))

    clf_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
    n_iter = getattr(search, "n_iter", len(search.cv_results_.get("params", [])))
    return {
        "best_params": clf_params,
        "cv_accuracy": best_acc_cv,
        "test_accuracy": test_acc,
        "n_iter": n_iter,
    }


def main():
    use_refine = "--refine" in sys.argv
    mode = "refine (grid)" if use_refine else f"random (n_iter={N_ITER})"
    print(f"Tuning HistGradientBoosting hyperparameters [{mode}]...")
    print(f"TimeSeriesSplit CV folds: {CV_SPLITS}\n")
    result = run_tuning()
    bp = result["best_params"]
    print("\n" + "=" * 60)
    print("Best hyperparameters:")
    for k, v in sorted(bp.items()):
        print(f"  {k}: {v}")
    print("=" * 60)
    print(f"CV Accuracy (mean): {result['cv_accuracy']:.4f}")
    print(f"Test Accuracy (held-out 20%): {result['test_accuracy']:.4f}")
    print("\nTo apply: copy the dict above into CONFIG['model'] in src/config.py")


if __name__ == "__main__":
    main()
