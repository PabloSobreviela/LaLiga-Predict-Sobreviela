# src/ensemble.py — HGB multi-seed ensemble (lives here for stable pickle path)
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np


class HGBMultiSeedEnsemble(BaseEstimator, ClassifierMixin):
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
