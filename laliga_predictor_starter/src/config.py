from pathlib import Path

# Resolve paths relative to package root (works on Streamlit Cloud regardless of cwd)
_PKG_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "league": "SP1",
    "seasons": ["1314", "1415", "1516", "1617", "1819", "1920", "2021", "2122", "2223", "2324", "2425"],
    "raw_data_path": str(_PKG_ROOT / "data" / "raw"),
    "processed_data_path": str(_PKG_ROOT / "data" / "processed"),
    "model_path": str(_PKG_ROOT / "models" / "model.joblib"),

    # Elo settings (moderate home edge to avoid over-weighting home advantage)
    "elo": {
        "base": 1500,
        "k": 30,
        "home_adv": 65,
    },

    # Rolling window for "recent form"
    "rolling_windows": [10],  # was 5 — smoother, less noisy than last-5

    # Classifier: "hgb", "xgb", "ensemble" (HGB+XGB), or "hgb_multi" (5 seeds averaged)
    "model": {
        "classifier": "hgb_multi",
        "max_iter": 1200,
        "max_depth": 8,
        "learning_rate": 0.04,
        "min_samples_leaf": 10,
        "l2_regularization": 0.09,
        # XGBoost-specific
        "n_estimators": 400,
        "colsample_bytree": 0.8,
        "subsample": 0.9,
    },
}
