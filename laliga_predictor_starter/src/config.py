CONFIG = {
    "league": "SP1",
    # Add more seasons so Elo+form actually separate big teams like Barcelona
    "seasons": ["1314", "1415", "1516", "1617", "1819", "1920", "2021", "2122", "2223", "2324", "2425"],
    "raw_data_path": "data/raw",
    "processed_data_path": "data/processed",
    "model_path": "laliga_predictor_starter/models/model.joblib",

    # Elo settings (slightly stronger + real home edge)
    "elo": {
        "base": 1500,
        "k": 30,        # was 20 — a little more responsive
        "home_adv": 90  # was 60 — home teams do better on average
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
