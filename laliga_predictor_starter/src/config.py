CONFIG = {
    "league": "SP1",
    # Add more seasons so Elo+form actually separate big teams like Barcelona
    "seasons": ["2122", "2223", "2324", "2425"],
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
}
