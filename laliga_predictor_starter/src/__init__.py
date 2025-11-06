CONFIG = {
    "league": "SP1",
    "seasons": ["2324", "2425"],  
    "raw_data_path": "data/raw",
    "processed_data_path": "data/processed",
    "model_path": "models/model.joblib",

    # Elo settings (simple rating system)
    "elo": {
        "base": 1500,
        "k": 20,
        "home_adv": 60
    },

    # Rolling windows for "recent form"
    "rolling_windows": [5],  # start with last-5
}
