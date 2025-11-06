"""
Evaluate on a time-based test split (same as training split).
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from src.config import CONFIG

def load_features() -> pd.DataFrame:
    return pd.read_parquet(Path(CONFIG["processed_data_path"]) / "features.parquet")

def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("Date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def main():
    model_path = Path(CONFIG["model_path"])
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train it with train_model.py")
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    feature_order = bundle["feature_order"]

    df = load_features()
    _, test_df = time_split(df, test_frac=0.2)
    X_test = test_df[feature_order].values
    y_test = test_df["y"].values

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)

    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"LogLoss: {log_loss(y_test, probs):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nReport:\n", classification_report(y_test, preds, digits=3))

if __name__ == "__main__":
    main()
