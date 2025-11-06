"""
Quick leakage diagnostic.
Reports the top single features on the TEST split and their standalone accuracy.
If any single feature gets absurdly high accuracy, you likely have leakage.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.config import CONFIG

def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("Date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def main():
    fp = Path(CONFIG["processed_data_path"]) / "features.parquet"
    df = pd.read_parquet(fp)
    train_df, test_df = time_split(df, 0.2)

    # restrict to numeric, drop target and any obvious odds if present
    forbid = {"y","pH","pD","pA","prob_home_minus_away","AvgH","AvgD","AvgA","B365H","B365D","B365A","PSH","PSD","PSA"}
    feats = [c for c in test_df.select_dtypes(include="number").columns if c not in forbid]
    y = test_df["y"].values

    results = []
    for c in feats:
        x = test_df[c].values
        # super simple 1-feature rule: predict class by comparing to train means
        means = train_df.groupby("y")[c].mean().to_dict()
        # pick the class whose mean is closest to the test value
        preds = np.array([min(means.keys(), key=lambda k: abs(v - means[k])) for v in x])
        acc = accuracy_score(y, preds)
        results.append((c, acc))
    results.sort(key=lambda t: t[1], reverse=True)

    print("Top features by standalone test accuracy:")
    for name, acc in results[:10]:
        print(f"{name:30s}  acc={acc:.3f}")

if __name__ == "__main__":
    main()
