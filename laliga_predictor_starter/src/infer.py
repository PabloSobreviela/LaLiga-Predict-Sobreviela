"""
Predict H/D/A for a chosen matchup using the latest per-team state + optional odds.

Usage:
  python -m src.infer --home "Barcelona" --away "Sevilla"
  python -m src.infer --home "Barcelona" --away "Sevilla" --oddsH 1.9 --oddsD 3.4 --oddsA 4.2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from src.config import CONFIG

def normalize_probs_from_odds(oh: float, od: float, oa: float):
    raw = np.array([1/oh, 1/od, 1/oa], dtype=float)
    raw /= raw.sum()
    return raw.tolist()  # [pH, pD, pA]

def load_bundle():
    bundle = joblib.load(CONFIG["model_path"])
    return bundle["pipeline"], bundle["feature_order"], bundle.get("feature_means", {})


def load_team_state():
    fp = Path(CONFIG["processed_data_path"]) / "team_state.parquet"
    if not fp.exists():
        raise FileNotFoundError("team_state.parquet not found. Run: python -m src.features")
    # Index by team for easy lookup
    return pd.read_parquet(fp).set_index("Team")

def vector_for_match(home: str, away: str, feature_order, ts, feature_means: dict, odds=None):
    import pandas as pd

    # Start from training means (neutral baseline)
    row = pd.Series({c: float(feature_means.get(c, 0.0)) for c in feature_order}, dtype=float)

    # Elo
    if home not in ts.index or away not in ts.index:
        raise ValueError("Team not found in team_state. Check exact team names from CSV.")
    elo_home = float(ts.loc[home, "elo"])
    elo_away = float(ts.loc[away, "elo"])
    if "elo_home" in row.index: row["elo_home"] = elo_home
    if "elo_away" in row.index: row["elo_away"] = elo_away
    if "elo_diff" in row.index: row["elo_diff"] = elo_home - elo_away

    # Rolling form → home/away slots
    mapping = {
        "team_last5_gf": ("home_last5_gf", "away_last5_gf"),
        "team_last5_ga": ("home_last5_ga", "away_last5_ga"),
        "team_last5_gd": ("home_last5_gd", "away_last5_gd"),
        "team_last5_sot_for": ("home_last5_sot_for", "away_last5_sot_for"),
        "team_last5_sot_against": ("home_last5_sot_against", "away_last5_sot_against"),
        "team_rest_days": ("rest_days_home", "rest_days_away"),
    }
    for tcol, (hcol, acol) in mapping.items():
        if hcol in row.index: row[hcol] = float(ts.loc[home, tcol])
        if acol in row.index: row[acol] = float(ts.loc[away, tcol])

    # If engineered diffs exist, compute them now
    if "form_gd_diff" in row.index:
        row["form_gd_diff"] = row.get("home_last5_gd", 0.0) - row.get("away_last5_gd", 0.0)
    if "form_sot_for_diff" in row.index:
        row["form_sot_for_diff"] = row.get("home_last5_sot_for", 0.0) - row.get("away_last5_sot_for", 0.0)
    if "form_sot_against_diff" in row.index:
        row["form_sot_against_diff"] = row.get("home_last5_sot_against", 0.0) - row.get("away_last5_sot_against", 0.0)
    if "rest_days_diff" in row.index:
        row["rest_days_diff"] = row.get("rest_days_home", 0.0) - row.get("rest_days_away", 0.0)

    return pd.DataFrame([row.values], columns=feature_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, type=str)
    parser.add_argument("--away", required=True, type=str)
    parser.add_argument("--oddsH", type=float, default=None)
    parser.add_argument("--oddsD", type=float, default=None)
    parser.add_argument("--oddsA", type=float, default=None)
    args = parser.parse_args()

    odds = None
    if args.oddsH and args.oddsD and args.oddsA:
        odds = (args.oddsH, args.oddsD, args.oddsA)
    pipe, feature_order, feature_means = load_bundle()
    ts = load_team_state()
    X = vector_for_match(args.home, args.away, feature_order, ts, feature_means, odds)

    X_arr = X.values  # convert to numpy array to match training and silence sklearn warning
    probs = pipe.predict_proba(X_arr)[0].tolist()
    pred = int(np.argmax(probs))
    mapping_inv = {0:"H", 1:"D", 2:"A"}

    print({
        "home": args.home,
        "away": args.away,
        "prediction": mapping_inv[pred],
        "probs": {"H": round(probs[0],3), "D": round(probs[1],3), "A": round(probs[2],3)}
    })

if __name__ == "__main__":
    main()
