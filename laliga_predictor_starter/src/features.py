# src/features.py
"""
Builds processed datasets for the LaLiga predictor.

Outputs (under CONFIG['processed_data_path']):
- matches.parquet            : match-level raw (wide) table
- long_teams.parquet         : per-team, per-match long table with rolling metrics
- elo_history.parquet        : per-team Elo by match date
- features_train.parquet     : training rows with engineered features + label
- team_state.parquet         : latest per-team snapshot used by the app's UI
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from src.config import CONFIG

DATA_RAW = Path(CONFIG["raw_data_path"])
DATA_PROC = Path(CONFIG["processed_data_path"])

# ------------------------------- IO helpers -------------------------------

def _raw_path_for(season_code: str) -> Path:
    # football-data.co.uk dump is expected to have been saved as e.g. "laliga_2324.csv"
    return DATA_RAW / f"{CONFIG['league']}_{season_code}.csv"

def _read_raw_csv(path: Path) -> pd.DataFrame:
    # football-data columns vary a bit by year; we standardize right away
    df = pd.read_csv(path)
    # Try common column names
    col_map_candidates = [
        # (Home team, Away team, Home goals, Away goals, Result, Date, HST, AST)
        ("HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Date", "HST", "AST"),
        ("Home",     "Away",     "FTHG", "FTAG", "FTR", "Date", "HST", "AST"),
    ]
    use_map = None
    for hm, am, hg, ag, res, date, hst, ast in col_map_candidates:
        if hm in df.columns and am in df.columns and hg in df.columns and ag in df.columns and res in df.columns:
            use_map = (hm, am, hg, ag, res, date if date in df.columns else None,
                       hst if hst in df.columns else None,
                       ast if ast in df.columns else None)
            break
    if use_map is None:
        raise KeyError(f"Could not find expected football-data columns in {path.name}")

    hm, am, hg, ag, res, date, hst, ast = use_map
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date]) if date else pd.to_datetime(df.index, unit="D", origin="unix", errors="ignore"),
        "Home": df[hm].astype(str),
        "Away": df[am].astype(str),
        "FTHG": pd.to_numeric(df[hg], errors="coerce"),
        "FTAG": pd.to_numeric(df[ag], errors="coerce"),
        "FTR":  df[res].astype(str),
    })
    out["HST"] = pd.to_numeric(df[hst], errors="coerce") if hst else np.nan
    out["AST"] = pd.to_numeric(df[ast], errors="coerce") if ast else np.nan

    # Normalize FTR to H/D/A if some seasons use other tokens
    out["FTR"] = out["FTR"].map({"H":"H","D":"D","A":"A"}).fillna(
        np.where(out["FTHG"]>out["FTAG"], "H",
        np.where(out["FTHG"]<out["FTAG"], "A", "D"))
    )
    return out.sort_values("Date").reset_index(drop=True)

# ------------------------------- Elo utils --------------------------------

def _logistic_expectation(elo_a: float, elo_b: float, home_adv: float = 60.0) -> float:
    """Expected score for home side vs away."""
    return 1.0 / (1.0 + 10 ** (-( (elo_a + home_adv) - elo_b) / 400.0))

def _update_elo_once(elo_h: float, elo_a: float, result: str, k: float = 22.0, home_adv: float = 60.0) -> Tuple[float, float]:
    # result: "H"/"D"/"A"
    exp_home = _logistic_expectation(elo_h, elo_a, home_adv=home_adv)
    score_home = 1.0 if result == "H" else 0.5 if result == "D" else 0.0
    delta = k * (score_home - exp_home)
    return elo_h + delta, elo_a - delta

def compute_elo_history(matches: pd.DataFrame,
                        base_rating: float = 1500.0,
                        k: float = 22.0,
                        home_adv: float = 60.0) -> pd.DataFrame:
    """
    Returns long table with per-team Elo over time:
    columns: [Date, Team, elo]
    """
    elos: Dict[str, float] = {}
    rows = []
    for _, r in matches.iterrows():
        h, a, res, date = r["Home"], r["Away"], r["FTR"], r["Date"]
        eh = elos.get(h, base_rating)
        ea = elos.get(a, base_rating)
        new_h, new_a = _update_elo_once(eh, ea, res, k=k, home_adv=home_adv)
        elos[h], elos[a] = new_h, new_a
        rows.append((date, h, new_h))
        rows.append((date, a, new_a))
    elo_df = pd.DataFrame(rows, columns=["Date","Team","elo"]).sort_values(["Team","Date"]).reset_index(drop=True)
    return elo_df

# ------------------------- Rolling features (long) -------------------------

def _build_long_team_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Long table with one row per team per match and rolling 'last N' features.
    """
    # Build long rows
    home_rows = matches[["Date","Home","Away","FTHG","FTAG","HST","AST"]].copy()
    home_rows.rename(columns={
        "Home":"Team","Away":"Opp",
        "FTHG":"goals_for","FTAG":"goals_against",
        "HST":"sot_for","AST":"sot_against"
    }, inplace=True)
    home_rows["is_home"] = True

    away_rows = matches[["Date","Home","Away","FTHG","FTAG","HST","AST"]].copy()
    away_rows.rename(columns={
        "Away":"Team","Home":"Opp",
        "FTAG":"goals_for","FTHG":"goals_against",
        "AST":"sot_for","HST":"sot_against"
    }, inplace=True)
    away_rows["is_home"] = False

    long = pd.concat([home_rows[["Date","Team","Opp","is_home","goals_for","goals_against","sot_for","sot_against"]],
                      away_rows[["Date","Team","Opp","is_home","goals_for","goals_against","sot_for","sot_against"]]],
                     ignore_index=True).sort_values(["Team","Date"]).reset_index(drop=True)

    long["gd"] = long["goals_for"] - long["goals_against"]

    # rest days
    long["prev_date"] = long.groupby("Team")["Date"].shift(1)
    long["rest_days"] = (long["Date"] - long["prev_date"]).dt.days.astype("float")
    long["rest_days"] = long["rest_days"].fillna(long["rest_days"].median())

    # rolling windows
    for w in (5,):
        long[f"last{w}_gf"]  = long.groupby("Team")["goals_for"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        long[f"last{w}_ga"]  = long.groupby("Team")["goals_against"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        long[f"last{w}_gd"]  = long.groupby("Team")["gd"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        long[f"last{w}_sotf"] = long.groupby("Team")["sot_for"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        long[f"last{w}_sota"] = long.groupby("Team")["sot_against"].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    long = long.drop(columns=["prev_date"])
    return long

# ----------------------- Match-level features (wide) -----------------------

def assemble_match_features(matches: pd.DataFrame, long: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match row, join the 'pre-match' rolling stats for each team and the current Elo.
    We use the previous row's rolling value via shift on long table.
    """
    # Get "state as of before the match" by shifting by 1 per team
    cols_keep = ["Date","Team","rest_days","last5_gf","last5_ga","last5_gd","last5_sotf","last5_sota"]
    long_before = long.sort_values(["Team","Date"]).copy()
    for c in cols_keep[2:]:
        long_before[c] = long_before.groupby("Team")[c].shift(1)
    long_before["rest_days"] = long_before["rest_days"].fillna(long_before["rest_days"].median())
    long_before = long_before[cols_keep]

    # Prepare Elo "as-of previous"
    elo_before = elo_df.sort_values(["Team","Date"]).copy()
    elo_before["elo"] = elo_before.groupby("Team")["elo"].shift(1)  # pre-match elo
    elo_before = elo_before.dropna(subset=["elo"])

    # Home merge
    home_state = long_before.rename(columns={
        "Team":"Home",
        "rest_days":"rest_days_home",
        "last5_gf":"home_last_gf",
        "last5_ga":"home_last_ga",
        "last5_gd":"home_last_gd",
        "last5_sotf":"home_last_sot_for",
        "last5_sota":"home_last_sot_against",
    })
    away_state = long_before.rename(columns={
        "Team":"Away",
        "rest_days":"rest_days_away",
        "last5_gf":"away_last_gf",
        "last5_ga":"away_last_ga",
        "last5_gd":"away_last_gd",
        "last5_sotf":"away_last_sot_for",
        "last5_sota":"away_last_sot_against",
    })

    feats = matches.merge(home_state, on=["Date","Home"], how="left") \
                   .merge(away_state, on=["Date","Away"], how="left")

    # Elo merges
    elo_home = elo_before.rename(columns={"Team":"Home","elo":"elo_home"})
    elo_away = elo_before.rename(columns={"Team":"Away","elo":"elo_away"})
    feats = feats.merge(elo_home[["Date","Home","elo_home"]], on=["Date","Home"], how="left") \
                 .merge(elo_away[["Date","Away","elo_away"]], on=["Date","Away"], how="left")

    # Derived
    feats["elo_diff"]                = feats["elo_home"] - feats["elo_away"]
    feats["form_gd_diff"]            = feats["home_last_gd"] - feats["away_last_gd"]
    feats["form_sot_for_diff"]       = feats["home_last_sot_for"] - feats["away_last_sot_for"]
    feats["form_sot_against_diff"]   = feats["home_last_sot_against"] - feats["away_last_sot_against"]
    feats["rest_days_diff"]          = feats["rest_days_home"] - feats["rest_days_away"]

    # Label safety
    if "FTR" not in feats.columns or feats["FTR"].isna().all():
        feats["FTR"] = np.where(feats["FTHG"] > feats["FTAG"], "H",
                        np.where(feats["FTHG"] < feats["FTAG"], "A", "D"))

    return feats

# ------------------------------- Public API -------------------------------

def build_features(seasons: Optional[List[str]] = None) -> None:
    """
    Main entry: read available raw CSVs for given seasons, build all processed artifacts.
    """
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # 1) read raw
    if seasons is None:
        seasons = []
        for p in DATA_RAW.glob(f"{CONFIG['league']}_*.csv"):
            try:
                seasons.append(p.stem.split("_")[-1])
            except Exception:
                pass
        seasons = sorted(seasons)
    frames = []
    for s in seasons:
        p = _raw_path_for(s)
        if p.exists():
            frames.append(_read_raw_csv(p))
    if not frames:
        raise FileNotFoundError("No raw CSVs found for requested seasons in data/raw")

    matches = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    # Save raw-wide
    matches.to_parquet(DATA_PROC / "matches.parquet", index=False)

    # 2) long + elo
    long = _build_long_team_table(matches)
    long.to_parquet(DATA_PROC / "long_teams.parquet", index=False)

    elo_df = compute_elo_history(matches)
    elo_df.to_parquet(DATA_PROC / "elo_history.parquet", index=False)

    # 3) match-level engineered features
    feats = assemble_match_features(matches, long, elo_df)

    # Minimal NA handling to keep training happy; model pipeline will impute again
    feature_cols = [
        "elo_home","elo_away","elo_diff",
        "home_last_gf","away_last_gf",
        "home_last_ga","away_last_ga",
        "home_last_gd","away_last_gd",
        "home_last_sot_for","away_last_sot_for",
        "home_last_sot_against","away_last_sot_against",
        "rest_days_home","rest_days_away",
        "form_gd_diff","form_sot_for_diff","form_sot_against_diff","rest_days_diff",
    ]
    for c in feature_cols:
        if c not in feats.columns:
            feats[c] = np.nan
    feats[feature_cols] = feats[feature_cols].astype(float)

    feats.to_parquet(DATA_PROC / "features_train.parquet", index=False)

    # 4) team_state snapshot (latest row per team, after last match)
    last_date = matches["Date"].max()
    # last known rolling values (no shift) for snapshot
    snap_cols = ["Team","Date","rest_days","last5_gf","last5_ga","last5_gd","last5_sotf","last5_sota"]
    latest_long = long.sort_values("Date").groupby("Team", as_index=False).last()[snap_cols]

    latest_long.rename(columns={
        "rest_days":"team_rest_days",
        "last5_gf":"team_last_gf",
        "last5_ga":"team_last_ga",
        "last5_gd":"team_last_gd",
        "last5_sotf":"team_last_sot_for",
        "last5_sota":"team_last_sot_against",
    }, inplace=True)

    latest_elo = elo_df.sort_values("Date").groupby("Team", as_index=False).last()[["Team","elo"]]

    team_state = latest_long.merge(latest_elo, on="Team", how="left")
    # If any NaNs remain (brand new teams etc.), fill with column medians
    for c in ["team_rest_days","team_last_gf","team_last_ga","team_last_gd","team_last_sot_for","team_last_sot_against","elo"]:
        if c in team_state.columns:
            team_state[c] = pd.to_numeric(team_state[c], errors="coerce")
            team_state[c] = team_state[c].fillna(team_state[c].median())

    team_state.to_parquet(DATA_PROC / "team_state.parquet", index=False)

def run_build_features(seasons: Optional[List[str]] = None) -> None:
    """Alias kept for backward-compat with some app versions."""
    return build_features(seasons=seasons)

# ------------------------------ CLI support -------------------------------

def main():
    # If run as a script: build for whatever raw CSVs are present.
    build_features()

if __name__ == "__main__":
    main()
