# src/features.py
# Build training features and a per-team snapshot for inference.
# Writes:
#   - data/processed/features.parquet (row = match)
#   - data/processed/team_state.parquet (row = team, latest)
#
# Safe to call from Streamlit or CLI:
#   python -m src.features

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from src.config import CONFIG

# --- Safety helpers ---
def ensure_cols(df, cols, fill_val=np.nan):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_val
    return df

def alias_if_missing(df, src, dst):
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]
    return df


RAW_DIR = Path(CONFIG["raw_data_path"])
PROC_DIR = Path(CONFIG["processed_data_path"])
LEAGUE = CONFIG.get("league", "SP1")

# ----------------------------
# Utilities
# ----------------------------
def _season_codes_default() -> List[str]:
    # Extend if you want; app passes explicit selections anyway.
    return ["2122", "2223", "2324", "2425", "2526"]

def _load_one_csv(season: str) -> Optional[pd.DataFrame]:
    fp = RAW_DIR / f"{LEAGUE}_{season}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    # Normalize column names we use
    # Football-data columns we rely on:
    #   Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HST, AST
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    # Normalize common variants (rare)
    if "hometeam" not in cols and "home" in cols: rename_map[cols["home"]] = "HomeTeam"
    if "awayteam" not in cols and "away" in cols: rename_map[cols["away"]] = "AwayTeam"
    if "hst" not in cols and "hson" in cols: rename_map[cols["hson"]] = "HST"
    if "ast" not in cols and "ason" in cols: rename_map[cols["ason"]] = "AST"

    # Apply renames if any
    if rename_map:
        df = df.rename(columns=rename_map)

    # Parse Date (football-data is dd/mm/yy or dd/mm/yyyy)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    else:
        # If missing, create a monotonic date (very unlikely)
        df["Date"] = pd.date_range("2000-01-01", periods=len(df), freq="D")

    # Keep only rows with valid teams
    need_cols = ["HomeTeam", "AwayTeam"]
    for c in need_cols:
        if c not in df.columns:
            # If catastrophically missing, bail on this season
            return None
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()

    # Ensure SOT exists (fill with NaN if absent -> becomes 0 later in rolling means)
    for c in ["HST", "AST"]:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure goals/label exist (training can derive FTR if missing)
    for c in ["FTHG", "FTAG", "FTR"]:
        if c not in df.columns:
            df[c] = np.nan

    df["Season"] = season
    return df

def load_raw(seasons: Optional[List[str]] = None) -> pd.DataFrame:
    if seasons is None:
        seasons = _season_codes_default()
    frames = []
    for s in seasons:
        one = _load_one_csv(s)
        if one is not None and len(one):
            frames.append(one)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("Date").reset_index(drop=True)
    return out

# ----------------------------
# Simple Elo implementation
# ----------------------------
def _elo_expected(rA: float, rB: float, hfa: float = 60.0, is_homeA: bool = True) -> float:
    # Add home-field advantage to home team rating
    rA_eff = rA + (hfa if is_homeA else 0.0)
    rB_eff = rB + (0.0 if is_homeA else hfa)
    return 1.0 / (1.0 + 10 ** ((rB_eff - rA_eff) / 400.0))

def _elo_update(rA: float, rB: float, scoreA: float, K: float, hfa: float, is_homeA: bool) -> Tuple[float, float]:
    expA = _elo_expected(rA, rB, hfa=hfa, is_homeA=is_homeA)
    rA_new = rA + K * (scoreA - expA)
    # scoreB = 1 - scoreA
    expB = _elo_expected(rB, rA, hfa=hfa, is_homeA=(not is_homeA))
    rB_new = rB + K * ((1.0 - scoreA) - expB)
    return rA_new, rB_new

def compute_elo_timeseries(matches: pd.DataFrame,
                           base_rating: float = 1500.0,
                           K: float = 22.0,
                           hfa: float = 60.0) -> pd.DataFrame:
    """
    Input: matches sorted by Date. Expects columns: Date, HomeTeam, AwayTeam, FTR or (FTHG,FTAG)
    Returns long frame: columns=['Date','Team','elo']
    """
    if matches.empty:
        return pd.DataFrame(columns=["Date","Team","elo"])

    ratings: Dict[str, float] = {}
    rows: List[Tuple[pd.Timestamp, str, float]] = []

    def _result_to_score(ftr: str, fthg: float, ftag: float) -> float:
        # Return 1/0.5/0 for home team
        if isinstance(ftr, str) and ftr in {"H","D","A"}:
            return {"H":1.0, "D":0.5, "A":0.0}[ftr]
        # Fallback via goals
        if pd.notna(fthg) and pd.notna(ftag):
            if fthg > ftag: return 1.0
            if fthg < ftag: return 0.0
            return 0.5
        # Unknown, treat as draw-ish (minimal movement)
        return 0.5

    for _, r in matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        d = r["Date"]
        ftr = r.get("FTR", np.nan)
        fthg = r.get("FTHG", np.nan)
        ftag = r.get("FTAG", np.nan)

        rH = ratings.get(h, base_rating)
        rA = ratings.get(a, base_rating)

        score_home = _result_to_score(ftr, fthg, ftag)

        # Log pre-update ratings (snapshot before match)
        rows.append((d, h, rH))
        rows.append((d, a, rA))

        # Update
        rH_new, rA_new = _elo_update(rH, rA, score_home, K=K, hfa=hfa, is_homeA=True)
        ratings[h] = rH_new
        ratings[a] = rA_new

    elo_df = pd.DataFrame(rows, columns=["Date","Team","elo"])
    # Keep last seen per date; groupby last then cummax by date works, but we just keep as-is sorted
    return elo_df

# ----------------------------
# Long-format for rolling pre-match features
# ----------------------------
def make_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    # For each match, make two rows (home perspective & away perspective)
    # Keep only columns we need and coerce to numeric as needed.
    keep = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HST","AST","Season"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    tmp = df[keep].copy()

    home_part = pd.DataFrame({
        "Date": tmp["Date"],
        "Team": tmp["HomeTeam"],
        "Opponent": tmp["AwayTeam"],
        "is_home": True,
        "GF": pd.to_numeric(tmp["FTHG"], errors="coerce"),
        "GA": pd.to_numeric(tmp["FTAG"], errors="coerce"),
        "SOT_for": pd.to_numeric(tmp["HST"], errors="coerce"),
        "SOT_against": pd.to_numeric(tmp["AST"], errors="coerce"),
        "Season": tmp["Season"]
    })
    away_part = pd.DataFrame({
        "Date": tmp["Date"],
        "Team": tmp["AwayTeam"],
        "Opponent": tmp["HomeTeam"],
        "is_home": False,
        "GF": pd.to_numeric(tmp["FTAG"], errors="coerce"),
        "GA": pd.to_numeric(tmp["FTHG"], errors="coerce"),
        "SOT_for": pd.to_numeric(tmp["AST"], errors="coerce"),
        "SOT_against": pd.to_numeric(tmp["HST"], errors="coerce"),
        "Season": tmp["Season"]
    })
    long = pd.concat([home_part, away_part], ignore_index=True)
    long = long.sort_values(["Team","Date"]).reset_index(drop=True)
    return long

def add_pre_match_rollups(long: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # Compute rolling means *before* each match (shift by 1)
    long = long.copy()
    g = long.groupby("Team", group_keys=False)

    long["team_last_gf"]  = g["GF"].shift(1).rolling(window, min_periods=1).mean()
    long["team_last_ga"]  = g["GA"].shift(1).rolling(window, min_periods=1).mean()
    long["team_last_gd"]  = long["team_last_gf"] - long["team_last_ga"]
    long["team_last_sot_for"]     = g["SOT_for"].shift(1).rolling(window, min_periods=1).mean()
    long["team_last_sot_against"] = g["SOT_against"].shift(1).rolling(window, min_periods=1).mean()

    # Rest days since last match
    long["prev_date"] = g["Date"].shift(1)
    long["team_rest_days"] = (long["Date"] - long["prev_date"]).dt.days
    long["team_rest_days"] = long["team_rest_days"].fillna(long["team_rest_days"].median())
    long = long.drop(columns=["prev_date"])
    return long

# ----------------------------
# Feature assembly
# ----------------------------
def assemble_match_features(df: pd.DataFrame,
                            long: pd.DataFrame,
                            elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-match rolling stats + Elo.
    Produces:
      home_last_* / away_last_* , home_rest_days / away_rest_days,
      rest_days_home/rest_days_away (aliases),
      elo_home/elo_away/elo_diff,
      *_diff engineered features.
    """

    # --- columns we expect from the long table ---
    pre_cols = [
        "team_last_gf", "team_last_ga", "team_last_gd",
        "team_last_sot_for", "team_last_sot_against",
        "team_rest_days",
    ]
    # be robust if long is missing any expected columns
    long = ensure_cols(long, pre_cols, fill_val=np.nan)

    long_pre = long[["Team", "Date"] + pre_cols].copy()

    def _rename_for_side(prefix: str) -> dict:
        mapping = {}
        for c in pre_cols:
            base = c[5:] if c.startswith("team_") else c
            mapping[c] = f"{prefix}_{base}"  # e.g., home_last_gf, away_rest_days
        return mapping

    # --- join for home side ---
    home_join = long_pre.rename(columns=_rename_for_side("home")).rename(columns={"Team": "HomeTeam"})
    df = df.merge(home_join, on=["HomeTeam", "Date"], how="left")

    # --- join for away side ---
    away_join = long_pre.rename(columns=_rename_for_side("away")).rename(columns={"Team": "AwayTeam"})
    df = df.merge(away_join, on=["AwayTeam", "Date"], how="left")

    # --- Elo joins (pre-match Elo snapshot by (Team, Date)) ---
    # Be robust: ensure expected columns exist
    elo_df = ensure_cols(elo_df, ["Team", "Date", "elo"], fill_val=np.nan)

    elo_home = elo_df.rename(columns={"Team": "HomeTeam", "elo": "elo_home"})
    elo_away = elo_df.rename(columns={"Team": "AwayTeam", "elo": "elo_away"})
    df = df.merge(elo_home, on=["HomeTeam", "Date"], how="left")
    df = df.merge(elo_away, on=["AwayTeam", "Date"], how="left")
    df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # --- make sure rest-day aliases exist for the app ---
    df = alias_if_missing(df, "home_rest_days", "rest_days_home")
    df = alias_if_missing(df, "away_rest_days", "rest_days_away")
    df = ensure_cols(df, ["rest_days_home", "rest_days_away"], fill_val=np.nan)

    # --- engineered diffs (guard every input) ---
    df = ensure_cols(df, [
        "home_last_gd","away_last_gd",
        "home_last_sot_for","away_last_sot_for",
        "home_last_sot_against","away_last_sot_against",
    ], fill_val=0.0)

    df["form_gd_diff"]           = df["home_last_gd"] - df["away_last_gd"]
    df["form_sot_for_diff"]      = df["home_last_sot_for"] - df["away_last_sot_for"]
    df["form_sot_against_diff"]  = df["home_last_sot_against"] - df["away_last_sot_against"]
    df["rest_days_diff"]         = df["rest_days_home"] - df["rest_days_away"]

    return df

# ----------------------------
# Robust Elo snapshot (fix for 'Team' missing)
# ----------------------------
def _maybe_promote_index_to_team(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Team" in out.columns:
        return out
    # Index name is Team/Club?
    if out.index.name and out.index.name.lower() in {"team", "club"}:
        out = out.reset_index().rename(columns={out.index.name: "Team"})
        return out
    # Unnamed object index that looks like team names → promote
    if out.index.dtype == "object":
        sample = out.index[:5].tolist()
        if all(isinstance(x, str) for x in sample):
            out = out.reset_index().rename(columns={"index": "Team"})
            return out
    return out

def _robust_elo_snapshot(last_elo: pd.DataFrame, last_elo2: pd.DataFrame) -> pd.DataFrame:
    cand = pd.concat([last_elo, last_elo2], ignore_index=False)

    cand = _maybe_promote_index_to_team(cand)

    # Normalize likely variants
    rename_map = {}
    for c in cand.columns:
        lc = c.lower()
        if lc == "club" and "Team" not in cand.columns:
            rename_map[c] = "Team"
        elif lc == "team" and "Team" not in cand.columns:
            rename_map[c] = "Team"
        elif lc == "date" and "Date" not in cand.columns:
            rename_map[c] = "Date"
        elif lc == "elo" and "elo" not in cand.columns:
            rename_map[c] = "elo"
    cand = cand.rename(columns=rename_map)

    # Fix Elo casing
    if "elo" not in cand.columns:
        for alt in ["Elo","ELO"]:
            if alt in cand.columns:
                cand = cand.rename(columns={alt:"elo"})
                break

    missing = [x for x in ["Team","elo"] if x not in cand.columns]
    if missing:
        preview = cand.head(5).to_dict(orient="list")
        raise KeyError(
            f"Expected columns {missing} in Elo snapshot; got columns={list(cand.columns)} "
            f"(index name={cand.index.name!r}). Preview: {preview}"
        )

    if "Date" not in cand.columns:
        cand["Date"] = pd.Timestamp("1970-01-01")

    out = (
        cand.sort_values("Date")
            .drop_duplicates(subset=["Team"], keep="last")
            [["Team","elo"]]
            .reset_index(drop=True)
    )
    return out

# ----------------------------
# Public entry points
# ----------------------------
def build_features(seasons: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and persist features + team_state.
    Returns (features_df, team_state_df)
    """
    df = load_raw(seasons)
    if df.empty:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        # Write empty to keep downstream code predictable
        features_fp = PROC_DIR / "features.parquet"
        team_fp = PROC_DIR / "team_state.parquet"
        pd.DataFrame().to_parquet(features_fp, index=False)
        pd.DataFrame().to_parquet(team_fp, index=False)
        return df, pd.DataFrame()

    df = df.sort_values("Date").reset_index(drop=True)

    # Long and pre features
    long = make_long_frame(df)
    long = add_pre_match_rollups(long, window=5)

    # Elo time series (pre-update ratings logged each match)
    elo_df = compute_elo_timeseries(df, base_rating=1500.0, K=22.0, hfa=60.0)

    # Assemble per-match features
    feats = assemble_match_features(df, long, elo_df)

    # ------- Build team_state (latest snapshot per team) -------
    # Take latest date seen overall
    last_day = feats["Date"].max()

    # Last per-team rolling stats on/just before last_day
    long_latest = (
        long.sort_values("Date")
            .drop_duplicates(subset=["Team"], keep="last")
            .loc[:, ["Team",
                     "team_last_gf","team_last_ga","team_last_gd",
                     "team_last_sot_for","team_last_sot_against",
                     "team_rest_days"]]
            .rename(columns={
                "team_last_gf":"team_last_gf",
                "team_last_ga":"team_last_ga",
                "team_last_gd":"team_last_gd",
                "team_last_sot_for":"team_last_sot_for",
                "team_last_sot_against":"team_last_sot_against",
                "team_rest_days":"team_rest_days",
            })
            .reset_index(drop=True)
    )

    # Last Elo per team — robust against 'Team' in index
    elo_sorted = elo_df.sort_values("Date")
    last_elo = elo_sorted.groupby("Team", as_index=False).last(numeric_only=True)
    # Also try taking last on full frame in case of index issues (defensive)
    last_elo2 = elo_sorted.set_index("Team").groupby(level=0).tail(1).reset_index()

    elo_snap = _robust_elo_snapshot(last_elo, last_elo2)

    team_state = pd.merge(elo_snap, long_latest, on="Team", how="outer")

    # Persist
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "features.parquet").write_bytes(b"")  # ensure file exists even if conversion fails
    feats.to_parquet(PROC_DIR / "features.parquet", index=False)
    team_state.to_parquet(PROC_DIR / "team_state.parquet", index=False)

    print(f"Wrote {PROC_DIR/'features.parquet'} with shape={feats.shape}")
    print(f"Wrote {PROC_DIR/'team_state.parquet'} with shape={team_state.shape}")
    return feats, team_state


def run_build_features(seasons: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alias used by app.py"""
    return build_features(seasons=seasons)


def main():
    # CLI entry
    seasons_env = os.environ.get("SEASONS", "")
    seasons = None
    if seasons_env:
        seasons = [s.strip() for s in seasons_env.split(",") if s.strip()]
    build_features(seasons=seasons)


if __name__ == "__main__":
    main()
