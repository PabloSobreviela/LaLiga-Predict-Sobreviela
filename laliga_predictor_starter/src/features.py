# src/features.py
# End-to-end feature builder for LaLiga predictor
# - Robust CSV reader for football-data.co.uk (handles small schema variations)
# - Safe FTR normalization (falls back to goals with an aligned Series)
# - Team-long table with rest-days and rolling "recent form"
# - Simple Elo with home-advantage; final Elo snapshot exported
# - Match-level features parquet + team_state snapshot parquet

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG

RAW_DIR = Path(CONFIG["raw_data_path"])
PROC_DIR = Path(CONFIG["processed_data_path"])
LEAGUE_PREFIX = CONFIG.get("league", "SP1")  # used for file names like SP1_2324.csv

# ------------------------------ CSV READER ------------------------------ #

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first column present in df whose lowercase name matches any candidate
    (also in lowercase). If not found, return None.
    """
    lowmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read one football-data CSV and normalize to:
      Date (datetime64), Home, Away, FTHG (int), FTAG (int), FTR (H/D/A)
    Handles mild naming / encoding differences and missing FTR by deriving from goals.
    """
    df = pd.read_csv(path)

    # Likely column names in football-data
    date_col = _find_col(df, ["Date"])
    home_col = _find_col(df, ["HomeTeam", "Home", "Home Team", "Home_Team"])
    away_col = _find_col(df, ["AwayTeam", "Away", "Away Team", "Away_Team"])
    fthg_col = _find_col(df, ["FTHG", "HomeGoals", "HG"])
    ftag_col = _find_col(df, ["FTAG", "AwayGoals", "AG"])
    ftr_col  = _find_col(df, ["FTR", "Result"])

    # Date parsing: football-data dates are day-first (e.g., 21/08/2024)
    if date_col:
        date_parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, utc=False)
    else:
        # If somehow no date column exists, try to coerce the index (rare/fallback)
        date_parsed = pd.to_datetime(df.index, errors="coerce", utc=False)

    # Goals numeric
    if fthg_col is None or ftag_col is None:
        raise KeyError(
            f"Could not find goal columns in {path.name}. "
            f"Found columns: {list(df.columns)}"
        )
    hg = pd.to_numeric(df[fthg_col], errors="coerce")
    ag = pd.to_numeric(df[ftag_col], errors="coerce")

    # FTR robust normalization
    if ftr_col is not None:
        # Keep only valid codes; coerce rest to NaN
        ftr_raw = df[ftr_col].astype(str).str.upper()
        ftr_valid = ftr_raw.where(ftr_raw.isin(["H", "D", "A"]))
    else:
        ftr_valid = pd.Series(index=df.index, dtype="object")

    # Build an ALIGNED Series fallback from goals (important: not a bare ndarray)
    fallback = pd.Series(
        np.where(hg > ag, "H", np.where(hg < ag, "A", "D")),
        index=df.index
    )

    out = pd.DataFrame(
        {
            "Date": date_parsed,
            "Home": df[home_col].astype(str) if home_col else pd.Series("", index=df.index, dtype=str),
            "Away": df[away_col].astype(str) if away_col else pd.Series("", index=df.index, dtype=str),
            "FTHG": hg,
            "FTAG": ag,
            "FTR":  ftr_valid.fillna(fallback),
        }
    )

    # Drop rows with missing criticals
    out = out.dropna(subset=["Date", "Home", "Away"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=False)
    out = out.dropna(subset=["Date"]).reset_index(drop=True)

    # Ensure types
    out["FTHG"] = out["FTHG"].fillna(0).astype(int)
    out["FTAG"] = out["FTAG"].fillna(0).astype(int)
    out["FTR"]  = out["FTR"].where(out["FTR"].isin(["H", "D", "A"])).fillna(
        np.where(out["FTHG"] > out["FTAG"], "H", np.where(out["FTHG"] < out["FTAG"], "A", "D"))
    )

    return out


# --------------------------- LONG TABLE & FORM -------------------------- #

def _long_from_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-team long table:
      - Date, Team, Opponent, GF, GA, is_home, points, result
      - 'rest_days' between team matches
    """
    # Home rows
    home_long = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["Home"],
            "Opponent": df["Away"],
            "GF": df["FTHG"],
            "GA": df["FTAG"],
            "is_home": True,
            "result": df["FTR"].map({"H": 1, "D": 0, "A": -1}),
            "points": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
            "sot_for": np.nan,      # placeholder if you later add SoT from extended CSVs
            "sot_against": np.nan,
        }
    )

    # Away rows
    away_long = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["Away"],
            "Opponent": df["Home"],
            "GF": df["FTAG"],
            "GA": df["FTHG"],
            "is_home": False,
            "result": df["FTR"].map({"A": 1, "D": 0, "H": -1}),
            "points": df["FTR"].map({"A": 3, "D": 1, "H": 0}),
            "sot_for": np.nan,
            "sot_against": np.nan,
        }
    )

    long = pd.concat([home_long, away_long], ignore_index=True).sort_values("Date")
    # rest days per team
    long["rest_days"] = (
        long.sort_values(["Team", "Date"])
            .groupby("Team")["Date"]
            .diff()
            .dt.days
            .fillna(7)  # start-of-season default rest
    )

    # simple rolling "recent form" stats (last 5)
    long = long.sort_values(["Team", "Date"])
    for col in ["GF", "GA", "points", "sot_for", "sot_against"]:
        long[f"last5_{col}"] = (
            long.groupby("Team")[col]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    long["last5_gd"] = long["last5_GF"] - long["last5_GA"]

    return long.reset_index(drop=True)


# --------------------------------- ELO ---------------------------------- #

def _elo_expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def _elo_update(ra: float, rb: float, score_a: float, k: float = 20.0) -> Tuple[float, float]:
    ea = _elo_expect(ra, rb)
    eb = 1.0 - ea
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1.0 - score_a) - eb)
    return ra2, rb2

def compute_elo(long_df: pd.DataFrame, base_rating: float = 1500.0, k: float = 20.0, home_adv: float = 50.0
                ) -> pd.DataFrame:
    """
    Compute simple Elo on the long table. Home team gets +home_adv to its rating
    for the expectation calculation only.
    Returns a per-row Elo AFTER the match for each team. Also returns a final snapshot.
    """
    ratings: Dict[str, float] = {}
    elo_rows = []

    for _, row in long_df.sort_values("Date").iterrows():
        team = row["Team"]
        opp  = row["Opponent"]
        is_home = bool(row["is_home"])
        res = row["result"]  # 1, 0, -1

        ra = ratings.get(team, base_rating)
        rb = ratings.get(opp,  base_rating)

        # Translate result to 1/0/0.5-ish (we used 1/0/-1 for convenience above)
        if res > 0:
            score_a = 1.0
        elif res < 0:
            score_a = 0.0
        else:
            score_a = 0.5

        # Home advantage only in expectation
        ra_eff = ra + (home_adv if is_home else 0.0)
        rb_eff = rb + (home_adv if not is_home else 0.0)

        ra2, rb2 = _elo_update(ra_eff, rb_eff, score_a, k=k)

        # Remove the added home_adv from stored rating to keep ratings comparable
        if is_home:
            # ra_eff = ra + HA; the delta is the same, so apply delta to original ra
            delta = ra2 - ra_eff
            ra_store = ra + delta
            rb_store = rb + (rb2 - rb_eff)
        else:
            delta = ra2 - ra_eff
            ra_store = ra + delta
            rb_store = rb + (rb2 - rb_eff)

        ratings[team] = ra_store
        ratings[opp]  = rb_store

        elo_rows.append(
            {
                "Date": row["Date"],
                "Team": team,
                "elo": ratings[team],
            }
        )

    elo_df = pd.DataFrame(elo_rows).sort_values(["Team", "Date"]).reset_index(drop=True)
    return elo_df


# ------------------------------ FEATURESET ------------------------------ #

def _most_recent_snapshot(elo_df: pd.DataFrame, on_date_col: str = "Date") -> pd.DataFrame:
    """
    For each team, take the last row (max date). Returns columns: Team, elo (and Date).
    """
    if "Team" not in elo_df.columns:
        raise KeyError(f"Expected 'Team' in Elo table; got columns={list(elo_df.columns)}")
    snap = (
        elo_df.sort_values(on_date_col)
              .dropna(subset=["Team"])
              .drop_duplicates("Team", keep="last")
    )
    return snap[["Team", "elo", on_date_col]].rename(columns={on_date_col: "Date"})


def assemble_match_features(matches: pd.DataFrame,
                            long_df: pd.DataFrame,
                            elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-match features using recent form & Elo.
    Output columns include:
      - elo_home, elo_away, elo_diff
      - home_last_* / away_last_* (+ diffs)
      - rest_days_home/away (+ diff)
      - label FTR, and numeric y (0,1,2)
    """
    df = matches.sort_values("Date").reset_index(drop=True)

    # Prepare lookup frames keyed by (Team, Date) → last5 stats & rest_days *before* match
    # We'll find the latest long row for team up to the match date.
    long_df = long_df.sort_values(["Team", "Date"]).copy()

    # Helper: for a given team, at match date, fetch last known row BEFORE that date
    def _last_row_before(team: str, date: pd.Timestamp) -> pd.Series:
        sub = long_df[(long_df["Team"] == team) & (long_df["Date"] < date)]
        if sub.empty:
            return pd.Series(dtype=float)
        return sub.iloc[-1]

    features = []
    for _, m in df.iterrows():
        date = m["Date"]
        h, a = m["Home"], m["Away"]

        # Elo at-most-current
        elo_h = elo_df[(elo_df["Team"] == h) & (elo_df["Date"] <= date)]
        elo_a = elo_df[(elo_df["Team"] == a) & (elo_df["Date"] <= date)]
        elo_home = float(elo_h["elo"].iloc[-1]) if not elo_h.empty else 1500.0
        elo_away = float(elo_a["elo"].iloc[-1]) if not elo_a.empty else 1500.0

        # last5 stats
        lh = _last_row_before(h, date)
        la = _last_row_before(a, date)

        def _get(s: pd.Series, key: str, default: float = 0.0) -> float:
            try:
                val = float(s.get(key, default))
                if np.isnan(val):
                    return default
                return val
            except Exception:
                return default

        row = {
            "Date": date,
            "Home": h,
            "Away": a,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,
            "home_last_gf": _get(lh, "last5_GF"),
            "home_last_ga": _get(lh, "last5_GA"),
            "home_last_gd": _get(lh, "last5_gd"),
            "home_last_sot_for": _get(lh, "last5_sot_for"),
            "home_last_sot_against": _get(lh, "last5_sot_against"),
            "rest_days_home": _get(lh, "rest_days", 7.0),
            "away_last_gf": _get(la, "last5_GF"),
            "away_last_ga": _get(la, "last5_GA"),
            "away_last_gd": _get(la, "last5_gd"),
            "away_last_sot_for": _get(la, "last5_sot_for"),
            "away_last_sot_against": _get(la, "last5_sot_against"),
            "rest_days_away": _get(la, "rest_days", 7.0),
            # labels
            "FTR": m["FTR"],
            "FTHG": m["FTHG"],
            "FTAG": m["FTAG"],
        }

        # diffs
        row["form_gd_diff"] = row["home_last_gd"] - row["away_last_gd"]
        row["form_sot_for_diff"] = row["home_last_sot_for"] - row["away_last_sot_for"]
        row["form_sot_against_diff"] = row["home_last_sot_against"] - row["away_last_sot_against"]
        row["rest_days_diff"] = row["rest_days_home"] - row["rest_days_away"]

        features.append(row)

    feats = pd.DataFrame(features).sort_values("Date").reset_index(drop=True)

    # Encode numeric label y (0=H,1=D,2=A) for training convenience
    feats["y"] = feats["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)

    # Final NA cleanups for model-safety
    for col in feats.columns:
        if feats[col].dtype.kind in "fc":  # float or complex
            feats[col] = feats[col].fillna(0.0)
    return feats


# ------------------------------ ENTRY POINT ----------------------------- #

def build_features(seasons: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features for the provided seasons (list of 'YYZZ' strings), e.g. ['2122','2223','2324'].
    If seasons is None, use any CSV present in RAW_DIR matching f"{LEAGUE_PREFIX}_*.csv".
    Writes:
      - processed/features.parquet          (match-level features)
      - processed/team_state.parquet        (per-team snapshot: Elo + recent form)
    Returns (features_df, team_state_df)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Gather list of CSVs to load
    csv_paths: List[Path] = []
    if seasons:
        for s in seasons:
            p = RAW_DIR / f"{LEAGUE_PREFIX}_{s}.csv"
            if p.exists():
                csv_paths.append(p)
    else:
        csv_paths = sorted(RAW_DIR.glob(f"{LEAGUE_PREFIX}_*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No raw CSVs found in {RAW_DIR}. Expected names like {LEAGUE_PREFIX}_2324.csv"
        )

    # Read & concat normalized match tables
    frames = []
    for p in csv_paths:
        frames.append(_read_raw_csv(p))
    matches = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

    # Build team-long with "recent form" and rest-days
    long_df = _long_from_matches(matches)

    # Elo timeline per team
    elo_df = compute_elo(long_df, base_rating=1500.0, k=20.0, home_adv=50.0)

    # Assemble match-level features
    feats = assemble_match_features(matches, long_df, elo_df)

    # Team-state snapshot (for inference UI defaults)
    snap = _most_recent_snapshot(elo_df)  # Team, elo, Date (renamed)
    # add a small subset of long-based recents at snapshot time
    # we’ll take the last known values in long_df for each team:
    last_form = (
        long_df.sort_values("Date")
               .drop_duplicates("Team", keep="last")
               .loc[:, ["Team", "last5_GF", "last5_GA", "last5_gd", "last5_sot_for", "last5_sot_against", "rest_days"]]
               .rename(columns={
                   "last5_GF": "team_last_gf",
                   "last5_GA": "team_last_ga",
                   "last5_gd": "team_last_gd",
                   "last5_sot_for": "team_last_sot_for",
                   "last5_sot_against": "team_last_sot_against",
                   "rest_days": "team_rest_days",
               })
    )

    team_state = (
        snap.merge(last_form, on="Team", how="left")
            .sort_values("Team")
            .reset_index(drop=True)
    )

    # Fill numeric NAs in team_state
    for c in team_state.columns:
        if team_state[c].dtype.kind in "fc":
            team_state[c] = team_state[c].fillna(0.0)

    # Save
    feats_path = PROC_DIR / "features.parquet"
    team_state_path = PROC_DIR / "team_state.parquet"
    feats.to_parquet(feats_path, index=False)
    team_state.to_parquet(team_state_path, index=False)

    print(f"Wrote {feats_path} with shape={feats.shape}")
    print(f"Wrote {team_state_path} with shape={team_state.shape}")

    return feats, team_state


def main():
    """
    CLI entry:
      python -m src.features                # uses all CSVs found in RAW_DIR
      python -m src.features 2122 2223 2324 # uses explicit season codes
    """
    seasons = sys.argv[1:] if len(sys.argv) > 1 else None
    if seasons:
        print(f"Building features for seasons={seasons}")
    else:
        print("Building features for all CSVs in raw_data_path…")
    build_features(seasons=seasons)


if __name__ == "__main__":
    main()
