# src/features.py
# Build training features + current team snapshot for the LaLiga predictor.
# Produces:
#   data/processed/match_features.parquet
#   data/processed/team_state.parquet

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG

RAW_DIR = Path(CONFIG["raw_data_path"])
PROC_DIR = Path(CONFIG["processed_data_path"])
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- small helpers -----------------------------

def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column name from `candidates` that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _num(s) -> pd.Series:
    """Coerce to numeric (float), keep NaN on errors."""
    return pd.to_numeric(s, errors="coerce")

def _safe_mean(values: deque, default: float = 0.0) -> float:
    return float(np.nanmean(values)) if len(values) else float(default)

# ----------------------------- CSV normalization -----------------------------

def _read_raw_csv(path: Path) -> pd.DataFrame:
    """
    Normalize one Football-Data.co.uk CSV into a standard frame with columns:
      Date (datetime64[ns]), HomeTeam, AwayTeam, FTHG, FTAG, FTR (H/D/A),
      HST (opt), AST (opt)
    """
    df = pd.read_csv(path, encoding="latin-1")

    # Identify relevant columns under common aliases
    date_col = _first_col(df, ["Date", "DATE"])
    home_col = _first_col(df, ["HomeTeam", "Home", "HT", "Home Team"])
    away_col = _first_col(df, ["AwayTeam", "Away", "AT", "Away Team"])
    fthg_col = _first_col(df, ["FTHG", "HG", "HomeGoals", "HomeGoalsFT"])
    ftag_col = _first_col(df, ["FTAG", "AG", "AwayGoals", "AwayGoalsFT"])
    ftr_col  = _first_col(df, ["FTR", "Res", "ResultFT"])

    hst_col  = _first_col(df, ["HST", "HSOT", "HomeShotsOnTarget"])
    ast_col  = _first_col(df, ["AST", "ASOT", "AwayShotsOnTarget"])

    # Validate essentials
    needed = {"HomeTeam": home_col, "AwayTeam": away_col, "FTHG": fthg_col, "FTAG": ftag_col}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise KeyError(f"{path.name}: missing essential columns for: {', '.join(missing)}")

    # Build normalized frame
    out = pd.DataFrame({
        "HomeTeam": df[home_col].astype(str).str.strip(),
        "AwayTeam": df[away_col].astype(str).str.strip(),
        "FTHG": _num(df[fthg_col]),
        "FTAG": _num(df[ftag_col]),
    })

    # Date
    if date_col:
        # Football-Data often uses dd/mm/yy -> dayfirst=True
        out["Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    else:
        # best effort fallback: unsorted or missing date -> NaT
        out["Date"] = pd.NaT

    # Shots on target if available
    if hst_col:
        out["HST"] = _num(df[hst_col])
    else:
        out["HST"] = np.nan
    if ast_col:
        out["AST"] = _num(df[ast_col])
    else:
        out["AST"] = np.nan

    # Label FTR
    if ftr_col:
        ftr = df[ftr_col].astype(str).str.upper().str.strip()
        ftr = ftr.replace({"HOME":"H", "AWAY":"A", "DRAW":"D"})
        ftr = ftr.where(ftr.isin(["H", "D", "A"]), np.nan)
    else:
        ftr = pd.Series(np.nan, index=out.index)

    # If still NaN, derive from goals
    need_fill = ftr.isna()
    if need_fill.any():
        derived = np.where(out["FTHG"] > out["FTAG"], "H",
                   np.where(out["FTHG"] < out["FTAG"], "A", "D"))
        ftr = ftr.fillna(pd.Series(derived, index=ftr.index))

    out["FTR"] = ftr

    # Clean impossible rows
    out = out.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    # Order by date (NaT last)
    out = out.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return out

# ----------------------------- Elo + rolling features -----------------------------

@dataclass
class TeamState:
    elo: float = 1500.0
    last_date: Optional[pd.Timestamp] = None
    gf: deque = field(default_factory=lambda: deque(maxlen=5))
    ga: deque = field(default_factory=lambda: deque(maxlen=5))
    gd: deque = field(default_factory=lambda: deque(maxlen=5))
    sot_for: deque = field(default_factory=lambda: deque(maxlen=5))
    sot_against: deque = field(default_factory=lambda: deque(maxlen=5))

    def snapshot(self) -> Dict[str, float]:
        return dict(
            elo=self.elo,
            team_last_gf=_safe_mean(self.gf, 0.0),
            team_last_ga=_safe_mean(self.ga, 0.0),
            team_last_gd=_safe_mean(self.gd, 0.0),
            team_last_sot_for=_safe_mean(self.sot_for, 0.0),
            team_last_sot_against=_safe_mean(self.sot_against, 0.0),
            # Rest days = 7 by convention if unknown
            team_rest_days=np.nan,
        )

def _expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def _elo_update(elo_home: float, elo_away: float, outcome: float,
                k: float = 20.0, home_adv: float = 60.0) -> Tuple[float, float]:
    # outcome from home perspective: 1 = home win, 0.5 = draw, 0 = away win
    exp_home = _expected_score(elo_home + home_adv, elo_away)
    delta = k * (outcome - exp_home)
    return elo_home + delta, elo_away - delta

# ----------------------------- main builder -----------------------------

def build_features(seasons: Optional[List[str]] = None) -> None:
    """
    Build:
      - match_features.parquet : one row per match with *pre-match* features and label
      - team_state.parquet     : latest per-team snapshot for the Predict UI
    """
    # choose files
    if seasons:
        candidates = [RAW_DIR / f"{CONFIG['league']}_{s}.csv" for s in seasons]
    else:
        candidates = sorted(RAW_DIR.glob(f"{CONFIG['league']}_*.csv"))

    frames: List[pd.DataFrame] = []
    for p in candidates:
        if p.exists():
            frames.append(_read_raw_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No CSVs found in {RAW_DIR}. Expected files like '{CONFIG['league']}_2324.csv' etc."
        )

    df = pd.concat(frames, ignore_index=True)
    # Ensure dates monotonic for Elo updates; push NaT to end deterministically
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)

    # Iterative feature build (pre-match)
    states: Dict[str, TeamState] = {}
    feature_rows: List[Dict[str, float]] = []

    global_last_date = df["Date"].dropna().max()
    if pd.isna(global_last_date):
        # extremely unlikely but guard
        global_last_date = pd.Timestamp.today().normalize()

    for _, row in df.iterrows():
        ht = str(row["HomeTeam"])
        at = str(row["AwayTeam"])
        date: pd.Timestamp = row["Date"]
        fthg = float(row["FTHG"])
        ftag = float(row["FTAG"])
        hst = float(row["HST"]) if not math.isnan(row["HST"]) else np.nan
        ast = float(row["AST"]) if not math.isnan(row["AST"]) else np.nan
        ftr = row["FTR"]

        sH = states.setdefault(ht, TeamState())
        sA = states.setdefault(at, TeamState())

        # Rest days (pre-match)
        rest_h = (date - sH.last_date).days if (sH.last_date is not None and pd.notna(date)) else 7
        rest_a = (date - sA.last_date).days if (sA.last_date is not None and pd.notna(date)) else 7

        # Pre-match snapshot means
        features = {
            "Date": date,
            "home_team": ht,
            "away_team": at,

            # Elo pre-match
            "elo_home": sH.elo,
            "elo_away": sA.elo,
            "elo_diff": sH.elo - sA.elo,

            # Form (rolling 5-match means)
            "home_last_gf": _safe_mean(sH.gf, 0.0),
            "home_last_ga": _safe_mean(sH.ga, 0.0),
            "home_last_gd": _safe_mean(sH.gd, 0.0),
            "home_last_sot_for": _safe_mean(sH.sot_for, 0.0),
            "home_last_sot_against": _safe_mean(sH.sot_against, 0.0),

            "away_last_gf": _safe_mean(sA.gf, 0.0),
            "away_last_ga": _safe_mean(sA.ga, 0.0),
            "away_last_gd": _safe_mean(sA.gd, 0.0),
            "away_last_sot_for": _safe_mean(sA.sot_for, 0.0),
            "away_last_sot_against": _safe_mean(sA.sot_against, 0.0),

            "rest_days_home": float(rest_h),
            "rest_days_away": float(rest_a),

            # Label
            "FTR": ftr,  # H/D/A
        }
        feature_rows.append(features)

        # ----- Post-match state updates -----
        # Elo
        if ftr == "H":
            outcome = 1.0
        elif ftr == "D":
            outcome = 0.5
        else:
            outcome = 0.0
        newH, newA = _elo_update(sH.elo, sA.elo, outcome)
        sH.elo, sA.elo = newH, newA

        # Rolling forms
        sH.gf.append(fthg)
        sH.ga.append(ftag)
        sH.gd.append(fthg - ftag)
        if not math.isnan(hst): sH.sot_for.append(hst)
        if not math.isnan(ast): sH.sot_against.append(ast)

        sA.gf.append(ftag)
        sA.ga.append(fthg)
        sA.gd.append(ftag - fthg)
        if not math.isnan(ast): sA.sot_for.append(ast)
        if not math.isnan(hst): sA.sot_against.append(hst)

        # Last dates
        if pd.notna(date):
            sH.last_date = date
            sA.last_date = date

    # Match-feature table
    feats = pd.DataFrame(feature_rows)
    # One-hot/categorical label for training convenience kept as 'FTR'
    feats.to_parquet(PROC_DIR / "match_features.parquet", index=False)

    # Latest per-team snapshot for Predict tab
    snap_rows = []
    for team, st in states.items():
        snap = st.snapshot()
        # Rest days vs latest known date
        if st.last_date is not None and pd.notna(st.last_date):
            snap["team_rest_days"] = float((global_last_date - st.last_date).days)
        else:
            snap["team_rest_days"] = 7.0
        snap["Team"] = team
        snap_rows.append(snap)

    team_state = pd.DataFrame(snap_rows)
    team_state = team_state[[
        "Team",
        "elo",
        "team_last_gf", "team_last_ga", "team_last_gd",
        "team_last_sot_for", "team_last_sot_against",
        "team_rest_days",
    ]].sort_values("Team")
    team_state.to_parquet(PROC_DIR / "team_state.parquet", index=False)

# ----------------------------- CLI entry -----------------------------

def main():
    """Allow:  python -m src.features  (uses all CSVs found)."""
    build_features(seasons=None)
    print("Built features and team_state →", PROC_DIR)

if __name__ == "__main__":
    main()
