# src/understat_loader.py
# Fetch La Liga xG data from Understat and save for feature merge.
# Optional: if understatapi is not installed or fetch fails, features fall back without xG.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import CONFIG

RAW_DIR = Path(CONFIG["raw_data_path"])
PROC_DIR = Path(CONFIG["processed_data_path"])
XG_PATH = PROC_DIR / "understat_xg.parquet"

# Understat La Liga team names -> football-data.co.uk equivalents
UNDERSTAT_TO_FD: Dict[str, str] = {
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Athletic Club": "Athletic Club",
    "Real Sociedad": "Real Sociedad",
    "Real Betis": "Real Betis",
    "Villarreal": "Villarreal",
    "Sevilla": "Sevilla",
    "Valencia": "Valencia",
    "Celta Vigo": "Celta Vigo",
    "Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Mallorca": "Mallorca",
    "Las Palmas": "Las Palmas",
    "Cadiz": "Cadiz",
    "Alaves": "Alaves",
    "Granada": "Granada",
    "Almeria": "Almeria",
    "Valladolid": "Valladolid",
    "Espanyol": "Espanyol",
    "Elche": "Elche",
    "Levante": "Levante",
    "Leganes": "Leganes",
    "Eibar": "Eibar",
    "Huesca": "Huesca",
    "Real Valladolid": "Valladolid",
}

SEASON_MAP = {"14": "1415", "15": "1516", "16": "1617", "17": "1718", "18": "1819",
              "19": "1920", "20": "2021", "21": "2122", "22": "2223", "23": "2324", "24": "2425"}


def _norm_team(name: str) -> str:
    return (name or "").replace("_", " ").strip()


def fetch_understat_xg(seasons: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch La Liga match xG from Understat via understatapi.
    Returns DataFrame with columns: Date, HomeTeam, AwayTeam, xg_home, xg_away, Season
    """
    try:
        from understatapi import UnderstatClient
    except ImportError:
        return pd.DataFrame()

    season_codes = seasons or ["2122", "2223", "2324", "2425"]
    all_matches: List[Dict] = []
    seen: set = set()

    with UnderstatClient() as understat:
        for sc in season_codes:
            us_year = sc[:2]  # 2122 -> 21 (Understat uses "21" for 2021/22)
            try:
                league_data = understat.league(league="La_liga").get_match_data(season=us_year)
            except Exception:
                continue
            if not league_data:
                continue
            for m in league_data:
                key = (m.get("date"), m.get("h"), m.get("a"))
                if key in seen:
                    continue
                seen.add(key)
                h_team = UNDERSTAT_TO_FD.get(
                    _norm_team(m.get("h", "")),
                    _norm_team(m.get("h", "")),
                )
                a_team = UNDERSTAT_TO_FD.get(
                    _norm_team(m.get("a", "")),
                    _norm_team(m.get("a", "")),
                )
                xg_home = float(m.get("xG", m.get("hxG", 0)) or 0)
                xg_away = float(m.get("xGA", m.get("axG", 0)) or 0)
                all_matches.append({
                    "Date": m.get("date"),
                    "HomeTeam": h_team,
                    "AwayTeam": a_team,
                    "xg_home": xg_home,
                    "xg_away": xg_away,
                    "Season": sc,
                })

    if not all_matches:
        return pd.DataFrame()
    df = pd.DataFrame(all_matches)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_and_merge_xg(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load cached Understat xG (or fetch if missing) and merge into base_df.
    Adds xg_home, xg_away, xg_diff columns.
    """
    if base_df.empty:
        return base_df
    xg_df = _load_xg()
    if xg_df.empty:
        base_df["xg_home"] = pd.NA
        base_df["xg_away"] = pd.NA
        base_df["xg_diff"] = 0.0
        return base_df
    base_df["Date"] = pd.to_datetime(base_df["Date"])
    xg_df["Date"] = pd.to_datetime(xg_df["Date"]).dt.normalize()
    base_df["_date_norm"] = base_df["Date"].dt.normalize()
    merged = base_df.merge(
        xg_df[["Date", "HomeTeam", "AwayTeam", "xg_home", "xg_away"]],
        left_on=["_date_norm", "HomeTeam", "AwayTeam"],
        right_on=["Date", "HomeTeam", "AwayTeam"],
        how="left",
        suffixes=("", "_xg"),
    )
    if "xg_home" in merged.columns:
        merged["xg_diff"] = (merged["xg_home"].fillna(0) - merged["xg_away"].fillna(0)).astype(float)
    else:
        merged["xg_home"] = pd.NA
        merged["xg_away"] = pd.NA
        merged["xg_diff"] = 0.0
    merged = merged.drop(columns=["_date_norm"], errors="ignore")
    merged = merged.drop(columns=["Date_xg"], errors="ignore")
    return merged


def _load_xg() -> pd.DataFrame:
    if XG_PATH.exists():
        return pd.read_parquet(XG_PATH)
    return pd.DataFrame()


def refresh_xg(seasons: Optional[List[str]] = None) -> bool:
    """Fetch xG from Understat and save. Returns True on success."""
    df = fetch_understat_xg(seasons)
    if df.empty:
        return False
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(XG_PATH, index=False)
    return True
