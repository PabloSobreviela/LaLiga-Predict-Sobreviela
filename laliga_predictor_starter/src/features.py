# src/features.py
# Builds processed features for model training and team_state for the UI.
# - Reads SP1_<season>.csv (LaLiga) from data/raw
# - (Optional) reads SP2_<season>.csv (Segunda) for promoted-team priors
# - Outputs:
#     data/processed/features.parquet
#     data/processed/team_state.parquet

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from .config import CONFIG

RAW_DIR = Path(CONFIG["raw_data_path"])
PROC_DIR = Path(CONFIG["processed_data_path"])
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- Cold-start constants for promoted teams ---
PROMOTED_ELO_PENALTY = 70    # Elo starts ~70 below LaLiga mean
SEG_ATTACK_SCALE     = 0.78  # Segunda -> LaLiga attack shrink
SEG_DEFENSE_SCALE    = 1.15  # Segunda -> LaLiga defense worse
SEG_SOT_SCALE        = 0.80  # Segunda -> LaLiga SoT shrink
BAYES_M              = 10    # prior strength for form smoothing

# --- Elo parameters ---
ELO_START   = 1500.0
ELO_K       = 22.0
ELO_HA      = 55.0  # home-advantage points


# ===================== Small Utilities =====================

def _safe_median(s, default=0.0) -> float:
    try:
        v = float(pd.Series(s).median())
        if pd.isna(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _league_medians(team_state_df: pd.DataFrame) -> dict:
    cols = [
        "elo", "team_last_gf", "team_last_ga", "team_last_gd",
        "team_last_sot_for", "team_last_sot_against", "team_rest_days"
    ]
    return {c: _safe_median(team_state_df[c], default=0.0) for c in cols if c in team_state_df.columns}


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_ftr(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'FTR' in {H,D,A}. If missing or dirty, derive from goals.
    if "FTR" not in df.columns:
        df["FTR"] = np.where(df["FTHG"] > df["FTAG"], "H",
                      np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    else:
        mask_bad = ~df["FTR"].isin(["H", "D", "A"])
        if mask_bad.any():
            fill = np.where(df["FTHG"] > df["FTAG"], "H",
                    np.where(df["FTHG"] < df["FTAG"], "A", "D"))
            df.loc[mask_bad, "FTR"] = fill[mask_bad]
    return df


# ===================== Reading Raw CSVs =====================

def _read_raw_csv(p: Path) -> pd.DataFrame:
    """
    Normalize SP1 raw CSV to a common schema:
      Date, HomeTeam, AwayTeam, FTHG, FTAG, HST, AST
    """
    df = pd.read_csv(p)

    date_col = _first_col(df, ["Date", "date", "MatchDate"])
    home_col = _first_col(df, ["HomeTeam", "Home", "HT", "Home_Team"])
    away_col = _first_col(df, ["AwayTeam", "Away", "AT", "Away_Team"])
    fthg_col = _first_col(df, ["FTHG", "HG", "HomeGoals"])
    ftag_col = _first_col(df, ["FTAG", "AG", "AwayGoals"])
    hst_col  = _first_col(df, ["HST", "HomeShotsOnTarget", "HSOT"])
    ast_col  = _first_col(df, ["AST", "AwayShotsOnTarget", "ASOT"])

    if any(c is None for c in [home_col, away_col, fthg_col, ftag_col]):
        raise ValueError(f"Raw file {p.name} missing essential columns")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.to_datetime("1970-01-01"),
        "HomeTeam": df[home_col].astype(str),
        "AwayTeam": df[away_col].astype(str),
        "FTHG": pd.to_numeric(df[fthg_col], errors="coerce"),
        "FTAG": pd.to_numeric(df[ftag_col], errors="coerce"),
    })

    # Optional SoT
    out["HST"] = pd.to_numeric(df[hst_col], errors="coerce") if hst_col else np.nan
    out["AST"] = pd.to_numeric(df[ast_col], errors="coerce") if ast_col else np.nan

    out = _ensure_ftr(out)
    # Drop rows with bad dates or teams
    out = out.dropna(subset=["HomeTeam", "AwayTeam"])
    # If any Date couldn't parse, set minimal increasing date just to keep order stable
    if out["Date"].isna().any():
        out["Date"] = out["Date"].fillna(pd.Timestamp("1970-01-01"))
    return out.sort_values("Date").reset_index(drop=True)


def _read_sp2_csvs(raw_dir: Path, seasons: List[str]) -> List[pd.DataFrame]:
    frames = []
    for s in seasons or []:
        p = raw_dir / f"SP2_{s}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                needed_any = {"HomeTeam","AwayTeam","FTHG","FTAG"}
                if not needed_any.issubset(set(df.columns)):
                    continue
                df["Season"] = s
                frames.append(df)
            except Exception:
                continue
    return frames


def _build_sp2_team_stats(sp2_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not sp2_frames:
        return pd.DataFrame(columns=[
            "Team","gp","gf_pg","ga_pg","sot_for_pg","sot_against_pg","finish_rank"
        ])

    df = pd.concat(sp2_frames, ignore_index=True, sort=False)

    rows = []
    for _, r in df.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        fthg = r.get("FTHG", np.nan); ftag = r.get("FTAG", np.nan)
        hst  = r.get("HST",  np.nan); ast  = r.get("AST",  np.nan)
        rows.append({"Team": h, "gf": fthg, "ga": ftag, "sot_for": hst, "sot_against": ast})
        rows.append({"Team": a, "gf": ftag, "ga": fthg, "sot_for": ast, "sot_against": hst})

    long = pd.DataFrame(rows)
    gp = long.groupby("Team").size().rename("gp")
    agg = long.groupby("Team").agg({
        "gf":"mean", "ga":"mean", "sot_for":"mean", "sot_against":"mean"
    }).rename(columns={
        "gf":"gf_pg", "ga":"ga_pg", "sot_for":"sot_for_pg", "sot_against":"sot_against_pg"
    })

    out = pd.concat([agg, gp], axis=1).reset_index().rename(columns={"index":"Team"})
    out["finish_rank"] = np.nan  # optional (not provided by CSV)
    return out


# ===================== Feature Engineering =====================

def _compute_rest_days(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for each team before each match date.
    """
    # Build previous match date per team
    prev_date = {}
    rest_days_home = []
    rest_days_away = []
    for _, r in df_sorted.iterrows():
        d = pd.Timestamp(r["Date"])
        h = r["HomeTeam"]; a = r["AwayTeam"]
        # home
        d_prev_h = prev_date.get(h)
        rest_days_home.append((d - d_prev_h).days if d_prev_h is not None else np.nan)
        prev_date[h] = d
        # away
        d_prev_a = prev_date.get(a)
        rest_days_away.append((d - d_prev_a).days if d_prev_a is not None else np.nan)
        prev_date[a] = d

    df_sorted = df_sorted.copy()
    df_sorted["rest_days_home"] = rest_days_home
    df_sorted["rest_days_away"] = rest_days_away
    return df_sorted


def _update_elo(elo_h: float, elo_a: float, res: str) -> tuple[float,float]:
    # result: H=home win, D=draw, A=away win
    # Expected scores with home advantage
    Eh = 1.0 / (1.0 + 10 ** (-( (elo_h + ELO_HA) - elo_a ) / 400.0))
    Ea = 1.0 - Eh
    if res == "H":
        Sh, Sa = 1.0, 0.0
    elif res == "A":
        Sh, Sa = 0.0, 1.0
    else:
        Sh, Sa = 0.5, 0.5
    elo_h2 = elo_h + ELO_K * (Sh - Eh)
    elo_a2 = elo_a + ELO_K * (Sa - Ea)
    return elo_h2, elo_a2


def _rolling_team_form(fixtures: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Build rolling (last-N) per-team form metrics BEFORE each match.
    Outputs columns for each fixture row:
      home_last_gf, home_last_ga, home_last_sot_for, home_last_sot_against, home_last_gd
      away_last_* (same)
    """
    # Long form by team side, ordered by date
    rows = []
    for _, r in fixtures.iterrows():
        d = r["Date"]; h = r["HomeTeam"]; a = r["AwayTeam"]
        fthg = r["FTHG"]; ftag = r["FTAG"]; hst = r["HST"]; ast = r["AST"]
        rows.append({"Date": d, "Team": h, "gf": fthg, "ga": ftag, "sot_for": hst, "sot_against": ast})
        rows.append({"Date": d, "Team": a, "gf": ftag, "ga": fthg, "sot_for": ast, "sot_against": hst})
    long = pd.DataFrame(rows).sort_values(["Team","Date"]).reset_index(drop=True)

    # rolling means prior to the current match: use shift(1)
    long["gf_roll"]  = long.groupby("Team")["gf"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["ga_roll"]  = long.groupby("Team")["ga"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["sf_roll"]  = long.groupby("Team")["sot_for"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["sa_roll"]  = long.groupby("Team")["sot_against"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["gd_roll"]  = long["gf_roll"] - long["ga_roll"]

    # pivot back to fixture rows (home/away)
    def _pick(team, date, col):
        # pick row for (team,date)
        m = (long["Team"]==team) & (long["Date"]==date)
        v = long.loc[m, col]
        return v.iloc[0] if len(v) else np.nan

    res = {
        "home_last_gf": [], "home_last_ga": [], "home_last_sot_for": [], "home_last_sot_against": [], "home_last_gd": [],
        "away_last_gf": [], "away_last_ga": [], "away_last_sot_for": [], "away_last_sot_against": [], "away_last_gd": [],
    }
    for _, r in fixtures.iterrows():
        d = r["Date"]; h = r["HomeTeam"]; a = r["AwayTeam"]
        res["home_last_gf"].append(_pick(h,d,"gf_roll"))
        res["home_last_ga"].append(_pick(h,d,"ga_roll"))
        res["home_last_sot_for"].append(_pick(h,d,"sf_roll"))
        res["home_last_sot_against"].append(_pick(h,d,"sa_roll"))
        res["home_last_gd"].append(_pick(h,d,"gd_roll"))

        res["away_last_gf"].append(_pick(a,d,"gf_roll"))
        res["away_last_ga"].append(_pick(a,d,"ga_roll"))
        res["away_last_sot_for"].append(_pick(a,d,"sf_roll"))
        res["away_last_sot_against"].append(_pick(a,d,"sa_roll"))
        res["away_last_gd"].append(_pick(a,d,"gd_roll"))

    return pd.DataFrame(res)


def _finalize_team_state(fixtures: pd.DataFrame,
                         elo_snap: pd.DataFrame,
                         form_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Build team_state with columns expected by app.py:
      Team, elo, team_last_gf, team_last_ga, team_last_gd,
      team_last_sot_for, team_last_sot_against, team_rest_days
    """
    # latest date per team for rest days (use last known rest_days from fixtures table)
    # First compute rest per team-side at each fixture date
    rest = fixtures[["Date","HomeTeam","AwayTeam","rest_days_home","rest_days_away"]].copy()
    rows = []
    for _, r in rest.iterrows():
        rows.append({"Team": r["HomeTeam"], "Date": r["Date"], "team_rest_days": r["rest_days_home"]})
        rows.append({"Team": r["AwayTeam"], "Date": r["Date"], "team_rest_days": r["rest_days_away"]})
    rest_long = pd.DataFrame(rows).sort_values(["Team","Date"])

    # last known per-team metrics from form_snap
    # form_snap rows are aligned with fixtures; we want the last per team
    # Build long view for form by team similarly:
    form_rows = []
    for i, r in fixtures.iterrows():
        d = r["Date"]; h = r["HomeTeam"]; a = r["AwayTeam"]
        form_rows.append({"Team": h, "Date": d,
                          "team_last_gf": form_snap.loc[i, "home_last_gf"],
                          "team_last_ga": form_snap.loc[i, "home_last_ga"],
                          "team_last_gd": form_snap.loc[i, "home_last_gd"],
                          "team_last_sot_for": form_snap.loc[i, "home_last_sot_for"],
                          "team_last_sot_against": form_snap.loc[i, "home_last_sot_against"]})
        form_rows.append({"Team": a, "Date": d,
                          "team_last_gf": form_snap.loc[i, "away_last_gf"],
                          "team_last_ga": form_snap.loc[i, "away_last_ga"],
                          "team_last_gd": form_snap.loc[i, "away_last_gd"],
                          "team_last_sot_for": form_snap.loc[i, "away_last_sot_for"],
                          "team_last_sot_against": form_snap.loc[i, "away_last_sot_against"]})
    form_long = pd.DataFrame(form_rows).sort_values(["Team","Date"])

    # take last observation per team
    rest_last = rest_long.sort_values("Date").dropna(subset=["team_rest_days"]).drop_duplicates("Team", keep="last")
    form_last = form_long.sort_values("Date").drop_duplicates("Team", keep="last")
    elo_last  = elo_snap.sort_values("Date").drop_duplicates("Team", keep="last")

    # merge them
    ts = pd.merge(elo_last[["Team","elo","Date"]], form_last.drop(columns=["Date"]), on="Team", how="outer")
    ts = pd.merge(ts, rest_last.drop(columns=["Date"]), on="Team", how="left")

    return ts


def _apply_promoted_priors(team_state_df: pd.DataFrame,
                           sp2_team_df: pd.DataFrame) -> pd.DataFrame:
    ts = team_state_df.copy()
    if ts.empty:
        return ts

    med = _league_medians(ts)
    sp2_map: Dict[str, Dict[str, Any]] = {r["Team"]: r for _, r in sp2_team_df.iterrows()}

    def _needs_prior(row) -> bool:
        keys = ["team_last_gf","team_last_ga","team_last_sot_for","team_last_sot_against"]
        bad = 0
        for k in keys:
            v = row.get(k, np.nan)
            if pd.isna(v) or abs(float(v)) < 1e-6:
                bad += 1
        return bad >= 2

    for idx, row in ts.iterrows():
        if not _needs_prior(row):
            continue

        team = idx
        r2 = sp2_map.get(team)
        if r2 is not None:
            gp = max(1.0, float(r2.get("gp", 0.0)))
            w  = gp / (gp + BAYES_M)

            gf_pg = float(r2.get("gf_pg", med.get("team_last_gf", 1.1))) * SEG_ATTACK_SCALE
            ga_pg = float(r2.get("ga_pg", med.get("team_last_ga", 1.1))) * SEG_DEFENSE_SCALE
            sf_pg = float(r2.get("sot_for_pg", med.get("team_last_sot_for", 3.5))) * SEG_SOT_SCALE
            sa_pg = float(r2.get("sot_against_pg", med.get("team_last_sot_against", 3.5))) * (2.0 - SEG_SOT_SCALE)

            ts.at[idx, "team_last_gf"]           = w * gf_pg + (1-w) * med.get("team_last_gf", 1.1)
            ts.at[idx, "team_last_ga"]           = w * ga_pg + (1-w) * med.get("team_last_ga", 1.1)
            ts.at[idx, "team_last_gd"]           = ts.at[idx, "team_last_gf"] - ts.at[idx, "team_last_ga"]
            ts.at[idx, "team_last_sot_for"]      = w * sf_pg + (1-w) * med.get("team_last_sot_for", 3.5)
            ts.at[idx, "team_last_sot_against"]  = w * sa_pg + (1-w) * med.get("team_last_sot_against", 3.5)

            elo_mean = _safe_median(ts["elo"], default=ELO_START)
            bonus = 0.0
            rank = r2.get("finish_rank", np.nan)
            if not pd.isna(rank):
                if rank <= 1: bonus = -20
                elif rank <= 3: bonus = -10
            ts.at[idx, "elo"] = elo_mean - PROMOTED_ELO_PENALTY + bonus
        else:
            ts.at[idx, "team_last_gf"]          = med.get("team_last_gf", 1.1)
            ts.at[idx, "team_last_ga"]          = med.get("team_last_ga", 1.1)
            ts.at[idx, "team_last_gd"]          = ts.at[idx, "team_last_gf"] - ts.at[idx, "team_last_ga"]
            ts.at[idx, "team_last_sot_for"]     = med.get("team_last_sot_for", 3.5)
            ts.at[idx, "team_last_sot_against"] = med.get("team_last_sot_against", 3.5)
            ts.at[idx, "elo"]                   = _safe_median(ts["elo"], default=ELO_START) - PROMOTED_ELO_PENALTY

        if "team_rest_days" in ts.columns:
            if pd.isna(ts.at[idx, "team_rest_days"]) or ts.at[idx, "team_rest_days"] <= 0:
                ts.at[idx, "team_rest_days"] = med.get("team_rest_days", 7.0)

    return ts


# ===================== Public entrypoint =====================

def build_features(seasons: Optional[List[str]] = None) -> None:
    """
    Build features & team_state for the selected seasons.
    Expects raw files:
      data/raw/SP1_<season>.csv   (LaLiga)
    Optional for priors:
      data/raw/SP2_<season>.csv   (Segunda)
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Read SP1 (LaLiga) raw fixtures across seasons --------
    frames = []
    for s in (seasons or []):
        p = RAW_DIR / f"SP1_{s}.csv"
        if not p.exists():
            # silently skip missing seasons (UI warns on download)
            continue
        frames.append(_read_raw_csv(p))

    if not frames:
        raise FileNotFoundError("No SP1_<season>.csv found in data/raw. Download some seasons first.")

    matches = pd.concat(frames, ignore_index=True, sort=False)
    matches = matches.sort_values("Date").reset_index(drop=True)

    # -------- Rest days (per team) --------
    matches = _compute_rest_days(matches)

    # -------- Elo over time --------
    # Track elo per team evolving through fixtures
    elo_map: Dict[str, float] = {}
    elo_rows = []
    for _, r in matches.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        res  = r["FTR"]
        eh = elo_map.get(h, ELO_START)
        ea = elo_map.get(a, ELO_START)
        # snapshot BEFORE match
        elo_rows.append({"Date": r["Date"], "Team": h, "elo": eh})
        elo_rows.append({"Date": r["Date"], "Team": a, "elo": ea})
        # update AFTER match result
        eh2, ea2 = _update_elo(eh, ea, res)
        elo_map[h] = eh2
        elo_map[a] = ea2
    elo_snap = pd.DataFrame(elo_rows)

    # -------- Rolling form (last 5 games) --------
    form = _rolling_team_form(matches, window=5)

    # -------- Assemble training features for each fixture --------
    feats = pd.DataFrame({
        "Date": matches["Date"],
        "HomeTeam": matches["HomeTeam"],
        "AwayTeam": matches["AwayTeam"],
        "FTR": matches["FTR"],
        "FTHG": matches["FTHG"],
        "FTAG": matches["FTAG"],
        "HST": matches["HST"],
        "AST": matches["AST"],
        "rest_days_home": matches["rest_days_home"],
        "rest_days_away": matches["rest_days_away"],
    })
    # Merge elo BEFORE match for home & away
    def _elo_at(team_col, date_col):
        m = pd.merge(
            feats[[team_col, date_col]].rename(columns={team_col: "Team", date_col: "Date"}),
            elo_snap, on=["Team","Date"], how="left"
        )
        return m["elo"]

    feats["elo_home"] = _elo_at("HomeTeam","Date")
    feats["elo_away"] = _elo_at("AwayTeam","Date")
    feats["elo_diff"] = feats["elo_home"] - feats["elo_away"]

    # Attach rolling form (already aligned with matches index)
    for c in ["home_last_gf","home_last_ga","home_last_sot_for","home_last_sot_against","home_last_gd",
              "away_last_gf","away_last_ga","away_last_sot_for","away_last_sot_against","away_last_gd"]:
        feats[c] = form[c]

    # Engineered diffs used in app.py
    feats["form_gd_diff"]           = feats["home_last_gd"] - feats["away_last_gd"]
    feats["form_sot_for_diff"]      = feats["home_last_sot_for"] - feats["away_last_sot_for"]
    feats["form_sot_against_diff"]  = feats["home_last_sot_against"] - feats["away_last_sot_against"]
    feats["rest_days_diff"]         = feats["rest_days_home"] - feats["rest_days_away"]

    # Numeric label target 0/1/2 (H/D/A)
    feats["target"] = feats["FTR"].map({"H":0, "D":1, "A":2}).astype(int)

    # NaN safety for training
    num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        feats[c] = feats[c].fillna(feats[c].median())

    # -------- Build team_state snapshot for the UI --------
    team_state = _finalize_team_state(matches, elo_snap, form)

    # Apply promoted priors using SP2 if available
    sp2_frames = _read_sp2_csvs(RAW_DIR, seasons or [])
    sp2_team   = _build_sp2_team_stats(sp2_frames)

    team_state = team_state.copy()
    if "Team" in team_state.columns:
        team_state = team_state.set_index("Team")

    team_state = _apply_promoted_priors(team_state, sp2_team)

    # Fill any remaining NaNs with medians
    meds = _league_medians(team_state)
    for c, m in meds.items():
        if c in team_state.columns:
            team_state[c] = team_state[c].fillna(m)

    # -------- Save outputs --------
    feats.to_parquet(PROC_DIR / "features.parquet", index=False)
    team_state.reset_index().to_parquet(PROC_DIR / "team_state.parquet", index=False)


# Allow module run: python -m src.features
def main():
    # Default: try to build with whatever SP1_* exist in raw
    # We infer seasons from files present
    sp1_files = sorted(RAW_DIR.glob("SP1_*.csv"))
    if not sp1_files:
        raise FileNotFoundError("No SP1_*.csv in data/raw. Download seasons first.")
    seasons = [p.stem.split("_")[1] for p in sp1_files]
    build_features(seasons=seasons)


if __name__ == "__main__":
    main()
