import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

import src.features as F
from src.config import CONFIG
from src.team_display import get_crest_url, get_display_name
from src.train_model import train as train_model, evaluate_bundle

DATA_RAW = Path(CONFIG["raw_data_path"])
DATA_PROC = Path(CONFIG["processed_data_path"])
MODEL_PATH = Path(CONFIG["model_path"])
FEATURES_PATH = DATA_PROC / "features.parquet"
TEAM_STATE_PATH = DATA_PROC / "team_state.parquet"

st.set_page_config(
    page_title="Match Predictor",
    page_icon="https://assets.laliga.com/assets/logos/LL_RGB_h_color/LL_RGB_h_color.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 0.5rem; padding-bottom: 1.5rem; max-width: 1100px;}
[data-testid="stAppViewContainer"] { background: #0f1419; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #2d3748; }
.stTabs [data-baseweb="tab"] {
  padding: .6rem 1.2rem; font-weight: 700; color: #94a3b8;
  border: none; background: transparent;
}
.stTabs [aria-selected="true"] { color: #F9A01B; border-bottom: 3px solid #F9A01B; }
.section-label {
  font-size: .7rem; font-weight: 800; letter-spacing: .2em; color: #F9A01B;
  margin-bottom: .5rem; text-transform: uppercase;
}
.section-box {
  background: #1a202c; border: 1px solid #2d3748; border-radius: 8px;
  padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.section-box h3 { margin: 0 0 .6rem 0; font-size: 1rem; color: #e2e8f0; font-weight: 600; }
.metric-row { display: flex; gap: .75rem; margin: .75rem 0; flex-wrap: wrap; }
.metric-item {
  background: #252d3a; border: 1px solid #2d3748; border-radius: 6px;
  padding: .5rem .75rem; flex: 1; min-width: 100px;
}
.metric-item .label { font-size: .65rem; color: #718096; text-transform: uppercase; }
.metric-item .val { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
.outcome-row { display: flex; gap: .5rem; margin: .75rem 0; justify-content: center; }
.outcome-box {
  flex: 1; background: #252d3a; border: 1px solid #2d3748; border-radius: 6px;
  padding: .6rem .8rem; text-align: center;
}
.outcome-box.winner { border-color: #F9A01B; background: #2d2a1f; }
.outcome-box .label { font-size: .75rem; color: #a0aec0; }
.outcome-box .pct { font-size: 1.4rem; font-weight: 800; color: #e2e8f0; }
.outcome-box .odds { font-size: .7rem; color: #718096; }
.result-bar {
  padding: .8rem 1rem; border-radius: 6px; font-weight: 700; font-size: 1.1rem;
  background: #1e3a5f; border-left: 4px solid #F9A01B; color: #e2e8f0;
}
.train-primary {
  background: #F9A01B !important; color: #0f1419 !important; font-weight: 700 !important;
  padding: .65rem 1.2rem !important; border-radius: 6px !important;
}
.predict-primary {
  background: #2b6cb0 !important; color: #e2e8f0 !important; font-weight: 700 !important;
  padding: .7rem 1.5rem !important; border-radius: 6px !important; font-size: 1rem !important;
}
.stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input {
  border-radius: 6px !important; border-color: #2d3748 !important;
}
.stButton > button[kind="primary"] { background: #2b6cb0 !important; color: #e2e8f0 !important; font-weight: 700 !important; }
div[data-testid="stVerticalBlock"]:has([data-testid="stButton"]) .stButton > button { border-radius: 6px !important; }
.match-display { display: flex; align-items: center; justify-content: center; width: 100%; margin: 0 auto; }
.match-display-inner { display: flex; align-items: center; justify-content: space-between; width: 100%; max-width: 600px; }
.match-left { flex: 1; display: flex; justify-content: flex-end; align-items: center; gap: .5rem; padding-right: 1rem; }
.match-vs { flex: 0 0 auto; font-weight: 800; color: #718096; letter-spacing: .1em; }
.match-right { flex: 1; display: flex; justify-content: flex-start; align-items: center; gap: .5rem; padding-left: 1rem; }
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

ODDS_SPORT_KEY = "soccer_spain_la_liga"
ODDS_EVENTS = "https://api.the-odds-api.com/v4/sports/{sport}/events"
ODDS_EVENT_ODDS = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
H2H_MARKET = "h2h"
DEFAULT_ODDS = {"H": 2.0, "D": 3.4, "A": 3.2, "src": None}

STATIC_ALIAS_EXTRAS: Dict[str, List[str]] = {
    "Barcelona": ["FC Barcelona", "Barca", "FCBarcelona", "Barcelona CF"],
    "Real Madrid": ["Real Madrid CF", "RealMadrid"],
    "Atletico Madrid": ["Atletico de Madrid", "Atletico Madrid"],
    "Athletic Club": ["Athletic Bilbao", "Athletic de Bilbao", "Athletic"],
    "Real Sociedad": ["Sociedad", "RealSociedad"],
    "Celta Vigo": ["RC Celta", "Celta de Vigo"],
    "Real Betis": ["Real Betis Balompie", "Betis"],
    "Villarreal": ["Villarreal CF"],
    "Osasuna": ["CA Osasuna"],
    "Alaves": ["Deportivo Alaves", "Alaves"],
    "Almeria": ["UD Almeria", "Almeria"],
    "Getafe": ["Getafe CF"],
    "Girona": ["Girona FC"],
    "Mallorca": ["RCD Mallorca"],
    "Las Palmas": ["UD Las Palmas"],
    "Leganes": ["CD Leganes", "Leganes"],
    "Cadiz": ["Cadiz CF", "Cadiz"],
    "Granada": ["Granada CF", "Granada CF SAD"],
    "Rayo Vallecano": ["Vallecano", "Rayo"],
    "Elche": ["Elche CF"],
    "Sevilla": ["Sevilla FC"],
    "Espanyol": ["RCD Espanyol"],
    "Levante": ["Levante UD"],
}
PREFIXES = ("fc ", "cf ", "cd ", "ud ", "rcd ", "rc ", "ca ", "real ", "club ")


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def season_code_to_label(code: str) -> str:
    return f"{code[:2]}/{code[2:]}"


def label_to_season_code(label: str) -> str:
    return label.replace("/", "")


def available_season_codes() -> List[str]:
    base = CONFIG.get("seasons", ["2122", "2223", "2324", "2425"])
    if "2526" not in base:
        return base + ["2526"]  # Add current for prediction dropdown
    return base


def fair_odds(probabilities: List[float]) -> List[float]:
    return [(1 / p if p > 0 else float("inf")) for p in probabilities]


def raw_data_files() -> List[Path]:
    return sorted(DATA_RAW.glob(f"{CONFIG['league']}_*.csv"))


def artifact_status() -> Dict[str, bool]:
    return {
        "raw": len(raw_data_files()) > 0,
        "features": FEATURES_PATH.exists(),
        "team_state": TEAM_STATE_PATH.exists(),
        "model": MODEL_PATH.exists(),
    }


LALIGA_LOGO_URL = "https://assets.laliga.com/assets/logos/LALIGA_RGB_h_color/LALIGA_RGB_h_color.png"


def render_hero(meta: Dict) -> None:
    seasons_used = meta.get("seasons_used") or []
    season_text = ", ".join(season_code_to_label(s) for s in seasons_used) if seasons_used else "—"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.25rem;margin-bottom:1rem;padding-bottom:.75rem;border-bottom:1px solid #2d3748;">
          <img src="{LALIGA_LOGO_URL}" alt="LALIGA" style="height:48px;width:auto;object-fit:contain;" onerror="this.style.display='none'"/>
          <div>
            <h1 style="margin:0;font-size:1.6rem;font-weight:800;color:#e2e8f0;">Match Predictor</h1>
            <span style="font-size:.85rem;color:#718096;">Seasons: {season_text} · Elo + form · football-data.co.uk</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: List[Tuple[str, str, str]]) -> None:
    items = [f'<div class="metric-item"><div class="label">{l}</div><div class="val">{v}</div><div class="label">{h}</div></div>' for l, v, h in metrics]
    st.markdown(f'<div class="metric-row">{"".join(items)}</div>', unsafe_allow_html=True)


def render_probability_cards(probabilities: List[float], fair_lines: List[float], winner_idx: int) -> None:
    labels = [("Home", probabilities[0], fair_lines[0]), ("Draw", probabilities[1], fair_lines[1]), ("Away", probabilities[2], fair_lines[2])]
    boxes = []
    for i, (label, prob, fair_line) in enumerate(labels):
        cls = " outcome-box winner" if i == winner_idx else " outcome-box"
        boxes.append(f'<div class="{cls.strip()}"><div class="label">{label}</div><div class="pct">{prob*100:.1f}%</div><div class="odds">Fair: {fair_line:.2f}</div></div>')
    st.markdown(f'<div class="outcome-row">{"".join(boxes)}</div>', unsafe_allow_html=True)


def build_alias_lookup(teams: List[str]) -> Dict[str, Set[str]]:
    lookup: Dict[str, Set[str]] = {}
    for team in teams:
        base = _norm(team)
        aliases: Set[str] = {base}
        team_lower = team.lower()
        for prefix in PREFIXES:
            if team_lower.startswith(prefix):
                aliases.add(_norm(team[len(prefix):]))
        for variant in STATIC_ALIAS_EXTRAS.get(team, []):
            aliases.add(_norm(variant))
        for canonical, variants in STATIC_ALIAS_EXTRAS.items():
            if _norm(canonical) == base:
                aliases.update({_norm(v) for v in variants})
        lookup[base] = aliases
    return lookup


def is_alias_match(team_a: str, team_b: str, alias_lookup: Dict[str, Set[str]]) -> bool:
    a_norm = _norm(team_a)
    b_norm = _norm(team_b)
    if a_norm in b_norm or b_norm in a_norm:
        return True
    a_aliases = alias_lookup.get(a_norm, {a_norm})
    b_aliases = alias_lookup.get(b_norm, {b_norm})
    return bool(a_aliases.intersection(b_aliases))


DEFAULT_ODDS_API_KEY = "f966622cfeab29bf23d64720a3e97e19"


def get_odds_api_key() -> str:
    api_key = st.session_state.get("odds_api_key", "")
    if not api_key:
        try:
            api_key = st.secrets.get("odds_api_key", "")
        except Exception:
            api_key = ""
    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        api_key = DEFAULT_ODDS_API_KEY
    return api_key


def show_quota(headers: Dict[str, str]) -> None:
    remaining = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    if remaining or used:
        st.caption(f"Odds API quota: remaining={remaining}, used={used}")


@st.cache_data(ttl=60)
def list_upcoming_events(api_key: str, days_from: int = 90) -> Tuple[List[Dict], Optional[str]]:
    if not api_key:
        return [], "No API key configured."
    try:
        response = requests.get(
            ODDS_EVENTS.format(sport=ODDS_SPORT_KEY),
            params={"apiKey": api_key, "daysFrom": str(days_from)},
            timeout=15,
        )
        show_quota(response.headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return [], f"Unexpected response: {payload}"
        return payload, None
    except Exception as exc:
        return [], f"Events request failed: {exc}"


def coerce_bookmaker_payload(payload) -> Tuple[Optional[List[Dict]], Optional[str]]:
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict) and isinstance(payload.get("bookmakers"), list):
        return payload["bookmakers"], None
    return None, f"Unexpected payload type: {type(payload).__name__}"


def fetch_live_odds(
    home: str,
    away: str,
    api_key: str,
    alias_lookup: Dict[str, Set[str]],
    days_from: int = 90,
) -> Tuple[Optional[Tuple[float, float, float, str]], Optional[str]]:
    if not api_key:
        return None, "No API key configured."
    try:
        events_response = requests.get(
            ODDS_EVENTS.format(sport=ODDS_SPORT_KEY),
            params={"apiKey": api_key, "daysFrom": str(days_from)},
            timeout=15,
        )
        show_quota(events_response.headers)
        events_response.raise_for_status()
        events = events_response.json()
    except Exception as exc:
        return None, f"Events request failed: {exc}"

    event_id = None
    for event in events if isinstance(events, list) else []:
        if is_alias_match(event.get("home_team", ""), home, alias_lookup) and is_alias_match(event.get("away_team", ""), away, alias_lookup):
            event_id = event.get("id")
            break
    if not event_id:
        return None, "Fixture not found in the upcoming Odds API window."

    try:
        odds_response = requests.get(
            ODDS_EVENT_ODDS.format(sport=ODDS_SPORT_KEY, event_id=event_id),
            params={"apiKey": api_key, "regions": "eu,uk,us", "markets": H2H_MARKET, "oddsFormat": "decimal"},
            timeout=15,
        )
        show_quota(odds_response.headers)
        odds_response.raise_for_status()
        payload = odds_response.json()
    except Exception as exc:
        return None, f"Event odds request failed: {exc}"

    bookmakers, _ = coerce_bookmaker_payload(payload)
    if bookmakers is None:
        return None, f"Unexpected response from event odds: {payload}"

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != H2H_MARKET:
                continue
            home_odds = draw_odds = away_odds = None
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue
                if _norm(name) in {"home", _norm(home)} or is_alias_match(name, home, alias_lookup):
                    home_odds = float(price)
                elif _norm(name) in {"away", _norm(away)} or is_alias_match(name, away, alias_lookup):
                    away_odds = float(price)
                elif _norm(name) in {"draw", "tie"}:
                    draw_odds = float(price)
            if home_odds and draw_odds and away_odds:
                return (home_odds, draw_odds, away_odds, bookmaker.get("title", "bookmaker")), None
    return None, "No complete 1X2 market was returned."


def normalize_probs_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> Tuple[List[float], float]:
    raw = np.array([1 / odds_home, 1 / odds_draw, 1 / odds_away], dtype=float)
    total = float(raw.sum())
    return (raw / total).tolist(), total if total > 0 else 0.0


def run_build_features(seasons: Optional[List[str]] = None):
    if hasattr(F, "build_features"):
        return F.build_features(seasons=seasons)
    if hasattr(F, "run_build_features"):
        return F.run_build_features(seasons=seasons)
    raise AttributeError("Neither build_features nor run_build_features exists in src.features.")


def download_missing_seasons(season_codes: List[str]) -> List[str]:
    notes: List[str] = []
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for season in season_codes:
        if season == "2526":
            notes.append("25/26 is metadata-only until a CSV is published.")
            continue
        target = DATA_RAW / f"{CONFIG['league']}_{season}.csv"
        if target.exists():
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            target.write_bytes(response.content)
            notes.append(f"Downloaded {season_code_to_label(season)}.")
        except Exception as exc:
            notes.append(f"Could not download {season_code_to_label(season)}: {exc}")
    return notes


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    meta = bundle.get("meta", {})
    # Recompute accuracy on current data with same split as training — ensures displayed = actual
    if FEATURES_PATH.exists():
        try:
            acc, prec, ll = evaluate_bundle(bundle)
            meta = {**meta, "accuracy_time_split": acc, "precision_time_split": prec, "logloss_time_split": ll}
        except Exception:
            pass
    return (
        bundle["pipeline"],
        bundle["feature_order"],
        bundle.get("feature_means", {}),
        meta,
    )


@st.cache_data
def load_team_state():
    if not TEAM_STATE_PATH.exists():
        raise FileNotFoundError(f"{TEAM_STATE_PATH} not found.")
    ts = pd.read_parquet(TEAM_STATE_PATH)
    if "Team" not in ts.columns:
        raise KeyError("team_state.parquet must contain a 'Team' column.")
    ts = ts.dropna(subset=["Team"]).set_index("Team").sort_index()
    return ts, ts.index.tolist()


@st.cache_data
def get_teams_for_season(predict_code: str) -> List[str]:
    """
    Return active teams for the given prediction season from features.parquet.
    If the season has no data (e.g. 25/26), fall back to the latest available season.
    """
    if not FEATURES_PATH.exists():
        return []
    df = pd.read_parquet(FEATURES_PATH)
    if "Season" not in df.columns or "HomeTeam" not in df.columns or "AwayTeam" not in df.columns or df.empty:
        return []
    sub = df[df["Season"].astype(str) == str(predict_code)]
    if sub.empty:
        seasons = df["Season"].dropna().astype(str).unique()
        if len(seasons) == 0:
            return []
        latest = sorted(seasons)[-1]
        sub = df[df["Season"].astype(str) == latest]
    if sub.empty:
        return []
    teams = pd.concat([sub["HomeTeam"], sub["AwayTeam"]]).dropna().astype(str).str.strip().unique()
    return sorted([t for t in teams if t])


def vector_for_match(home: str, away: str, feature_order, ts: pd.DataFrame, feature_means: Dict[str, float]) -> pd.DataFrame:
    row = pd.Series({col: float(feature_means.get(col, 0.0)) for col in feature_order}, dtype=float)
    if home not in ts.index or away not in ts.index:
        raise ValueError("Team not found in team state.")

    elo_home = float(ts.loc[home, "elo"])
    elo_away = float(ts.loc[away, "elo"])
    if "elo_home" in row.index:
        row["elo_home"] = elo_home
    if "elo_away" in row.index:
        row["elo_away"] = elo_away
    if "elo_diff" in row.index:
        row["elo_diff"] = elo_home - elo_away

    mapping = {
        "team_last_gf": ("home_last_gf", "away_last_gf"),
        "team_last_ga": ("home_last_ga", "away_last_ga"),
        "team_last_gd": ("home_last_gd", "away_last_gd"),
        "team_last_sot_for": ("home_last_sot_for", "away_last_sot_for"),
        "team_last_sot_against": ("home_last_sot_against", "away_last_sot_against"),
        "team_last_shots_for": ("home_last_shots_for", "away_last_shots_for"),
        "team_last_corners_for": ("home_last_corners_for", "away_last_corners_for"),
        "team_last_fouls_for": ("home_last_fouls_for", "away_last_fouls_for"),
        "team_rest_days": ("rest_days_home", "rest_days_away"),
    }
    for team_col, (home_col, away_col) in mapping.items():
        if home_col in row.index and team_col in ts.columns:
            row[home_col] = float(ts.loc[home, team_col])
        if away_col in row.index and team_col in ts.columns:
            row[away_col] = float(ts.loc[away, team_col])

    if "form_gd_diff" in row.index:
        row["form_gd_diff"] = row.get("home_last_gd", 0.0) - row.get("away_last_gd", 0.0)
    if "form_sot_for_diff" in row.index:
        row["form_sot_for_diff"] = row.get("home_last_sot_for", 0.0) - row.get("away_last_sot_for", 0.0)
    if "form_sot_against_diff" in row.index:
        row["form_sot_against_diff"] = row.get("home_last_sot_against", 0.0) - row.get("away_last_sot_against", 0.0)
    if "form_shots_for_diff" in row.index:
        row["form_shots_for_diff"] = row.get("home_last_shots_for", 0.0) - row.get("away_last_shots_for", 0.0)
    if "form_corners_for_diff" in row.index:
        row["form_corners_for_diff"] = row.get("home_last_corners_for", 0.0) - row.get("away_last_corners_for", 0.0)
    if "form_fouls_for_diff" in row.index:
        row["form_fouls_for_diff"] = row.get("home_last_fouls_for", 0.0) - row.get("away_last_fouls_for", 0.0)
    if "rest_days_diff" in row.index:
        row["rest_days_diff"] = row.get("rest_days_home", 0.0) - row.get("rest_days_away", 0.0)

    return pd.DataFrame([row], columns=feature_order)


def bootstrap_artifacts(seasons: Optional[List[str]] = None) -> Dict:
    selected_seasons = seasons or CONFIG.get("seasons")
    try:
        run_build_features(seasons=selected_seasons)
        acc, ll, meta_out = train_model()
        bundle = joblib.load(MODEL_PATH)
        bundle["meta"] = {**bundle.get("meta", {}), "seasons_used": selected_seasons}
        joblib.dump(bundle, MODEL_PATH)
        load_bundle.clear()
        load_team_state.clear()
        get_teams_for_season.clear()
        return {"ok": True, "accuracy": acc, "logloss": ll, "meta": meta_out, "error": None}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def ensure_runtime() -> Dict:
    try:
        pipe, feature_order, feature_means, meta = load_bundle()
        team_state, teams = load_team_state()
        return {
            "ready": True,
            "auto_rebuilt": False,
            "pipe": pipe,
            "feature_order": feature_order,
            "feature_means": feature_means,
            "meta": meta,
            "team_state": team_state,
            "teams": teams,
            "bootstrap_message": None,
        }
    except Exception as exc:
        if not raw_data_files():
            return {
                "ready": False,
                "auto_rebuilt": False,
                "error": f"{type(exc).__name__}: {exc}",
                "bootstrap_message": "Artifacts are missing and there are no raw CSV files available to rebuild from automatically.",
            }
        rebuilt = bootstrap_artifacts()
        if not rebuilt["ok"]:
            return {
                "ready": False,
                "auto_rebuilt": False,
                "error": rebuilt["error"],
                "bootstrap_message": f"Automatic rebuild failed after startup error: {type(exc).__name__}: {exc}",
            }
        pipe, feature_order, feature_means, meta = load_bundle()
        team_state, teams = load_team_state()
        return {
            "ready": True,
            "auto_rebuilt": True,
            "pipe": pipe,
            "feature_order": feature_order,
            "feature_means": feature_means,
            "meta": meta,
            "team_state": team_state,
            "teams": teams,
            "bootstrap_message": f"Artifacts were rebuilt automatically after startup error: {type(exc).__name__}: {exc}",
        }


runtime = ensure_runtime()
render_hero(runtime.get("meta", {}))

if runtime.get("auto_rebuilt"):
    st.success(runtime["bootstrap_message"])

if not runtime.get("ready"):
    status = artifact_status()
    render_metric_cards([
        ("Raw data", "Ready" if status["raw"] else "Missing", "CSV history available"),
        ("Features", "Ready" if status["features"] else "Missing", "Pre-match feature table"),
        ("Team state", "Ready" if status["team_state"] else "Missing", "Latest team snapshot"),
        ("Model", "Ready" if status["model"] else "Missing", "Serialized predictor bundle"),
    ])
    st.error(runtime.get("bootstrap_message", "The app could not initialize."))
    st.code(runtime.get("error", "Unknown startup error"))
    if st.button("Retry automatic setup", key="btn_retry_setup"):
        rebuilt = bootstrap_artifacts()
        if rebuilt["ok"]:
            st.success("Artifacts rebuilt successfully. Reloading app...")
            st.rerun()
        st.error(rebuilt["error"])
    st.stop()

pipe = runtime["pipe"]
feature_order = runtime["feature_order"]
feature_means = runtime["feature_means"]
meta = runtime["meta"]
ts = runtime["team_state"]
predict_season = meta.get("predict_season", "2526")
season_teams = get_teams_for_season(predict_season)
teams = [t for t in season_teams if t in ts.index]
if not teams:
    teams = runtime["teams"]
alias_lookup = build_alias_lookup(teams)

render_metric_cards([
    ("Accuracy", f"{meta.get('accuracy_time_split', 0.0)*100:.1f}%", "Time-split 83/17"),
    ("Precision", f"{meta.get('precision_time_split', 0.0)*100:.1f}%", "Macro avg"),
    ("Log loss", f"{meta.get('logloss_time_split', 0.0):.3f}", "Lower is better"),
    ("Teams", str(len(teams)), f"Active in {season_code_to_label(predict_season)}"),
])

tab_main, tab_about = st.tabs(["Train & Predict", "About"])

with tab_main:
    st.markdown('<p class="section-label">1. Training & data</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    status = artifact_status()
    render_metric_cards([
        ("Raw CSVs", str(len(raw_data_files())), "historical files"),
        ("Features", "✓" if status["features"] else "—", ""),
        ("Team state", "✓" if status["team_state"] else "—", ""),
        ("Model", "✓" if status["model"] else "—", ""),
    ])

    season_codes = available_season_codes()
    labels = [season_code_to_label(code) for code in season_codes]
    default_train = [l for l in labels if l not in ("25/26",)]  # Exclude future season
    selected_labels = st.multiselect(
        "Seasons to train on",
        options=labels,
        default=default_train,
        key="train_seasons",
        help="Select one or more seasons. Download will fetch missing CSVs from football-data.co.uk.",
    )
    selected_codes = [label_to_season_code(l) for l in selected_labels] or CONFIG.get("seasons", [])
    predict_label = st.selectbox("Prediction season", options=labels, index=labels.index("25/26") if "25/26" in labels else len(labels) - 1, key="predict_season")
    predict_code = label_to_season_code(predict_label)

    if st.button("Download missing data and retrain", key="btn_retrain", type="primary"):
        with st.status("Updating data and training model...", expanded=True) as status_box:
            for note in download_missing_seasons(selected_codes):
                status_box.write(note)
            status_box.write("Building features...")
            rebuilt = bootstrap_artifacts(selected_codes)
            if rebuilt["ok"]:
                bundle = joblib.load(MODEL_PATH)
                bundle["meta"] = {
                    **bundle.get("meta", {}),
                    "seasons_used": selected_codes,
                    "predict_season": predict_code,
                }
                joblib.dump(bundle, MODEL_PATH)
                load_bundle.clear()
                load_team_state.clear()
                get_teams_for_season.clear()
                prec = rebuilt.get("meta", {}).get("precision_time_split", 0)
                status_box.update(
                    label=f"Training complete. Acc={rebuilt['accuracy']*100:.1f}% Prec={prec*100:.1f}% LogLoss={rebuilt['logloss']:.3f}",
                    state="complete",
                )
                st.success("Artifacts refreshed. Reloading app...")
                st.rerun()
            else:
                status_box.update(label="Training failed", state="error")
                st.error(rebuilt["error"])
    st.markdown('</div>', unsafe_allow_html=True)

    if meta:
        st.markdown('<p class="section-label" style="margin-top:0.5rem;">Model info</p>', unsafe_allow_html=True)
        metadata_rows = [
            ("Accuracy", f"{meta.get('accuracy_time_split', 0)*100:.1f}%" if meta.get("accuracy_time_split") is not None else "Not recorded"),
            ("Precision", f"{meta.get('precision_time_split', 0)*100:.1f}%" if meta.get("precision_time_split") is not None else "Not recorded"),
            ("Log loss", meta.get("logloss_time_split")),
            ("Train rows", meta.get("n_train")),
            ("Test rows", meta.get("n_test")),
            ("Seasons used", ", ".join(season_code_to_label(s) for s in (meta.get("seasons_used") or [])) if meta.get("seasons_used") else "Not recorded"),
            ("Prediction season", season_code_to_label(meta.get("predict_season", "2526")) if meta.get("predict_season") else "Not recorded"),
        ]
        st.dataframe(pd.DataFrame(metadata_rows, columns=["Field", "Value"]), use_container_width=True, hide_index=True)

    st.markdown(f'<p class="section-label" style="margin-top:1.25rem;">2. Predict · {season_code_to_label(predict_season)}</p>', unsafe_allow_html=True)
    # Default to El Clásico: Barcelona vs Real Madrid
    home_idx = teams.index("Barcelona") if "Barcelona" in teams else 0
    col_home, col_vs, col_away = st.columns([2, 0.4, 2], vertical_alignment="center")
    with col_home:
        home = st.selectbox(
            "Home",
            teams,
            index=home_idx,
            key="home_team_ui",
            label_visibility="collapsed",
            format_func=lambda t: get_display_name(t),
        )
    with col_vs:
        st.markdown('<div style="display:flex;align-items:center;justify-content:center;font-weight:700;color:#718096;">VS</div>', unsafe_allow_html=True)
    with col_away:
        away_options = [t for t in teams if t != home]
        away_idx = away_options.index("Real Madrid") if "Real Madrid" in away_options else 0
        away = st.selectbox(
            "Away",
            away_options if away_options else teams,
            index=away_idx,
            key="away_team_ui",
            label_visibility="collapsed",
            format_func=lambda t: get_display_name(t),
        )

    # La Liga match format: Team Name | Shield | VS | Shield | Team Name — absolutely centered
    home_crest = get_crest_url(home)
    away_crest = get_crest_url(away)
    home_display = get_display_name(home)
    away_display = get_display_name(away)
    crest_style = "height:40px;width:40px;object-fit:contain;vertical-align:middle;"
    home_img = f'<img src="{home_crest}" alt="" style="{crest_style}" onerror="this.style.display=\'none\'"/>' if home_crest else ""
    away_img = f'<img src="{away_crest}" alt="" style="{crest_style}" onerror="this.style.display=\'none\'"/>' if away_crest else ""
    st.markdown(
        f"""
        <div class="match-display" style="margin:1rem 0;">
          <div class="match-display-inner">
            <div class="match-left">
              <span style="font-weight:700;font-size:1.35rem;color:#e2e8f0;">{home_display}</span>
              <span>{home_img}</span>
            </div>
            <span class="match-vs" style="font-size:1rem;">VS</span>
            <div class="match-right">
              <span>{away_img}</span>
              <span style="font-weight:700;font-size:1.35rem;color:#e2e8f0;">{away_display}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate prediction button — bottom right
    btn_spacer, btn_col = st.columns([3, 1])
    with btn_col:
        predict_clicked = st.button("Generate prediction", key="btn_predict_main", type="primary", use_container_width=True)

    with st.expander("Blend with bookmaker odds (optional)"):
        use_market = st.toggle("Blend live bookmaker odds", value=False, help="Weight the model with normalized 1X2 market odds.")
        market_weight = 0.25
        odds_home = odds_draw = odds_away = None
        if use_market:
            market_weight = st.slider("Market weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            if "odds_values" not in st.session_state:
                st.session_state["odds_values"] = DEFAULT_ODDS.copy()
            if st.button("Fetch live odds", key="btn_fetch_odds"):
                live_odds, reason = fetch_live_odds(home, away, get_odds_api_key(), alias_lookup)
                if live_odds:
                    oh, od, oa, source = live_odds
                    st.session_state["odds_values"] = {"H": oh, "D": od, "A": oa, "src": source}
                    st.success(f"Loaded from {source}.")
                else:
                    st.warning(reason or "Could not fetch odds.")
            ov = st.session_state["odds_values"]
            c1, c2, c3 = st.columns(3)
            odds_home = c1.number_input("Home", min_value=1.01, value=float(ov["H"]), step=0.01, format="%.2f")
            odds_draw = c2.number_input("Draw", min_value=1.01, value=float(ov["D"]), step=0.01, format="%.2f")
            odds_away = c3.number_input("Away", min_value=1.01, value=float(ov["A"]), step=0.01, format="%.2f")

    if home and away and not predict_clicked:
        with st.expander("Team comparison"):
            team_compare = pd.DataFrame({
                "Metric": ["Elo", "Goals for (last 5)", "Goal diff (last 5)", "Rest days"],
                get_display_name(home): [round(float(ts.loc[home, "elo"]), 1), round(float(ts.loc[home, "team_last_gf"]), 2), round(float(ts.loc[home, "team_last_gd"]), 2), round(float(ts.loc[home, "team_rest_days"]), 1)],
                get_display_name(away): [round(float(ts.loc[away, "elo"]), 1), round(float(ts.loc[away, "team_last_gf"]), 2), round(float(ts.loc[away, "team_last_gd"]), 2), round(float(ts.loc[away, "team_rest_days"]), 1)],
            })
            st.dataframe(team_compare, use_container_width=True, hide_index=True)
    if predict_clicked:
        try:
            match_vector = vector_for_match(home, away, feature_order, ts, feature_means)
            model_probs = pipe.predict_proba(match_vector)[0].tolist()
            final_probs = model_probs[:]
            result_note = "Model only"
            if use_market and odds_home and odds_draw and odds_away:
                market_probs, _ = normalize_probs_from_odds(odds_home, odds_draw, odds_away)
                blended = (1 - market_weight) * np.array(model_probs) + market_weight * np.array(market_probs)
                final_probs = (blended / blended.sum()).tolist()
                result_note = f"Blended with market weight {market_weight*100:.0f}%"

            winner_idx = int(np.argmax(final_probs))
            label_txt = ["Home win", "Draw", "Away win"][winner_idx]
            st.markdown(f"**Prediction: {label_txt}** ({result_note})")
            fair_lines = fair_odds(final_probs)
            render_probability_cards(final_probs, fair_lines, winner_idx)

            details_df = pd.DataFrame(
                {
                    "Source": ["Model", "Final output"],
                    "H": [f"{model_probs[0]*100:.1f}%", f"{final_probs[0]*100:.1f}%"],
                    "D": [f"{model_probs[1]*100:.1f}%", f"{final_probs[1]*100:.1f}%"],
                    "A": [f"{model_probs[2]*100:.1f}%", f"{final_probs[2]*100:.1f}%"],
                }
            )
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

with tab_about:
    st.markdown('<p class="section-label">About</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("**LaLiga Match Predictor.** End-to-end H/D/A outcome predictor with optional blend of real bookmaker odds.")
    st.markdown("**Live app:** [laliga-predict-sobreviela.streamlit.app](https://laliga-predict-sobreviela.streamlit.app/)")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
    - **Model:** HistGradientBoostingClassifier ensemble (5 seeds averaged) with sample weights for class balance
    - **Target:** H/D/A (Home, Draw, Away)
    - **Features:** Elo ratings (home, away, diff), rolling form over last 10 matches (goals for/against, shots on target, corners, fouls), rest days (home, away, diff), and engineered diffs
    - **Data:** LaLiga SP1 CSVs from football-data.co.uk, stored as Parquet
    - **Split:** Time-based 83/17: train on first 83% of matches by date, evaluate on last 17%. No future leakage.
    """)
    st.markdown("**Accuracy.** Computed in train_model.py via chronological 83/17 time-split. The model trains on the first 83% of matches (by date) and is evaluated on the held-out 17%.")
    st.markdown("**Odds.** Toggle \"Blend live bookmaker odds\" to use The Odds API (1X2). Aliases match team variants (e.g. FC Barcelona and Barcelona).")
    st.markdown("**Seasons (new).** Select 09/10 to 25/26. Missing CSVs auto-download from football-data.co.uk.")
    st.markdown("**La Liga branding (new).** Official logo, team crests, and display names with accents (Atlético de Madrid, Deportivo Alavés).")
    st.markdown("---")
    st.markdown("**Credits:** football-data.co.uk, The Odds API, scikit-learn, Streamlit")
    st.markdown("*Predictions are not financial advice.*")
    st.markdown('</div>', unsafe_allow_html=True)
