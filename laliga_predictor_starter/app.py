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
from src.train_model import train as train_model

DATA_RAW = Path(CONFIG["raw_data_path"])
DATA_PROC = Path(CONFIG["processed_data_path"])
MODEL_PATH = Path(CONFIG["model_path"])
FEATURES_PATH = DATA_PROC / "features.parquet"
TEAM_STATE_PATH = DATA_PROC / "team_state.parquet"

st.set_page_config(
    page_title="LaLiga Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1220px;}
.hero-shell {
  background:
    radial-gradient(circle at top left, rgba(59,130,246,.22), transparent 30%),
    radial-gradient(circle at top right, rgba(16,185,129,.18), transparent 28%),
    linear-gradient(135deg, rgba(15,23,42,.95), rgba(17,24,39,.84));
  border: 1px solid rgba(148,163,184,.16);
  border-radius: 24px;
  padding: 1.4rem 1.5rem 1.3rem 1.5rem;
  box-shadow: 0 18px 50px rgba(15,23,42,.30);
  margin-bottom: 1rem;
}
.hero-title {
  font-size: clamp(2rem, 3.6vw, 3.2rem);
  line-height: 1.05;
  font-weight: 800;
  margin: 0;
  background: linear-gradient(90deg, #f8fafc 0%, #93c5fd 55%, #6ee7b7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-subtitle {color: #cbd5e1; margin-top: .45rem; font-size: 1rem;}
.hero-badges {display:flex; gap:.55rem; flex-wrap:wrap; margin-top: .85rem;}
.hero-badge {
  display:inline-block;
  padding:.35rem .7rem;
  border-radius:999px;
  font-size:.82rem;
  color:#e2e8f0;
  background:rgba(15,23,42,.68);
  border:1px solid rgba(148,163,184,.18);
}
.glass-card {
  background: linear-gradient(180deg, rgba(15,23,42,.86), rgba(15,23,42,.72));
  border: 1px solid rgba(148,163,184,.14);
  border-radius: 22px;
  padding: 1.05rem 1.1rem;
  box-shadow: 0 12px 34px rgba(15,23,42,.20);
}
.glass-card h3 {
  margin: 0 0 .8rem 0;
  color: #f8fafc;
  font-size: 1.05rem;
}
.metric-strip {
  display:grid;
  grid-template-columns:repeat(4, minmax(0, 1fr));
  gap:.8rem;
  margin: 1rem 0 1.1rem 0;
}
.metric-card {
  background: rgba(15,23,42,.72);
  border: 1px solid rgba(148,163,184,.14);
  border-radius: 18px;
  padding: .95rem 1rem;
}
.metric-label {color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em;}
.metric-value {color:#f8fafc; font-size:1.45rem; font-weight:700; margin-top:.2rem;}
.metric-help {color:#cbd5e1; font-size:.82rem; margin-top:.2rem;}
.prob-card {
  background: linear-gradient(180deg, rgba(30,41,59,.88), rgba(15,23,42,.88));
  border: 1px solid rgba(96,165,250,.16);
  border-radius: 18px;
  padding: .95rem 1rem;
}
.prob-label {color:#cbd5e1; font-size:.9rem; margin-bottom:.35rem;}
.prob-value {color:#f8fafc; font-size:1.7rem; font-weight:800;}
.prob-note {color:#94a3b8; font-size:.82rem;}
.result-banner {
  padding: .9rem 1rem;
  border-radius: 18px;
  color: #f8fafc;
  background: linear-gradient(90deg, rgba(37,99,235,.92), rgba(16,185,129,.82));
  margin-bottom: .9rem;
  font-weight: 700;
}
.pill {
  display:inline-block;
  padding:.28rem .66rem;
  border-radius:999px;
  font-size:.78rem;
  border:1px solid rgba(148,163,184,.18);
  background:rgba(15,23,42,.72);
  color:#e2e8f0;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 12px;
  border: 0;
  padding: .6rem 1rem;
  background: linear-gradient(90deg, #2563eb, #0f766e);
  color: white;
}
.stButton > button:hover, .stDownloadButton > button:hover {filter: brightness(1.06);}
.stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input {
  border-radius: 12px !important;
}
@media (max-width: 900px) {
  .metric-strip {grid-template-columns:repeat(2, minmax(0, 1fr));}
}
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
    return ["2122", "2223", "2324", "2425", "2526"]


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


def render_hero(meta: Dict) -> None:
    season_text = ", ".join(meta.get("seasons_used", [])) if meta.get("seasons_used") else "Local historical seasons"
    st.markdown(
        f"""
        <div class="hero-shell">
          <p class="hero-title">LaLiga Match Predictor</p>
          <div class="hero-subtitle">
            Modernized Streamlit dashboard for match probabilities, team form snapshots, and optional bookmaker blending.
          </div>
          <div class="hero-badges">
            <span class="hero-badge">Model: logistic regression</span>
            <span class="hero-badge">Data: football-data.co.uk + team-state features</span>
            <span class="hero-badge">Training seasons: {season_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: List[Tuple[str, str, str]]) -> None:
    cards = []
    for label, value, help_text in metrics:
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-help">{help_text}</div>
            </div>
            """
        )
    st.markdown(f'<div class="metric-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_probability_cards(probabilities: List[float], fair_lines: List[float]) -> None:
    labels = [("Home win", probabilities[0], fair_lines[0]), ("Draw", probabilities[1], fair_lines[1]), ("Away win", probabilities[2], fair_lines[2])]
    cols = st.columns(3)
    for col, (label, prob, fair_line) in zip(cols, labels):
        col.markdown(
            f"""
            <div class="prob-card">
              <div class="prob-label">{label}</div>
              <div class="prob-value">{prob * 100:.1f}%</div>
              <div class="prob-note">Fair odds: {fair_line:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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


def get_odds_api_key() -> str:
    api_key = st.session_state.get("odds_api_key", "")
    if not api_key:
        try:
            api_key = st.secrets.get("odds_api_key", "")
        except Exception:
            api_key = ""
    if not api_key:
        api_key = os.environ.get("ODDS_API_KEY", "")
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
    return (
        bundle["pipeline"],
        bundle["feature_order"],
        bundle.get("feature_means", {}),
        bundle.get("meta", {}),
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
teams = runtime["teams"]
alias_lookup = build_alias_lookup(teams)

render_metric_cards([
    ("Accuracy", f"{meta.get('accuracy_time_split', 0.0):.3f}", "Time-based validation"),
    ("Log loss", f"{meta.get('logloss_time_split', 0.0):.3f}", "Lower is better"),
    ("Teams", str(len(teams)), "Current team-state entries"),
    ("Features", str(len(feature_order)), "Numeric model inputs"),
])

tab_predict, tab_train, tab_about = st.tabs(["Predict", "Train / Data", "About"])

with tab_predict:
    selector_col, market_col = st.columns([1.15, 0.85], gap="large")

    with selector_col:
        st.markdown('<div class="glass-card"><h3>Match setup</h3></div>', unsafe_allow_html=True)
        home = st.selectbox("Home team", teams, index=0, key="home_team_ui")
        away_options = [team for team in teams if team != home]
        away_default = 0 if away_options else None
        away = st.selectbox("Away team", away_options, index=away_default, key="away_team_ui")

        if home and away:
            team_compare = pd.DataFrame({
                "Metric": ["Elo", "Recent goals for", "Recent goal diff", "Rest days"],
                home: [
                    round(float(ts.loc[home, "elo"]), 1),
                    round(float(ts.loc[home, "team_last_gf"]), 2),
                    round(float(ts.loc[home, "team_last_gd"]), 2),
                    round(float(ts.loc[home, "team_rest_days"]), 1),
                ],
                away: [
                    round(float(ts.loc[away, "elo"]), 1),
                    round(float(ts.loc[away, "team_last_gf"]), 2),
                    round(float(ts.loc[away, "team_last_gd"]), 2),
                    round(float(ts.loc[away, "team_rest_days"]), 1),
                ],
            })
            st.dataframe(team_compare, use_container_width=True, hide_index=True)

    with market_col:
        st.markdown('<div class="glass-card"><h3>Market blend</h3></div>', unsafe_allow_html=True)
        use_market = st.toggle("Blend live bookmaker odds", value=False, help="Weight the model with normalized 1X2 market odds.")
        market_weight = 0.25
        if use_market:
            market_weight = st.slider("Market weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            api_key_ui = st.text_input(
                "Odds API key",
                value=st.session_state.get("odds_api_key", ""),
                type="password",
                help="Stored only in this local Streamlit session.",
            )
            if api_key_ui:
                st.session_state["odds_api_key"] = api_key_ui
            if "odds_values" not in st.session_state:
                st.session_state["odds_values"] = DEFAULT_ODDS.copy()

            fetch_col, _ = st.columns([1, 1])
            with fetch_col:
                if st.button("Fetch live odds", key="btn_fetch_odds"):
                    live_odds, reason = fetch_live_odds(home, away, get_odds_api_key(), alias_lookup)
                    if live_odds:
                        oh, od, oa, source = live_odds
                        st.session_state["odds_values"] = {"H": oh, "D": od, "A": oa, "src": source}
                        st.success(f"Loaded odds from {source}.")
                    else:
                        st.warning(reason or "Could not fetch odds.")

            odds_values = st.session_state["odds_values"]
            col_h, col_d, col_a = st.columns(3)
            odds_home = col_h.number_input("Home", min_value=1.01, value=float(odds_values["H"]), step=0.01, format="%.2f")
            odds_draw = col_d.number_input("Draw", min_value=1.01, value=float(odds_values["D"]), step=0.01, format="%.2f")
            odds_away = col_a.number_input("Away", min_value=1.01, value=float(odds_values["A"]), step=0.01, format="%.2f")
            if odds_values.get("src"):
                st.caption(f"Source: {odds_values['src']}")

            with st.expander("Upcoming fixtures from Odds API"):
                events, error_message = list_upcoming_events(get_odds_api_key())
                if error_message:
                    st.write(error_message)
                elif not events:
                    st.write("No events returned.")
                else:
                    event_rows = [
                        {"Home": event.get("home_team", ""), "Away": event.get("away_team", "")}
                        for event in events
                    ]
                    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
        else:
            odds_home = odds_draw = odds_away = None
            st.caption("Use the toggle above if you want to blend the model with current market prices.")

    predict_clicked = st.button("Generate prediction", key="btn_predict_main")
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
                result_note = f"Blended with market weight {market_weight:.2f}"

            predicted_label = ["H", "D", "A"][int(np.argmax(final_probs))]
            st.markdown(
                f'<div class="result-banner">Prediction for {home} vs {away}: {predicted_label} <span class="pill">{result_note}</span></div>',
                unsafe_allow_html=True,
            )
            fair_lines = fair_odds(final_probs)
            render_probability_cards(final_probs, fair_lines)

            chart_df = pd.DataFrame(
                {"Outcome": ["Home", "Draw", "Away"], "Probability": final_probs}
            ).set_index("Outcome")
            st.bar_chart(chart_df)

            details_df = pd.DataFrame(
                {
                    "Source": ["Model", "Final output"],
                    "H": [round(model_probs[0], 3), round(final_probs[0], 3)],
                    "D": [round(model_probs[1], 3), round(final_probs[1], 3)],
                    "A": [round(model_probs[2], 3), round(final_probs[2], 3)],
                }
            )
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

with tab_train:
    st.markdown('<div class="glass-card"><h3>Dataset and training controls</h3></div>', unsafe_allow_html=True)
    status = artifact_status()
    render_metric_cards([
        ("Raw CSVs", str(len(raw_data_files())), "Available historical files"),
        ("Features parquet", "Yes" if status["features"] else "No", str(FEATURES_PATH)),
        ("Team state", "Yes" if status["team_state"] else "No", str(TEAM_STATE_PATH)),
        ("Model bundle", "Yes" if status["model"] else "No", str(MODEL_PATH)),
    ])

    season_codes = available_season_codes()
    labels = [season_code_to_label(code) for code in season_codes]
    default_last_n = min(3, len(season_codes))
    default_idx = list(range(len(season_codes) - default_last_n, len(season_codes)))
    selected_labels = st.multiselect(
        "Training seasons",
        options=labels,
        default=[labels[i] for i in default_idx],
        key="train_seasons",
    )
    selected_codes = [label_to_season_code(label) for label in selected_labels] or CONFIG.get("seasons", [])
    predict_label = st.selectbox(
        "Prediction season label",
        options=labels,
        index=labels.index("25/26") if "25/26" in labels else len(labels) - 1,
        key="predict_season",
    )
    predict_code = label_to_season_code(predict_label)

    if st.button("Download missing data and retrain", key="btn_retrain"):
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
                status_box.update(
                    label=f"Training complete. Accuracy={rebuilt['accuracy']:.3f}, LogLoss={rebuilt['logloss']:.3f}",
                    state="complete",
                )
                st.success("Artifacts refreshed. Reloading app...")
                st.rerun()
            else:
                status_box.update(label="Training failed", state="error")
                st.error(rebuilt["error"])

    if meta:
        st.subheader("Current model metadata")
        metadata_rows = [
            ("Accuracy", meta.get("accuracy_time_split")),
            ("Log loss", meta.get("logloss_time_split")),
            ("Train rows", meta.get("n_train")),
            ("Test rows", meta.get("n_test")),
            ("Seasons used", ", ".join(meta.get("seasons_used", [])) if meta.get("seasons_used") else "Not recorded"),
            ("Prediction season", meta.get("predict_season", "Not recorded")),
        ]
        st.dataframe(pd.DataFrame(metadata_rows, columns=["Field", "Value"]), use_container_width=True, hide_index=True)

with tab_about:
    st.markdown('<div class="glass-card"><h3>What changed</h3></div>', unsafe_allow_html=True)
    st.write(
        """
        This version focuses on three things:

        - a more polished dashboard-style UI,
        - safer startup behavior when artifacts are missing or stale,
        - a cleaner training path that is more resilient to dependency updates.
        """
    )
    st.write(
        """
        The predictor still uses a lightweight logistic regression model over rolling team form, rest days, and Elo-style strength ratings.
        If you provide live 1X2 odds, the app can blend them with the model output to produce a more market-aware probability view.
        """
    )
