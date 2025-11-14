# app.py — LaLiga 25/26 Match Predictor (full app, robust class-order handling)

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.config import CONFIG
import src.features as F               # we'll call F.build_features via a helper below
from src.train_model import train as train_model

DATA_RAW = Path(CONFIG["raw_data_path"])
DATA_PROC = Path(CONFIG["processed_data_path"])

# --------------------------- Page chrome & CSS ---------------------------
st.set_page_config(
    page_title="LaLiga 25/26 Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
/* Hide Streamlit chrome we don't need */
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}

/* Container padding */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

/* Gradient hero title */
.hero h1 {
  font-size: clamp(1.8rem, 3.4vw, 3rem);
  line-height: 1.1;
  background: linear-gradient(90deg,#a78bfa 0%, #60a5fa 50%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: .25rem;
}
.hero .sub {
  color:#9ca3af; margin-top:0.25rem; font-size: .95rem;
}

/* Card */
.card {
  background: radial-gradient(1200px circle at 10% 10%, rgba(124,58,237,.12), transparent 40%),
              rgba(17,24,39,.65);
  border: 1px solid rgba(148,163,184,.12);
  border-radius: 16px;
  padding: 1rem 1rem 1.1rem;
  box-shadow: 0 10px 26px rgba(0,0,0,.22);
}

/* Section titles inside cards */
.card h3 { margin: 0 0 .75rem 0; font-size: 1.05rem; color:#e5e7eb; }

/* Inputs: slightly rounder */
.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
  border-radius: 10px !important;
}

/* Buttons (distinct keys used in code) */
.stButton>button {
  background: linear-gradient(90deg,#7c3aed,#2563eb);
  color: white; border: 0; border-radius: 12px; padding: .55rem 1rem;
}
.stButton>button:hover { filter: brightness(1.08); }

/* Pills */
.pill {
  display:inline-block; padding:.25rem .6rem; border-radius:999px;
  background:#1f2937; color:#cbd5e1; font-size:.8rem; border:1px solid #374151;
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# --------------------------- Odds API (v4) ---------------------------
ODDS_SPORT_KEY  = "soccer_spain_la_liga"
ODDS_EVENTS     = "https://api.the-odds-api.com/v4/sports/{sport}/events"
ODDS_EVENT_ODDS = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
H2H_MARKET      = "h2h"

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

# Alias list — extend as needed (covers common shapes seen in odds feeds)
STATIC_ALIAS_EXTRAS: Dict[str, List[str]] = {
    "Barcelona": ["FC Barcelona", "Barca", "FCBarcelona", "Barcelona CF"],
    "Real Madrid": ["Real Madrid CF", "RealMadrid"],
    "Atletico Madrid": ["Atlético Madrid", "Atletico de Madrid", "Atlético de Madrid"],
    "Athletic Club": ["Athletic Bilbao", "Athletic de Bilbao", "Athletic"],
    "Real Sociedad": ["Sociedad", "RealSociedad"],
    "Celta Vigo": ["RC Celta", "Celta de Vigo"],
    "Real Betis": ["Real Betis Balompie", "Real Betis Balompié", "Betis"],
    "Villarreal": ["Villarreal CF"],
    "Osasuna": ["CA Osasuna"],
    "Alaves": ["Deportivo Alaves", "Deportivo Alavés", "Alavés"],
    "Almeria": ["UD Almeria", "UD Almería", "Almería"],
    "Getafe": ["Getafe CF"],
    "Girona": ["Girona FC"],
    "Mallorca": ["RCD Mallorca"],
    "Las Palmas": ["UD Las Palmas"],
    "Leganes": ["CD Leganes", "CD Leganés", "Leganés"],
    "Cadiz": ["Cadiz CF", "Cádiz", "Cádiz CF"],
    "Granada": ["Granada CF", "Granada CF SAD", "Granada CF S.A.D."],
    "Rayo Vallecano": ["Vallecano", "Rayo"],
    "Elche": ["Elche CF"],
    "Sevilla": ["Sevilla FC"],
    "Espanyol": ["RCD Espanyol"],
    "Levante": ["Levante UD"],
}
PREFIXES = ("fc ", "cf ", "cd ", "ud ", "rcd ", "rc ", "ca ", "real ", "club ")

def build_alias_lookup(teams: List[str]) -> Dict[str, Set[str]]:
    lookup: Dict[str, Set[str]] = {}
    for t in teams:
        base = _norm(t)
        alts: Set[str] = {base}
        t_low = t.lower()
        for p in PREFIXES:
            if t_low.startswith(p):
                alts.add(_norm(t[len(p):]))
        if t in STATIC_ALIAS_EXTRAS:
            for v in STATIC_ALIAS_EXTRAS[t]:
                alts.add(_norm(v))
        for canon, variants in STATIC_ALIAS_EXTRAS.items():
            if _norm(canon) == base:
                for v in variants:
                    alts.add(_norm(v))
        lookup[base] = alts
    return lookup

def is_alias_match(teamA: str, teamB: str, alias_lookup: Dict[str, Set[str]]) -> bool:
    a = _norm(teamA); b = _norm(teamB)
    if a in b or b in a:
        return True
    a_set = alias_lookup.get(a, {a})
    b_set = alias_lookup.get(b, {b})
    return (a in b_set) or (b in a_set) or bool(a_set.intersection(b_set))

def get_odds_api_key() -> str:
    k = st.session_state.get("odds_api_key", "")
    if not k:
        try:
            k = st.secrets.get("odds_api_key", "")
        except Exception:
            k = ""
    if not k:
        k = os.environ.get("ODDS_API_KEY", "")
    return k

def _show_quota(headers):
    rem = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    if rem or used:
        st.caption(f"Odds API quota → remaining: {rem}, used: {used}")

@st.cache_data(ttl=60)
def list_upcoming_events(api_key: str, days_from: int = 60):
    if not api_key:
        return [], "No API key"
    try:
        r = requests.get(
            ODDS_EVENTS.format(sport=ODDS_SPORT_KEY),
            params={"apiKey": api_key, "daysFrom": str(days_from)},
            timeout=15
        )
        _show_quota(r.headers)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return [], f"Unexpected response: {data}"
        return data, None
    except Exception as e:
        return [], f"Events request failed: {e}"

def _coerce_bookmakers_payload(payload):
    """
    Event odds payload can be:
      - list[bookmaker]               (classic)
      - dict{..., bookmakers: list}   (alternate)
    Normalize to list[bookmaker] or return (None, reason).
    """
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        if "bookmakers" in payload and isinstance(payload["bookmakers"], list):
            return payload["bookmakers"], None
        return None, f"Unexpected dict payload (no 'bookmakers'): {payload}"
    return None, f"Unexpected payload type: {type(payload).__name__}"

def fetch_live_odds(home: str, away: str, api_key: str, alias_lookup: Dict[str, Set[str]], days_from: int = 60
                   ) -> Tuple[Optional[Tuple[float,float,float,str]], Optional[str]]:
    if not api_key:
        return None, "No API key"

    # 1) events
    try:
        r = requests.get(
            ODDS_EVENTS.format(sport=ODDS_SPORT_KEY),
            params={"apiKey": api_key, "daysFrom": str(days_from)},
            timeout=15
        )
        _show_quota(r.headers)
        r.raise_for_status()
        events = r.json()
        if not isinstance(events, list):
            return None, f"Unexpected response from events: {events}"
    except Exception as e:
        return None, f"Events request failed: {e}"

    # 2) match event id
    ev_id = None
    for ev in events:
        h = ev.get("home_team", "")
        a = ev.get("away_team", "")
        if is_alias_match(h, home, alias_lookup) and is_alias_match(a, away, alias_lookup):
            ev_id = ev.get("id")
            break
    if not ev_id:
        return None, "Fixture not found in upcoming events window"

    # 3) event odds
    try:
        r2 = requests.get(
            ODDS_EVENT_ODDS.format(sport=ODDS_SPORT_KEY, event_id=ev_id),
            params={"apiKey": api_key, "regions": "eu,uk,us", "markets": H2H_MARKET, "oddsFormat": "decimal"},
            timeout=15
        )
        _show_quota(r2.headers)
        r2.raise_for_status()
        payload = r2.json()
    except Exception as e:
        return None, f"Event odds request failed: {e}"

    books, reason = _coerce_bookmakers_payload(payload)
    if books is None:
        return None, f"Unexpected response from event odds: {payload}"

    for bm in books:
        if not isinstance(bm, dict):
            continue
        for m in (bm.get("markets") or []):
            if isinstance(m, dict) and m.get("key") == H2H_MARKET:
                oh = od = oa = None
                for o in (m.get("outcomes") or []):
                    if not isinstance(o, dict):
                        continue
                    name_raw = o.get("name")
                    price = o.get("price")
                    if price is None:
                        continue
                    if _norm(name_raw) in ("home", _norm(home)) or is_alias_match(name_raw, home, alias_lookup):
                        oh = float(price)
                    elif _norm(name_raw) in ("away", _norm(away)) or is_alias_match(name_raw, away, alias_lookup):
                        oa = float(price)
                    elif _norm(name_raw) in ("draw", "tie"):
                        od = float(price)
                if oh and od and oa:
                    return (oh, od, oa, bm.get("title", "bookmaker")), None

    return None, "No complete h2h market from bookmakers"

# --------------------------- Model helpers ---------------------------
def season_code_to_label(code: str) -> str:
    return f"{code[:2]}/{code[2:]}"

def label_to_season_code(label: str) -> str:
    return label.replace("/", "")

def available_season_codes() -> List[str]:
    # include 25/26 label for UI; downloader will skip if missing
    return ["2122", "2223", "2324", "2425", "2526"]

def fair_odds(probs):
    return [(1/p if p > 0 else float("inf")) for p in probs]

@st.cache_resource
def load_bundle():
    bundle = joblib.load(CONFIG["model_path"])
    return (
        bundle["pipeline"],
        bundle["feature_order"],
        bundle.get("feature_means", {}),
        bundle.get("meta", {}),
    )

@st.cache_data
def load_team_state():
    fp = DATA_PROC / "team_state.parquet"
    if not fp.exists():
        st.error("team_state.parquet not found. Click **Train / Data → Update dataset & retrain** first.")
        st.stop()
    ts = pd.read_parquet(fp).set_index("Team")
    teams = sorted(ts.index.tolist())
    return ts, teams

def vector_for_match(home: str, away: str, feature_order, ts, feature_means: dict):
    row = pd.Series({c: float(feature_means.get(c, 0.0)) for c in feature_order}, dtype=float)

    if home not in ts.index or away not in ts.index:
        raise ValueError("Team not found in team_state.parquet. Check team names.")

    # Elo
    elo_home = float(ts.loc[home, "elo"])
    elo_away = float(ts.loc[away, "elo"])
    if "elo_home" in row.index: row["elo_home"] = elo_home
    if "elo_away" in row.index: row["elo_away"] = elo_away
    if "elo_diff" in row.index: row["elo_diff"] = elo_home - elo_away

    # Recent form from team_state → feature row
    mapping = {
        "team_last_gf": ("home_last_gf", "away_last_gf"),
        "team_last_ga": ("home_last_ga", "away_last_ga"),
        "team_last_gd": ("home_last_gd", "away_last_gd"),
        "team_last_sot_for": ("home_last_sot_for", "away_last_sot_for"),
        "team_last_sot_against": ("home_last_sot_against", "away_last_sot_against"),
        "team_rest_days": ("rest_days_home", "rest_days_away"),
    }

    for tcol, (hcol, acol) in mapping.items():
        if hcol in row.index: row[hcol] = float(ts.loc[home, tcol])
        if acol in row.index: row[acol] = float(ts.loc[away, tcol])

    # Diffs if present
    if "form_gd_diff" in row.index:
        row["form_gd_diff"] = row.get("home_last_gd", 0.0) - row.get("away_last_gd", 0.0)
    if "form_sot_for_diff" in row.index:
        row["form_sot_for_diff"] = row.get("home_last_sot_for", 0.0) - row.get("away_last_sot_for", 0.0)
    if "form_sot_against_diff" in row.index:
        row["form_sot_against_diff"] = row.get("home_last_sot_against", 0.0) - row.get("away_last_sot_against", 0.0)
    if "rest_days_diff" in row.index:
        row["rest_days_diff"] = row.get("rest_days_home", 0.0) - row.get("rest_days_away", 0.0)

    return pd.DataFrame([row.values], columns=feature_order)

def normalize_probs_from_odds(oh: float, od: float, oa: float):
    raw = np.array([1/oh, 1/od, 1/oa], dtype=float)
    s = raw.sum()
    probs = raw / s if s > 0 else raw
    return probs.tolist(), s

# ---------- Robust class-order handling for predict_proba ----------
def _final_estimator(pipe):
    """Return the last step (estimator) from a scikit-learn Pipeline, or pipe if not a pipeline."""
    try:
        return pipe[-1]
    except Exception:
        try:
            return pipe.steps[-1][1]
        except Exception:
            return pipe

def _get_classes(pipe):
    """Find the learned classes_ from the final estimator or the pipeline itself."""
    est = _final_estimator(pipe)
    classes = getattr(est, "classes_", None)
    if classes is None:
        classes = getattr(pipe, "classes_", None)
    return classes

def predict_proba_HDA(pipe, Xrow) -> List[float]:
    """
    Return probabilities in [H, D, A] order, robust to either numeric classes (0,1,2)
    or label classes ('H','D','A'), regardless of how the model was fit.
    """
    proba = pipe.predict_proba(Xrow)[0]
    classes = _get_classes(pipe)

    # If the model doesn't expose classes_, assume proba already in H,D,A
    if classes is None:
        p = np.array(proba, dtype=float)
        if p.shape[0] == 3:
            s = p.sum()
            return (p / s).tolist() if s > 0 else p.tolist()
        raise ValueError("Model does not expose classes_ and predict_proba shape != 3.")

    # Build index map for H/D/A, accepting both numeric and string classes
    idxH = idxD = idxA = None
    for i, c in enumerate(list(classes)):
        if isinstance(c, str):
            token = c.strip().lower()
            if token in ("h", "home", "0"): idxH = i
            elif token in ("d", "draw", "tie", "1"): idxD = i
            elif token in ("a", "away", "2"): idxA = i
        else:
            if c == 0: idxH = i
            elif c == 1: idxD = i
            elif c == 2: idxA = i

    if any(v is None for v in (idxH, idxD, idxA)):
        if len(classes) == 3:
            idxH, idxD, idxA = 0, 1, 2
        else:
            raise ValueError(f"Unsupported class set: {classes!r}")

    pH, pD, pA = float(proba[idxH]), float(proba[idxD]), float(proba[idxA])
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH / s, pD / s, pA / s
    return [pH, pD, pA]

# Back-compat helper: call features.build_features from the app
def run_build_features(seasons: Optional[List[str]] = None):
    if hasattr(F, "build_features"):
        return F.build_features(seasons=seasons)
    if hasattr(F, "run_build_features"):
        return F.run_build_features(seasons=seasons)
    raise AttributeError("Neither build_features nor run_build_features found in src.features")

# --------------------------- Load artifacts before UI ---------------------------
pipe, feature_order, feature_means, meta = load_bundle()
ts, teams = load_team_state()
ALIAS_LOOKUP = build_alias_lookup(teams)

# --------------------------- HERO ---------------------------
st.markdown(
    """
    <div class="hero">
      <h1>LaLiga 25/26 Match Predictor</h1>
      <div class="sub">Pick a matchup, optionally blend with live market odds, and see calibrated probabilities.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- TABS ---------------------------
tab_predict, tab_train, tab_about = st.tabs(["🔮 Predict", "🧪 Train / Data", "ℹ️ About"])

# --------------------------- PREDICT ---------------------------
with tab_predict:
    # Card: Match & Options
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Match</h3>", unsafe_allow_html=True)
        home = st.selectbox("Home team", teams, index=0, key="home_team_ui")
        away = st.selectbox("Away team", [t for t in teams if t != home], index=0, key="away_team_ui")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Options</h3>", unsafe_allow_html=True)
        use_market = st.toggle("Enable live market odds", value=False, help="Blend model with normalized bookmaker odds.")
        market_weight = 0.25
        if use_market:
            market_weight = st.slider("Market weight", 0.0, 1.0, 0.25, 0.05, help="0 = model only, 1 = market only")
        st.markdown("</div>", unsafe_allow_html=True)

    # Card: Market odds (only if toggled)
    oddsH = oddsD = oddsA = None
    if use_market:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Market odds</h3>", unsafe_allow_html=True)

        api_cols = st.columns([1, 1, 2])
        with api_cols[0]:
            api_key_ui = st.text_input("Odds API key (kept local to this session)", type="password",
                                       value=st.session_state.get("odds_api_key", ""))
            if api_key_ui:
                st.session_state["odds_api_key"] = api_key_ui
        with api_cols[1]:
            auto_fetch = st.toggle("Auto-fetch on team change", value=True)
        with api_cols[2]:
            st.caption("Tip: set `odds_api_key` in `.streamlit/secrets.toml` or `ODDS_API_KEY` env var.")

        api_key = get_odds_api_key()
        fetch_now = st.button("Fetch live odds", key="btn_fetch_odds")

        # Persist odds between interactions
        if "odds_values" not in st.session_state:
            st.session_state["odds_values"] = {"H": 2.00, "D": 3.40, "A": 3.20, "src": None}

        changed = (home != st.session_state.get("prev_home")) or (away != st.session_state.get("prev_away"))
        if (auto_fetch and changed) or fetch_now:
            if api_key:
                result, reason = fetch_live_odds(home, away, api_key, ALIAS_LOOKUP, days_from=90)
                if result:
                    oh, od, oa, src = result
                    st.session_state["odds_values"] = {"H": oh, "D": od, "A": oa, "src": src}
                    st.success(f"Loaded odds from {src}", icon="✅")
                else:
                    st.warning(f"Couldn’t fetch odds: {reason}", icon="⚠️")
            else:
                st.info("No API key set. Enter it above or in secrets/env.", icon="🔑")

        st.session_state["prev_home"] = home
        st.session_state["prev_away"] = away

        odds_vals = st.session_state["odds_values"]
        cH, cD, cA = st.columns(3)
        oddsH = cH.number_input("Odds H", min_value=1.01, value=float(odds_vals["H"]), step=0.01, format="%.2f")
        oddsD = cD.number_input("Odds D", min_value=1.01, value=float(odds_vals["D"]), step=0.01, format="%.2f")
        oddsA = cA.number_input("Odds A", min_value=1.01, value=float(odds_vals["A"]), step=0.01, format="%.2f")

        if st.session_state["odds_values"].get("src"):
            st.caption(f'<span class="pill">Source: {st.session_state["odds_values"]["src"]}</span>',
                       unsafe_allow_html=True)

        with st.expander("Show upcoming fixtures (Odds API names)"):
            evs, err = list_upcoming_events(get_odds_api_key(), days_from=90)
            if err:
                st.write(err)
            elif not evs:
                st.write("No events returned.")
            else:
                st.caption("Home vs Away as returned by the API (normalized in italics):")
                for ev in evs:
                    h = ev.get("home_team", ""); a = ev.get("away_team", "")
                    st.write(f"• {h} vs {a}  _({_norm(h)} vs {_norm(a)})_")

        st.markdown("</div>", unsafe_allow_html=True)

    # Card: Predict
    st.markdown('<div class="card">', unsafe_allow_html=True)
    top_row = st.columns([1, 1])
    with top_row[0]:
        predict_clicked = st.button("Predict", key="btn_predict_main")
    with top_row[1]:
        st.caption("Calibrated probabilities; fair odds shown below.")

    if predict_clicked:
        X = vector_for_match(home, away, feature_order, ts, feature_means)
        # Robust class-order handling:
        model_probs = predict_proba_HDA(pipe, X.values)  # -> [H, D, A]

        final_probs = model_probs[:]
        if use_market and (oddsH and oddsD and oddsA):
            try:
                market_probs, _ = normalize_probs_from_odds(oddsH, oddsD, oddsA)
                blend = (1 - market_weight) * np.array(model_probs) + market_weight * np.array(market_probs)
                final_probs = (blend / blend.sum()).tolist()
            except Exception:
                st.warning("Invalid odds; falling back to model-only.")
                final_probs = model_probs

        labels = ["Home (H)", "Draw (D)", "Away (A)"]
        pred_idx = int(np.argmax(final_probs))
        pred_label = ["H", "D", "A"][pred_idx]

        st.markdown(f"**Prediction:** {pred_label}")
        st.write({"H": round(final_probs[0], 3), "D": round(final_probs[1], 3), "A": round(final_probs[2], 3)})

        chart_df = pd.DataFrame({"Outcome": labels, "Probability": final_probs})
        st.bar_chart(chart_df.set_index("Outcome"))

        fair = fair_odds(final_probs)
        st.caption(f"Fair odds →  H: {fair[0]:.2f} • D: {fair[1]:.2f} • A: {fair[2]:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- TRAIN / DATA ---------------------------
with tab_train:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Dataset & Training</h3>", unsafe_allow_html=True)

    season_codes = available_season_codes()
    labels = [season_code_to_label(s) for s in season_codes]
    default_last_n = 3
    default_idx = list(range(max(0, len(season_codes) - default_last_n), len(season_codes)))
    selected_labels = st.multiselect(
        "Training seasons",
        labels,
        default=[labels[i] for i in default_idx],
        key="train_seasons"
    )
    selected_codes = [label_to_season_code(x) for x in selected_labels]

    predict_label = st.selectbox(
        "Prediction season (metadata only)",
        [season_code_to_label(s) for s in season_codes],
        index=season_codes.index("2526"),
        key="predict_season"
    )
    predict_code = label_to_season_code(predict_label)

    retrain_clicked = st.button("Update dataset & retrain", key="btn_retrain")
    if retrain_clicked:
        with st.status("Preparing data…", expanded=True) as status:
            # Download CSVs (skip 25/26 if the file is not published yet)
            for s in selected_codes:
                if s == "2526":
                    st.write(f"Note: {season_code_to_label(s)} may not have CSVs yet; training will ignore it if missing.")
                    continue
                url = f"https://www.football-data.co.uk/mmz4281/{s}/SP1.csv"
                DATA_RAW.mkdir(parents=True, exist_ok=True)
                out = DATA_RAW / f"{CONFIG['league']}_{s}.csv"
                if not out.exists():
                    st.write(f"Downloading {season_code_to_label(s)}…")
                    try:
                        rr = requests.get(url, timeout=30)
                        rr.raise_for_status()
                        out.write_bytes(rr.content)
                    except Exception as e:
                        st.warning(f"Could not download {s}: {e}")

            st.write("Building features…")
            run_build_features(seasons=selected_codes)

            st.write("Training model…")
            acc, ll, meta_out = train_model()

            # Remember seasons & predict season in bundle
            bundle = joblib.load(CONFIG["model_path"])
            bundle["meta"] = {**bundle.get("meta", {}), "seasons_used": selected_codes, "predict_season": predict_code}
            joblib.dump(bundle, CONFIG["model_path"])

            # refresh caches/artifacts in this session
            load_bundle.clear()
            load_team_state.clear()

            # re-read artifacts for the live session
            pipe, feature_order, feature_means, meta = load_bundle()
            ts, teams = load_team_state()
            ALIAS_LOOKUP = build_alias_lookup(teams)

            status.update(label=f"Done → Acc={acc:.3f}, LogLoss={ll:.3f}", state="complete")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- ABOUT ---------------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>About</h3>", unsafe_allow_html=True)
    st.write("""
This project is a LaLiga Match predictor: by using logistic regression on datasets of past seasons this program predicts the future matches with an accuracy of ~50–70%.
There is also an option to blend bookmaker odds via The Odds API.

Thank you for using the program!
""")
    st.markdown("</div>", unsafe_allow_html=True)
