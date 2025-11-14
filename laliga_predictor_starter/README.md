# LaLiga 25/26 Match Predictor

**End-to-end football (soccer) match outcome predictor for LaLiga** with an optional blend of **real bookmaker odds**. Web UI in Streamlit + reproducible training pipeline.

---

## Highlights

- **Base model accuracy:** **~58–63%** on held-out, time-split validation **without** using market odds.  
- **Optional odds blend:** Pulls live **1X2** prices via The Odds API; typically improves calibration/log loss.
- **One-click retrain:** Choose seasons in the UI → auto-download CSVs → build features → train & save model.
- **Newly promoted teams:** Seeded Elo + back-filled rolling form to avoid zeros/NaNs and wild swings.
- **Nice UX:** Custom CSS, fair odds (1/p) display, odds alias matching (handles “FC Barcelona” vs “Barcelona”).

---

## How it works

- **Model:** Multiclass Logistic Regression (scikit-learn), optionally calibrated.
- **Target:** H/D/A (Home/Draw/Away).
- **Features (examples):**  
  - Team Elo (home/away/diff)  
  - Rolling form: GF/GA/GD, shots on target for/against (last N), rest days  
  - Engineered diffs (home − away)
- **Data:** LaLiga CSVs (SP1) from football-data.co.uk; processed to Parquet.

---

## Run locally

1. **Create env & install**
   ```bash
   python -m venv .venv
   # Windows: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r laliga_predictor_starter/requirements.txt
