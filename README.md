# LaLiga Match Predictor

**Live app:** [https://laliga-predict-sobreviela.streamlit.app/](https://laliga-predict-sobreviela.streamlit.app/)

End-to-end football match outcome predictor for LaLiga (H/D/A) with optional blend of real bookmaker odds. Clean Streamlit UI, reproducible training pipeline, and simple deployment.

---

## Features

- **Predict H/D/A** with calibrated probabilities and fair odds (1/p)
- **Live odds (optional)** via The Odds API (1X2) with alias matching (e.g. FC Barcelona and Barcelona)
- **One-click retraining** in the UI: select seasons, auto-download CSVs, build features, train and save
- **La Liga branding (new):** official logo, team crests, display names with accents (Atlético de Madrid, Deportivo Alavés)
- **Seasons 09/10 to 25/26 (new):** train on any subset from 16 seasons; missing CSVs download from football-data.co.uk

---

## How it works

- **Model:** HistGradientBoostingClassifier ensemble (5 seeds averaged) with sample weights for class balance
- **Target:** H/D/A (Home, Draw, Away)
- **Features:** Elo ratings (home, away, diff), rolling form over last 10 matches (goals for/against, shots on target, corners, fouls), rest days (home, away, diff), and engineered diffs
- **Data:** LaLiga SP1 CSVs from football-data.co.uk, stored as Parquet (features.parquet, team_state.parquet)
- **Split:** Time-based 83/17: train on first 83% of matches by date, evaluate on last 17%. No future leakage.

---

## Repo structure

```
laliga_predictor_starter/
├─ app.py                    # Streamlit app (UI, prediction, odds, train controls)
├─ src/
│  ├─ config.py              # Paths, seasons, Elo, model params
│  ├─ features.py            # CSV ingestion, feature engineering, team_state
│  ├─ train_model.py         # Train, evaluate, save bundle
│  ├─ ensemble.py            # HGBMultiSeedEnsemble
│  └─ team_display.py        # Team names and crest URLs for UI
├─ data/
│  ├─ raw/                   # SP1_*.csv per season
│  └─ processed/             # features.parquet, team_state.parquet
├─ models/
│  └─ model.joblib           # Saved pipeline (created after training)
├─ requirements.txt
└─ README.md
```

---

## Setup (local)

1. **Create venv and install**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate

   pip install -r laliga_predictor_starter/requirements.txt
   ```

2. **Run the app**
   ```bash
   streamlit run laliga_predictor_starter/app.py
   ```

3. **Train in the UI**
   - Select seasons (e.g. 09/10 to 24/25)
   - Click **Download missing data and retrain**
   - Model is saved to `models/model.joblib`

---

## Odds integration (optional)

- **Provider:** The Odds API (v4), market `h2h` (1X2)
- **Usage:** Toggle "Blend live bookmaker odds", then Fetch live odds or enter manually
- **Alias matching:** Handles variants (RCD Espanyol and Espanyol, FC Barcelona and Barcelona)

---

## Credits

- Match data: **football-data.co.uk** (LaLiga SP1)
- Odds: **The Odds API**
- ML: **scikit-learn**, **pandas**, **numpy**, **Streamlit**

---

## License

Intended for learning, research, and portfolio use. Predictions are not financial advice.
