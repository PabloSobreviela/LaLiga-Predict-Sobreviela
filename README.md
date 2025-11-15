# LaLiga 25/26 Match Predictor 

**Website: https://laliga-predict-sobreviela.streamlit.app/**

**End-to-end football (soccer) match outcome predictor for LaLiga** with an optional blend of **real bookmaker odds**. Clean Streamlit UI, reproducible training pipeline, and simple deployment.

> **Base model accuracy:** **~58–63%** on held-out, time-split validation **without** using market odds.  
> Adding market odds generally improves **calibration / LogLoss**.

---

## ✨ Features

- **Predict H/D/A** with calibrated probabilities (and **fair odds = 1/p**).
- **Live odds (optional)** via The Odds API (1X2) + alias matching (e.g., “FC Barcelona” vs “Barcelona”).
- **One-click retraining** in the UI: choose seasons → auto-download CSVs → build features → train and save.
- **Newly promoted teams support**: seeded Elo + back-filled form to avoid unstable 0/1 values.
- **Nice UX**: custom CSS, compact cards, tooltips, and helpful status messages.

---

## 🧠 How it works

- **Model:** Scikit-learn **Pipeline** with **Multiclass Logistic Regression** (optionally calibrated).
- **Target:** H/D/A (Home/Draw/Away).
- **Core features:**
  - Elo rating (home, away, diff).
  - Rolling form: GF/GA/GD, shots on target for/against (last *N* matches).
  - Rest days (home/away & diff).
  - Engineered diffs of the above.
- **Data source:** LaLiga SP1 CSVs from **football-data.co.uk** → normalized and stored as **Parquet**.

---

## 📦 Repo structure

```
laliga_predictor_starter/
├─ app.py                    # Streamlit app (UI, prediction, odds fetch, train controls)
├─ src/
│  ├─ config.py              # Paths / constants (models/, data/, etc.)
│  ├─ features.py            # CSV ingestion, normalization, feature engineering, team_state
│  └─ train_model.py         # Train/evaluate pipeline, save model bundle
├─ data/
│  ├─ raw/                   # Downloaded CSVs per season (SP1.csv)
│  └─ processed/             # features.parquet, team_state.parquet
├─ models/
│  └─ model.joblib           # Saved pipeline bundle (created after training)
├─ requirements.txt
├─ .streamlit/
│  └─ secrets.toml           # (optional) odds_api_key = "YOUR_KEY"
└─ README.md
```

---

## 🛠 Setup (Local)

1. **Create a virtual env & install deps**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate

   pip install -r laliga_predictor_starter/requirements.txt
   ```

2. **(Optional) Set an Odds API key**  
   - Environment variable:
     ```bash
     # macOS/Linux
     export ODDS_API_KEY="your_key"
     # Windows PowerShell
     $Env:ODDS_API_KEY="your_key"
     ```
   - Or create `.streamlit/secrets.toml`:
     ```toml
     odds_api_key = "your_key"
     ```

3. **Run the app**
   ```bash
   streamlit run laliga_predictor_starter/app.py
   ```

4. **Train in the UI**
   - Open the **🧪 Train / Data** tab.
   - Select seasons (e.g., 23/24, 24/25).
   - Click **Update dataset & retrain**.
   - After training, the model is saved to `models/model.joblib` and the UI refreshes.

---

## 📊 Training & Evaluation

- **Split:** Time-based split on features parquet (train past → test future).
- **Metrics (printed/logged):**
  - **Accuracy** (~58–63% depending on selected seasons)
  - **LogLoss**
  - Optional confusion matrix in development (CLI/eval scripts)
- **With odds blend:** Typically **better calibration** (lower LogLoss). Magnitude depends on bookmaker quality and fixture timing.

---

## 🔁 Odds Integration (Optional)

- **Provider:** The Odds API (v4).
- **Market:** `h2h` (1X2 home/draw/away), decimals.
- **Regions:** `eu, uk, us` (merged).
- **Alias matching:** Robust team-name matcher (e.g., “RCD Espanyol” vs “Espanyol”; “FC Barcelona” vs “Barcelona”).

**Usage in UI:**
- Toggle **Enable live market odds**.
- Enter API key once (persisted for the session via Streamlit state/secrets).
- Auto-fetch on team change, or click **Fetch live odds**.
- Manually edit odds if the API doesn’t return them.

---

## 🆙 Newly Promoted Teams

- **Seeded Elo**: Assign reasonable priors to avoid extreme 0/1 probability artifacts.
- **Back-filled rolling form**: Use league averages / conservative priors until enough matches are available.
- **Result**: More stable early-season predictions for promoted clubs.

## 🔒 Notes & Limitations

- No player-level injuries/lineups; macro team signals only.
- CSV columns can vary slightly by season; the pipeline normalizes common variants.
- The Odds API has quotas and regional availability; manual overrides are supported.
- Predictions are **not financial advice**.

---

## 📈 Roadmap

- Add tree-based model baselines (e.g., XGBoost/LightGBM) for comparison.
- SHAP-style explanations per fixture.
- Automated backtests with time-split sweeps + simple model registry.
- Dockerfile for containerized deployment.

---

## 📜 License / Usage

- Intended for learning, research, and portfolio use.
- Respect sportsbook/API terms and local regulations.

---

## 🙌 Credits

- Match CSVs: **football-data.co.uk** (LaLiga SP1).
- Odds: **The Odds API** (v4).
- ML/infra: **scikit-learn**, **pandas**, **numpy**, **Streamlit**.

1. **Create env & install**
   ```bash
   python -m venv .venv
   # Windows: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r laliga_predictor_starter/requirements.txt
