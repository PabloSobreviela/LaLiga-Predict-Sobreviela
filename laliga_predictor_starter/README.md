# LaLiga Predictor — Starter

Scaffold created on 2025-08-26 to jumpstart a match outcome prediction project.

## Layout

```
.
├── src/
│   ├── scrape_data.py
│   ├── features.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── infer.py
│   └── config.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── requirements.txt
└── pyproject.toml
```

- Keep your recap here or in a new chat: `LaLiga_Predictor_Recap.md`.
- Place any downloaded CSVs in `data/raw/`. Processed feature tables go to `data/processed/`. Trained models live in `models/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/scrape_data.py --season 2024-2025
python src/features.py
python src/train_model.py
python src/evaluate.py
python src/infer.py --home "Barcelona" --away "Sevilla"
```