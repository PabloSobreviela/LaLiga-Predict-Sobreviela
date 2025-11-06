"""
Download La Liga match results from football-data.co.uk.
Saves CSV files into data/raw.
"""
from pathlib import Path
import argparse
import requests
from src.config import CONFIG

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

def fetch_csv(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def main(season: str | None = None):
    raw = Path(CONFIG["raw_data_path"])
    seasons = [season] if season else CONFIG["seasons"]
    for s in seasons:
        url = BASE.format(season=s, league=CONFIG["league"])
        out = raw / f"{CONFIG['league']}_{s}.csv"
        print(f"Fetching {url} -> {out}")
        fetch_csv(url, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, help="Season code like 2425 (optional)")
    args = parser.parse_args()
    main(args.season)