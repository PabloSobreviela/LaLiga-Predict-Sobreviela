# La Liga team display names (with accents) and crest URLs.
# Maps football-data.co.uk CSV/features team names -> (display_name, crest_url)

from typing import Dict, Tuple

# football-data.org crest IDs (La Liga / Primera): api.football-data.org
CREST_BASE = "https://crests.football-data.org"

# Internal name (from CSV/features) -> (Display name with accents, crest ID or None)
TEAM_DISPLAY: Dict[str, Tuple[str, str | None]] = {
    # Current & recent La Liga
    "Alaves": ("Deportivo Alavés", f"{CREST_BASE}/263.svg"),
    "Almeria": ("UD Almería", f"{CREST_BASE}/2202.svg"),
    "Ath Bilbao": ("Athletic Club", f"{CREST_BASE}/77.svg"),
    "Ath Madrid": ("Atlético de Madrid", f"{CREST_BASE}/78.svg"),
    "Barcelona": ("FC Barcelona", f"{CREST_BASE}/81.svg"),
    "Betis": ("Real Betis", f"{CREST_BASE}/90.svg"),
    "Celta": ("RC Celta de Vigo", f"{CREST_BASE}/94.svg"),
    "Celta Vigo": ("RC Celta de Vigo", f"{CREST_BASE}/94.svg"),
    "Cordoba": ("Córdoba CF", f"{CREST_BASE}/277.svg"),
    "Eibar": ("SD Eibar", f"{CREST_BASE}/278.svg"),
    "Elche": ("Elche CF", f"{CREST_BASE}/797.svg"),
    "Espanol": ("RCD Espanyol de Barcelona", f"{CREST_BASE}/80.svg"),
    "Espanyol": ("RCD Espanyol de Barcelona", f"{CREST_BASE}/80.svg"),
    "Getafe": ("Getafe CF", f"{CREST_BASE}/82.svg"),
    "Girona": ("Girona FC", f"{CREST_BASE}/298.svg"),
    "Granada": ("Granada CF", f"{CREST_BASE}/83.svg"),
    "Huesca": ("SD Huesca", f"{CREST_BASE}/299.svg"),
    "La Coruna": ("RC Deportivo de La Coruña", f"{CREST_BASE}/560.svg"),
    "Las Palmas": ("UD Las Palmas", f"{CREST_BASE}/275.svg"),
    "Leganes": ("CD Leganés", f"{CREST_BASE}/569.svg"),
    "Levante": ("Levante UD", f"{CREST_BASE}/88.svg"),
    "Malaga": ("Málaga CF", f"{CREST_BASE}/84.svg"),
    "Mallorca": ("RCD Mallorca", f"{CREST_BASE}/798.svg"),
    "Osasuna": ("CA Osasuna", f"{CREST_BASE}/79.svg"),
    "Real Madrid": ("Real Madrid CF", f"{CREST_BASE}/86.svg"),
    "Sevilla": ("Sevilla FC", f"{CREST_BASE}/559.svg"),
    "Sociedad": ("Real Sociedad", f"{CREST_BASE}/92.svg"),
    "Sp Gijon": ("Sporting de Gijón", f"{CREST_BASE}/560.svg"),  # shared fallback
    "Valencia": ("Valencia CF", f"{CREST_BASE}/95.svg"),
    "Valladolid": ("Real Valladolid", f"{CREST_BASE}/250.svg"),
    "Vallecano": ("Rayo Vallecano", f"{CREST_BASE}/87.svg"),
    "Rayo Vallecano": ("Rayo Vallecano", f"{CREST_BASE}/87.svg"),
    "Villarreal": ("Villarreal CF", f"{CREST_BASE}/99.svg"),
    "Cadiz": ("Cádiz CF", f"{CREST_BASE}/795.svg"),
    # Aliases / alternate spellings from CSV
    "Athletic Club": ("Athletic Club", f"{CREST_BASE}/77.svg"),
    "Atletico Madrid": ("Atlético de Madrid", f"{CREST_BASE}/78.svg"),
    "Atletico de Madrid": ("Atlético de Madrid", f"{CREST_BASE}/78.svg"),
    "Real Betis": ("Real Betis", f"{CREST_BASE}/90.svg"),
    "Real Sociedad": ("Real Sociedad", f"{CREST_BASE}/92.svg"),
    "RC Celta": ("RC Celta de Vigo", f"{CREST_BASE}/94.svg"),
    "Villarreal CF": ("Villarreal CF", f"{CREST_BASE}/99.svg"),
    "CA Osasuna": ("CA Osasuna", f"{CREST_BASE}/79.svg"),
    "Deportivo Alaves": ("Deportivo Alavés", f"{CREST_BASE}/263.svg"),
    "UD Almeria": ("UD Almería", f"{CREST_BASE}/2202.svg"),
    "Getafe CF": ("Getafe CF", f"{CREST_BASE}/82.svg"),
    "Girona FC": ("Girona FC", f"{CREST_BASE}/298.svg"),
    "RCD Mallorca": ("RCD Mallorca", f"{CREST_BASE}/798.svg"),
    "UD Las Palmas": ("UD Las Palmas", f"{CREST_BASE}/275.svg"),
    "CD Leganes": ("CD Leganés", f"{CREST_BASE}/569.svg"),
    "Cadiz CF": ("Cádiz CF", f"{CREST_BASE}/795.svg"),
    "Granada CF": ("Granada CF", f"{CREST_BASE}/83.svg"),
    "Rayo Vallecano": ("Rayo Vallecano", f"{CREST_BASE}/87.svg"),
    "Elche CF": ("Elche CF", f"{CREST_BASE}/797.svg"),
    "Sevilla FC": ("Sevilla FC", f"{CREST_BASE}/559.svg"),
    "RCD Espanyol": ("RCD Espanyol de Barcelona", f"{CREST_BASE}/80.svg"),
    "Levante UD": ("Levante UD", f"{CREST_BASE}/88.svg"),
    "Real Madrid CF": ("Real Madrid CF", f"{CREST_BASE}/86.svg"),
    "Real Oviedo": ("Real Oviedo", f"{CREST_BASE}/263.svg"),  # fallback crest
}


def get_display_name(internal_name: str) -> str:
    """Return display name with accents for UI. Fallback to internal name."""
    if internal_name in TEAM_DISPLAY:
        return TEAM_DISPLAY[internal_name][0]
    return internal_name


def get_crest_url(internal_name: str) -> str | None:
    """Return crest URL for team, or None if unknown."""
    if internal_name in TEAM_DISPLAY:
        return TEAM_DISPLAY[internal_name][1]
    return None
