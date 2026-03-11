# La Liga team display names (with accents) and crest URLs.
# Maps football-data.co.uk CSV/features team names -> (display_name, crest_url)
# Crests from FootyLogos CDN (PNG) for accuracy; fallback to football-data.org for others.

from typing import Dict, Tuple

# FootyLogos CDN (correct La Liga crests)
FL = "https://cdn.prod.website-files.com/68f550992570ca0322737dc2"

# Internal name (from CSV/features) -> (Display name with accents, crest URL)
TEAM_DISPLAY: Dict[str, Tuple[str, str | None]] = {
    # Current & recent La Liga — FootyLogos URLs for correct crests
    "Alaves": ("Deportivo Alavés", f"{FL}/68f574c28abeb45efd9bbd68_deportivo-alaves-footballlogos-org.png"),
    "Almeria": ("UD Almería", "https://crests.football-data.org/2202.svg"),
    "Ath Bilbao": ("Athletic Club", "https://crests.football-data.org/77.svg"),
    "Ath Madrid": ("Atlético de Madrid", f"{FL}/68f57ae622e0827f30d03d63_atletico-madrid-footballlogos-org.png"),
    "Barcelona": ("FC Barcelona", "https://crests.football-data.org/81.svg"),
    "Betis": ("Real Betis", "https://crests.football-data.org/90.svg"),
    "Celta": ("RC Celta de Vigo", f"{FL}/68f573c734dc0a249b2a0f2d_celta-vigo-footballlogos-org.png"),
    "Celta Vigo": ("RC Celta de Vigo", f"{FL}/68f573c734dc0a249b2a0f2d_celta-vigo-footballlogos-org.png"),
    "Cordoba": ("Córdoba CF", "https://crests.football-data.org/277.svg"),
    "Eibar": ("SD Eibar", "https://crests.football-data.org/278.svg"),
    "Elche": ("Elche CF", "https://crests.football-data.org/797.svg"),
    "Espanol": ("RCD Espanyol de Barcelona", "https://crests.football-data.org/80.svg"),
    "Espanyol": ("RCD Espanyol de Barcelona", "https://crests.football-data.org/80.svg"),
    "Getafe": ("Getafe CF", "https://crests.football-data.org/82.svg"),
    "Girona": ("Girona FC", f"{FL}/68f578ba6fbc8de137b1bbe5_girona-fc-footballlogos-org.png"),
    "Granada": ("Granada CF", "https://crests.football-data.org/83.svg"),
    "Huesca": ("SD Huesca", "https://crests.football-data.org/299.svg"),
    "La Coruna": ("RC Deportivo de La Coruña", "https://crests.football-data.org/560.svg"),
    "Las Palmas": ("UD Las Palmas", "https://crests.football-data.org/275.svg"),
    "Leganes": ("CD Leganés", f"{FL}/68f784a4a6ab89649fe88b26_cd-leganes-footballlogos-org.png"),
    "Levante": ("Levante UD", "https://crests.football-data.org/88.svg"),
    "Malaga": ("Málaga CF", "https://crests.football-data.org/84.svg"),
    "Mallorca": ("RCD Mallorca", f"{FL}/68f57c30f79bc28af844c630_rcd-mallorca-footballlogos-org.png"),
    "Osasuna": ("CA Osasuna", "https://crests.football-data.org/79.svg"),
    "Real Madrid": ("Real Madrid CF", "https://crests.football-data.org/86.svg"),
    "Sevilla": ("Sevilla FC", "https://crests.football-data.org/559.svg"),
    "Sociedad": ("Real Sociedad", "https://crests.football-data.org/92.svg"),
    "Sp Gijon": ("Sporting de Gijón", "https://crests.football-data.org/560.svg"),
    "Valencia": ("Valencia CF", "https://crests.football-data.org/95.svg"),
    "Valladolid": ("Real Valladolid", "https://crests.football-data.org/250.svg"),
    "Vallecano": ("Rayo Vallecano", "https://crests.football-data.org/87.svg"),
    "Rayo Vallecano": ("Rayo Vallecano", "https://crests.football-data.org/87.svg"),
    "Villarreal": ("Villarreal CF", f"{FL}/68f56451311816a40ac12bf6_villarreal-cf-footballlogos-org.png"),
    "Cadiz": ("Cádiz CF", "https://crests.football-data.org/795.svg"),
    # Aliases / alternate spellings from CSV
    "Athletic Club": ("Athletic Club", "https://crests.football-data.org/77.svg"),
    "Atletico Madrid": ("Atlético de Madrid", f"{FL}/68f57ae622e0827f30d03d63_atletico-madrid-footballlogos-org.png"),
    "Atletico de Madrid": ("Atlético de Madrid", f"{FL}/68f57ae622e0827f30d03d63_atletico-madrid-footballlogos-org.png"),
    "Real Betis": ("Real Betis", "https://crests.football-data.org/90.svg"),
    "Real Sociedad": ("Real Sociedad", "https://crests.football-data.org/92.svg"),
    "RC Celta": ("RC Celta de Vigo", f"{FL}/68f573c734dc0a249b2a0f2d_celta-vigo-footballlogos-org.png"),
    "Villarreal CF": ("Villarreal CF", f"{FL}/68f56451311816a40ac12bf6_villarreal-cf-footballlogos-org.png"),
    "CA Osasuna": ("CA Osasuna", "https://crests.football-data.org/79.svg"),
    "Deportivo Alaves": ("Deportivo Alavés", f"{FL}/68f574c28abeb45efd9bbd68_deportivo-alaves-footballlogos-org.png"),
    "UD Almeria": ("UD Almería", "https://crests.football-data.org/2202.svg"),
    "Getafe CF": ("Getafe CF", "https://crests.football-data.org/82.svg"),
    "Girona FC": ("Girona FC", f"{FL}/68f578ba6fbc8de137b1bbe5_girona-fc-footballlogos-org.png"),
    "RCD Mallorca": ("RCD Mallorca", f"{FL}/68f57c30f79bc28af844c630_rcd-mallorca-footballlogos-org.png"),
    "UD Las Palmas": ("UD Las Palmas", "https://crests.football-data.org/275.svg"),
    "CD Leganes": ("CD Leganés", f"{FL}/68f784a4a6ab89649fe88b26_cd-leganes-footballlogos-org.png"),
    "Cadiz CF": ("Cádiz CF", "https://crests.football-data.org/795.svg"),
    "Granada CF": ("Granada CF", "https://crests.football-data.org/83.svg"),
    "Elche CF": ("Elche CF", "https://crests.football-data.org/797.svg"),
    "Sevilla FC": ("Sevilla FC", "https://crests.football-data.org/559.svg"),
    "RCD Espanyol": ("RCD Espanyol de Barcelona", "https://crests.football-data.org/80.svg"),
    "Levante UD": ("Levante UD", "https://crests.football-data.org/88.svg"),
    "Real Madrid CF": ("Real Madrid CF", "https://crests.football-data.org/86.svg"),
    "Real Oviedo": ("Real Oviedo", "https://crests.football-data.org/263.svg"),
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
