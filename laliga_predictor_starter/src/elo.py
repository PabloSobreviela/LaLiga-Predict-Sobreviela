from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class Elo:
    base: float = 1500.0
    k: float = 20.0
    home_adv: float = 60.0
    ratings: Dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expected_home(self, home: str, away: str) -> float:
        rh = self.get(home) + self.home_adv
        ra = self.get(away)
        return 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))

    def update(self, home: str, away: str, outcome: str) -> Tuple[float, float]:
        """
        outcome: 'H' (home win), 'D' (draw), 'A' (away win)
        returns (home_rating, away_rating) AFTER the match
        """
        eh = self.expected_home(home, away)
        if outcome == "H":
            sh, sa = 1.0, 0.0
        elif outcome == "D":
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0
        rh = self.get(home)
        ra = self.get(away)
        rh_new = rh + self.k * (sh - eh)
        ra_new = ra + self.k * (sa - (1.0 - eh))
        self.ratings[home] = rh_new
        self.ratings[away] = ra_new
        return rh_new, ra_new
