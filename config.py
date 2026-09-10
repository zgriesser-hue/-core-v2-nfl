from __future__ import annotations

from dataclasses import dataclass


TEAM_MAP = {
    "GNB": "GB",
    "JAC": "JAX",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NWE": "NE",
    "NOR": "NO",
    "OAK": "LV",
    "SD": "LAC",
    "SFO": "SF",
    "STL": "LA",
    "TAM": "TB",
    "WSH": "WAS",
}

CORE_CONSENSUS_BOOKS = frozenset(
    {"draftkings", "fanduel", "bet365", "caesars"}
)

EXCLUDED_BOOKS = frozenset({"betmgm"})


@dataclass(frozen=True)
class ProjectionConfig:
    default_hfa: float = 2.20
    line_round_digits: int = 3

