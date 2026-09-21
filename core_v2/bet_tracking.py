from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BET_COLUMNS = [
    "bet_id", "season", "week", "game_id", "matchup", "team", "bet_spread",
    "odds_american", "sportsbook", "stake", "entered_at_utc",
    "model_fair_home_spread_at_entry", "market_home_spread_at_entry",
    "model_side_at_entry", "model_edge_points_at_entry", "status", "notes",
]


def load_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    bets = pd.read_csv(path)
    for column in BET_COLUMNS:
        if column not in bets:
            bets[column] = pd.NA
    return bets[BET_COLUMNS].copy()


def make_bet_id(season: int, week: int, game_id: str, team: str, entered_at_utc: str) -> str:
    stamp = pd.Timestamp(entered_at_utc).strftime("%Y%m%dT%H%M%S")
    return f"{int(season)}-{int(week):02d}-{game_id}-{team}-{stamp}"


def append_bet(path: Path, record: dict) -> pd.DataFrame:
    bets = load_bets(path)
    row = {column: record.get(column, pd.NA) for column in BET_COLUMNS}
    if pd.isna(row["entered_at_utc"]):
        row["entered_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if pd.isna(row["bet_id"]):
        row["bet_id"] = make_bet_id(
            int(row["season"]), int(row["week"]), str(row["game_id"]),
            str(row["team"]), str(row["entered_at_utc"]),
        )
    updated = pd.concat([bets, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return updated


def american_profit(stake: float, odds: float) -> float:
    stake, odds = float(stake), float(odds)
    return stake * (100.0 / abs(odds)) if odds < 0 else stake * (odds / 100.0)


def selected_team_close(row: pd.Series, closing_home_spread: float) -> float:
    if pd.isna(closing_home_spread):
        return float("nan")
    if str(row["team"]) == str(row.get("home_team")):
        return float(closing_home_spread)
    return -float(closing_home_spread)
