"""Determine which dashboard games are still eligible for pregame odds."""

from datetime import datetime, timezone

import pandas as pd


TEAM_ALIASES = {
    "GNB": "GB",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "WSH": "WAS",
}


def _team(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return TEAM_ALIASES.get(value, value)


def select_pregame_game_keys(
    board,
    schedule=None,
    completed_game_keys=None,
    now_utc=None,
):
    """Return board games that sportsbooks should still offer pregame."""
    if board.empty:
        return tuple()

    if "game_key" in board.columns:
        board_keys = set(board["game_key"].dropna().astype(str))
    else:
        board_keys = {
            f"{int(row.season)}_W{int(row.week)}_"
            f"{_team(row.away_team)}_{_team(row.home_team)}"
            for row in board.itertuples()
        }

    completed = {
        str(key)
        for key in (completed_game_keys or set())
        if pd.notna(key)
    }
    eligible = board_keys - completed

    if schedule is None or schedule.empty or not eligible:
        return tuple(sorted(eligible))

    games = schedule.copy()
    required = {"season", "week", "away_team", "home_team"}
    if not required.issubset(games.columns):
        return tuple(sorted(eligible))

    games["away_team"] = games["away_team"].map(_team)
    games["home_team"] = games["home_team"].map(_team)
    games["_game_key"] = games.apply(
        lambda row: (
            f"{int(row['season'])}_W{int(row['week'])}_"
            f"{row['away_team']}_{row['home_team']}"
        ),
        axis=1,
    )
    games = games.loc[games["_game_key"].isin(eligible)].copy()

    if "kickoff_utc" in games.columns:
        kickoffs = pd.to_datetime(
            games["kickoff_utc"], errors="coerce", utc=True
        )
    elif "start_time" in games.columns:
        kickoffs = pd.to_datetime(
            games["start_time"], errors="coerce", utc=True
        )
    elif {"gameday", "gametime"}.issubset(games.columns):
        local_kickoffs = pd.to_datetime(
            games["gameday"].astype(str)
            + " "
            + games["gametime"].fillna("00:00").astype(str),
            errors="coerce",
        )
        kickoffs = local_kickoffs.dt.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    else:
        return tuple(sorted(eligible))

    now_value = (
        now_utc
        if now_utc is not None
        else datetime.now(timezone.utc)
    )
    now = pd.Timestamp(now_value)
    if now.tzinfo is None:
        now_utc_timestamp = now.tz_localize("UTC")
    else:
        now_utc_timestamp = now.tz_convert("UTC")

    started = set(
        games.loc[kickoffs.le(now_utc_timestamp), "_game_key"]
    )
    return tuple(sorted(eligible - started))
