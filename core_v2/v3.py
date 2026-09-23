from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "season",
    "week",
    "team",
    "offense_strength",
    "defense_strength",
    "qb_value",
    "personnel_adjustment",
    "market_strength",
    "state_version",
]

PROJECTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "away_offense_strength",
    "home_offense_strength",
    "away_defense_strength",
    "home_defense_strength",
    "away_qb_value",
    "home_qb_value",
    "away_personnel_adjustment",
    "home_personnel_adjustment",
    "hfa_component",
    "persistent_team_component_home",
    "qb_component_home",
    "personnel_component_home",
    "projected_home_margin",
    "fair_spread_home",
    "state_version",
]


@dataclass(frozen=True)
class V3Config:
    """Research defaults only. Parameters must earn promotion out of sample."""

    default_hfa: float = 2.20
    state_version: str = "V3.0_FOUNDATION"


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def validate_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    _require(
        schedule,
        ["season", "week", "game_id", "away_team", "home_team"],
        "schedule",
    )
    out = schedule.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    if out[["season", "week", "game_id"]].duplicated().any():
        raise ValueError("schedule contains duplicate game rows")
    return out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def initialize_state(
    teams: Iterable[str],
    season: int,
    preseason_priors: pd.DataFrame,
    config: V3Config | None = None,
) -> pd.DataFrame:
    """Create the state that exists immediately before Week 1.

    V3.0 deliberately keeps QB/personnel/market layers at zero. Later V3
    versions replace those placeholders without changing the walk-forward
    chronology.
    """
    config = config or V3Config()
    priors = preseason_priors.copy()
    _require(priors, ["team"], "preseason_priors")

    if "offense_strength" not in priors.columns:
        priors["offense_strength"] = 0.0
    if "defense_strength" not in priors.columns:
        priors["defense_strength"] = 0.0

    priors = priors.loc[priors["team"].isin(set(teams))].copy()
    if priors["team"].duplicated().any():
        raise ValueError("preseason_priors contains duplicate teams")

    missing = sorted(set(teams) - set(priors["team"]))
    if missing:
        raise ValueError(f"Missing preseason priors for teams: {missing}")

    state = priors[["team", "offense_strength", "defense_strength"]].copy()
    state["season"] = int(season)
    state["week"] = 1
    state["qb_value"] = 0.0
    state["personnel_adjustment"] = 0.0
    state["market_strength"] = 0.0
    state["state_version"] = config.state_version
    return state[STATE_COLUMNS].sort_values("team").reset_index(drop=True)


def project_week(
    schedule_week: pd.DataFrame,
    state: pd.DataFrame,
    config: V3Config | None = None,
) -> pd.DataFrame:
    """Freeze projections from the state entering a week.

    Positive projected_home_margin means the home team is favored.
    fair_spread_home uses sportsbook convention, so the sign is reversed.
    """
    config = config or V3Config()
    _require(schedule_week, ["season", "week", "game_id", "away_team", "home_team"], "schedule_week")
    _require(state, STATE_COLUMNS, "state")

    if schedule_week.empty:
        return pd.DataFrame(columns=PROJECTION_COLUMNS)

    seasons = schedule_week["season"].unique()
    weeks = schedule_week["week"].unique()
    if len(seasons) != 1 or len(weeks) != 1:
        raise ValueError("project_week requires exactly one season/week")

    season = int(seasons[0])
    week = int(weeks[0])
    entering = state.loc[(state["season"] == season) & (state["week"] == week)].copy()
    if entering.empty:
        raise ValueError(f"No entering state for {season} Week {week}")

    fields = [
        "team",
        "offense_strength",
        "defense_strength",
        "qb_value",
        "personnel_adjustment",
        "state_version",
    ]
    away = entering[fields].rename(columns={c: f"away_{c}" for c in fields if c != "team"} | {"team": "away_team"})
    home = entering[fields].rename(columns={c: f"home_{c}" for c in fields if c != "team"} | {"team": "home_team"})

    board = schedule_week.copy()
    board = board.merge(away, on="away_team", how="left", validate="many_to_one")
    board = board.merge(home, on="home_team", how="left", validate="many_to_one")

    needed = [
        "away_offense_strength", "home_offense_strength",
        "away_defense_strength", "home_defense_strength",
        "away_qb_value", "home_qb_value",
        "away_personnel_adjustment", "home_personnel_adjustment",
    ]
    if board[needed].isna().any().any():
        raise ValueError("At least one scheduled team is missing from the entering V3 state")

    neutral = pd.Series(False, index=board.index)
    if "location" in board.columns:
        neutral = board["location"].astype(str).str.lower().str.contains("neutral", na=False)
    board["hfa_component"] = np.where(neutral, 0.0, config.default_hfa)

    # Positive defense_strength means a better defense. A matchup projection is
    # therefore home offense - away defense minus (away offense - home defense).
    board["persistent_team_component_home"] = (
        board["home_offense_strength"]
        - board["away_defense_strength"]
        - board["away_offense_strength"]
        + board["home_defense_strength"]
    )
    board["qb_component_home"] = board["home_qb_value"] - board["away_qb_value"]
    board["personnel_component_home"] = (
        board["home_personnel_adjustment"] - board["away_personnel_adjustment"]
    )
    board["projected_home_margin"] = (
        board["persistent_team_component_home"]
        + board["qb_component_home"]
        + board["personnel_component_home"]
        + board["hfa_component"]
    )
    board["fair_spread_home"] = -board["projected_home_margin"]
    board["state_version"] = config.state_version
    return board[PROJECTION_COLUMNS].copy()


def run_walk_forward(
    schedule: pd.DataFrame,
    preseason_priors: pd.DataFrame,
    update_state: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int], pd.DataFrame],
    observations: pd.DataFrame | None = None,
    config: V3Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a deterministic season-by-season walk-forward simulation.

    The update callback is invoked only *after* a week's projections have been
    frozen. It receives observations through the completed week and must return
    the state entering week+1. This chronology is the V3 leakage barrier.
    """
    config = config or V3Config()
    sched = validate_schedule(schedule)
    observations = pd.DataFrame() if observations is None else observations.copy()

    all_states: list[pd.DataFrame] = []
    all_projections: list[pd.DataFrame] = []

    for season, season_schedule in sched.groupby("season", sort=True):
        weeks = sorted(season_schedule["week"].unique())
        teams = sorted(set(season_schedule["away_team"]) | set(season_schedule["home_team"]))
        state = initialize_state(teams, int(season), preseason_priors.loc[preseason_priors["season"].eq(season)] if "season" in preseason_priors.columns else preseason_priors, config)
        all_states.append(state.copy())

        for week in weeks:
            if week != 1:
                # state returned by the previous update must explicitly identify
                # the week it is safe to use.
                if not state["week"].eq(week).all():
                    raise ValueError(f"State chronology violation entering {season} Week {week}")

            sw = season_schedule.loc[season_schedule["week"].eq(week)].copy()
            frozen = project_week(sw, state, config)
            all_projections.append(frozen)

            if week == weeks[-1]:
                continue

            obs_through_week = observations.loc[
                (observations.get("season", pd.Series(dtype=int)).eq(season))
                & (observations.get("week", pd.Series(dtype=int)).le(week))
            ].copy() if not observations.empty else observations.copy()

            next_state = update_state(
                state.copy(),
                frozen.copy(),
                obs_through_week,
                int(season),
                int(week),
            ).copy()
            _require(next_state, STATE_COLUMNS, "updated state")
            next_state["season"] = int(season)
            next_state["week"] = int(week) + 1
            next_state["state_version"] = config.state_version
            if next_state["team"].duplicated().any():
                raise ValueError("updated state contains duplicate teams")
            state = next_state[STATE_COLUMNS].sort_values("team").reset_index(drop=True)
            all_states.append(state.copy())

    return (
        pd.concat(all_projections, ignore_index=True) if all_projections else pd.DataFrame(columns=PROJECTION_COLUMNS),
        pd.concat(all_states, ignore_index=True) if all_states else pd.DataFrame(columns=STATE_COLUMNS),
    )


def no_update_state(
    state: pd.DataFrame,
    frozen_projections: pd.DataFrame,
    observations_through_week: pd.DataFrame,
    season: int,
    completed_week: int,
) -> pd.DataFrame:
    """V3.0 harness smoke-test updater: carry state forward unchanged."""
    return state.copy()
