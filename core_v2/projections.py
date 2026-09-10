from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectionConfig, TEAM_MAP
from .io import clean_bool


def normalize_team(value: object) -> str:
    team = str(value).strip().upper()
    return TEAM_MAP.get(team, team)


def game_key(season: int, week: int, away: str, home: str) -> str:
    return f"{int(season)}_W{int(week)}_{away}_{home}"


def load_week_schedule(path: Path, season: int, week: int) -> pd.DataFrame:
    schedule = pd.read_csv(path, low_memory=False)
    required = {"season", "week", "away_team", "home_team"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {sorted(missing)}")
    schedule = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(week))
    ].copy()
    if "game_type" in schedule.columns:
        schedule = schedule.loc[schedule["game_type"].astype(str).eq("REG")]
    schedule["away_team"] = schedule["away_team"].map(normalize_team)
    schedule["home_team"] = schedule["home_team"].map(normalize_team)
    if schedule.empty:
        raise ValueError(f"No schedule rows found for {season} Week {week}")
    if schedule[["away_team", "home_team"]].duplicated().any():
        raise ValueError("Duplicate matchups found in weekly schedule")
    return schedule.reset_index(drop=True)


def load_ratings(path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(path)
    required = {"team", "current_team_rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings missing columns: {sorted(missing)}")
    ratings["team"] = ratings["team"].map(normalize_team)
    ratings["current_team_rating"] = pd.to_numeric(
        ratings["current_team_rating"], errors="coerce"
    )
    if ratings["team"].duplicated().any():
        raise ValueError("Ratings contain duplicate teams")
    if ratings["current_team_rating"].isna().any():
        raise ValueError("Ratings contain missing current_team_rating values")
    return ratings


def load_personnel(path: Path) -> pd.DataFrame:
    personnel = pd.read_csv(path)
    required = {
        "away_team",
        "home_team",
        "personnel_correction_current_clean",
        "personnel_correction_min_clean",
        "personnel_correction_max_clean",
        "report_coverage_complete_clean",
    }
    missing = required - set(personnel.columns)
    if missing:
        raise ValueError(f"Personnel input missing columns: {sorted(missing)}")
    personnel["away_team"] = personnel["away_team"].map(normalize_team)
    personnel["home_team"] = personnel["home_team"].map(normalize_team)
    if personnel[["away_team", "home_team"]].duplicated().any():
        raise ValueError("Personnel input contains duplicate matchups")
    for column in [
        "personnel_correction_current_clean",
        "personnel_correction_min_clean",
        "personnel_correction_max_clean",
    ]:
        personnel[column] = pd.to_numeric(personnel[column], errors="coerce")
    personnel["personnel_info_complete"] = personnel[
        "report_coverage_complete_clean"
    ].map(clean_bool)
    return personnel


def build_projection_board(
    schedule: pd.DataFrame,
    ratings: pd.DataFrame,
    personnel: pd.DataFrame,
    season: int,
    week: int,
    config: ProjectionConfig | None = None,
    historical_locks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble fair spreads from frozen team ratings, HFA and personnel inputs."""
    config = config or ProjectionConfig()
    board = schedule.copy()

    away = ratings[["team", "current_team_rating"]].rename(
        columns={
            "team": "away_team",
            "current_team_rating": "away_team_rating",
        }
    )
    home = ratings[["team", "current_team_rating"]].rename(
        columns={
            "team": "home_team",
            "current_team_rating": "home_team_rating",
        }
    )
    board = board.merge(away, on="away_team", how="left", validate="many_to_one")
    board = board.merge(home, on="home_team", how="left", validate="many_to_one")
    if board[["away_team_rating", "home_team_rating"]].isna().any().any():
        raise ValueError("At least one scheduled team is missing a production rating")

    neutral = pd.Series(False, index=board.index)
    if "location" in board.columns:
        neutral = board["location"].astype(str).str.lower().str.contains("neutral")
    board["neutral_site"] = neutral
    board["hfa_component"] = np.where(neutral, 0.0, config.default_hfa)
    board["team_strength_component_home"] = (
        board["home_team_rating"] - board["away_team_rating"]
    )
    board["projected_home_margin_pre_personnel"] = (
        board["team_strength_component_home"] + board["hfa_component"]
    )
    board["fair_spread_pre_personnel"] = -board[
        "projected_home_margin_pre_personnel"
    ]
    board["game_key"] = board.apply(
        lambda row: game_key(
            season, week, row["away_team"], row["home_team"]
        ),
        axis=1,
    )

    personnel_fields = [
        "away_team",
        "home_team",
        "personnel_correction_current_clean",
        "personnel_correction_min_clean",
        "personnel_correction_max_clean",
        "personnel_info_complete",
    ]
    board = board.merge(
        personnel[personnel_fields],
        on=["away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )
    board["personnel_info_complete"] = board[
        "personnel_info_complete"
    ].fillna(False).astype(bool)

    complete = board["personnel_info_complete"] & board[
        "personnel_correction_current_clean"
    ].notna()
    board["fair_spread_home"] = np.where(
        complete,
        -(
            board["projected_home_margin_pre_personnel"]
            + board["personnel_correction_current_clean"]
        ),
        board["fair_spread_pre_personnel"],
    )
    board["fair_spread_low"] = np.where(
        complete,
        -(
            board["projected_home_margin_pre_personnel"]
            + board["personnel_correction_max_clean"]
        ),
        board["fair_spread_pre_personnel"],
    )
    board["fair_spread_high"] = np.where(
        complete,
        -(
            board["projected_home_margin_pre_personnel"]
            + board["personnel_correction_min_clean"]
        ),
        board["fair_spread_pre_personnel"],
    )
    board["personnel_projection_status"] = np.where(
        complete, "UPDATED", "PROVISIONAL"
    )
    board["used_personnel_fallback"] = ~complete
    board["projection_display_status"] = np.where(
        complete, "UPDATED_PREGAME", "EARLY_PROVISIONAL"
    )

    if historical_locks is not None and not historical_locks.empty:
        locks = historical_locks.copy()
        locks["fair_spread_home_locked"] = pd.to_numeric(
            locks["fair_spread_home_locked"], errors="raise"
        )
        board = board.merge(
            locks[
                [
                    "game_key",
                    "fair_spread_home_locked",
                    "personnel_projection_status_locked",
                    "projection_display_status_locked",
                ]
            ],
            on="game_key",
            how="left",
            validate="one_to_one",
        )
        locked = board["fair_spread_home_locked"].notna()
        board.loc[locked, "fair_spread_home"] = board.loc[
            locked, "fair_spread_home_locked"
        ]
        board.loc[locked, "fair_spread_low"] = board.loc[
            locked, "fair_spread_home_locked"
        ]
        board.loc[locked, "fair_spread_high"] = board.loc[
            locked, "fair_spread_home_locked"
        ]
        board.loc[locked, "personnel_projection_status"] = board.loc[
            locked, "personnel_projection_status_locked"
        ]
        board.loc[locked, "projection_display_status"] = board.loc[
            locked, "projection_display_status_locked"
        ]
        board.loc[locked, "used_personnel_fallback"] = False
        board.loc[locked, "personnel_info_complete"] = True

    board["fair_spread_uncertainty_width"] = (
        board["fair_spread_high"] - board["fair_spread_low"]
    )
    for column in [
        "fair_spread_pre_personnel",
        "fair_spread_home",
        "fair_spread_low",
        "fair_spread_high",
        "fair_spread_uncertainty_width",
    ]:
        board[column] = board[column].round(config.line_round_digits)

    if board["game_key"].duplicated().any():
        raise ValueError("Projection output contains duplicate game keys")
    if board["fair_spread_home"].isna().any():
        raise ValueError("Projection output contains missing fair spreads")
    return board.reset_index(drop=True)


def build_from_files(
    schedule_path: Path,
    ratings_path: Path,
    personnel_path: Path,
    season: int,
    week: int,
    locks_path: Path | None = None,
) -> pd.DataFrame:
    locks = None
    if locks_path is not None and Path(locks_path).exists():
        locks = pd.read_csv(locks_path)
    return build_projection_board(
        load_week_schedule(schedule_path, season, week),
        load_ratings(ratings_path),
        load_personnel(personnel_path),
        season,
        week,
        historical_locks=locks,
    )

