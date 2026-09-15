from __future__ import annotations

import numpy as np
import pandas as pd

from .projections import game_key, normalize_team


DEFAULT_LEARNING_RATE = 0.10
DEFAULT_TEAM_GAME_CAP = 0.75


def latest_fully_graded_week(
    schedule: pd.DataFrame,
    grades: pd.DataFrame,
    season: int,
) -> int:
    """Return the latest week for which every scheduled game has a grade."""
    scheduled = schedule.copy()
    scheduled = scheduled.loc[
        pd.to_numeric(scheduled["season"], errors="coerce").eq(int(season))
    ]
    if "game_type" in scheduled.columns:
        scheduled = scheduled.loc[scheduled["game_type"].astype(str).eq("REG")]
    if "game_key" not in scheduled.columns:
        scheduled["away_team"] = scheduled["away_team"].map(normalize_team)
        scheduled["home_team"] = scheduled["home_team"].map(normalize_team)
        scheduled["game_key"] = scheduled.apply(
            lambda row: game_key(
                int(row["season"]),
                int(row["week"]),
                row["away_team"],
                row["home_team"],
            ),
            axis=1,
        )

    graded_keys = set(
        grades.loc[
            pd.to_numeric(grades["season"], errors="coerce").eq(int(season)),
            "game_key",
        ].astype(str)
    )
    complete_weeks: list[int] = []
    for week, games in scheduled.groupby("week"):
        keys = set(games["game_key"].astype(str))
        if keys and keys.issubset(graded_keys):
            complete_weeks.append(int(week))
    return max(complete_weeks, default=0)


def update_team_ratings(
    ratings: pd.DataFrame,
    grades: pd.DataFrame,
    season: int,
    through_week: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    team_game_cap: float = DEFAULT_TEAM_GAME_CAP,
) -> pd.DataFrame:
    """Recompute auditable, zero-sum postgame updates from the structural rating.

    Each game moves the two teams equally in opposite directions. The update is
    based on the final-margin residual against the released Core V2 projection,
    shrunk by ``learning_rate`` and capped per team/game. Recomputing from the
    structural components makes the operation idempotent and preserves later QB
    or manual changes.
    """
    required_ratings = {
        "team",
        "base_prior",
        "qb_adjustment",
        "manual_personnel_adjustment",
        "manual_injury_adjustment",
    }
    required_grades = {
        "season",
        "week",
        "game_key",
        "away_team",
        "home_team",
        "actual_home_margin",
        "projected_home_margin",
    }
    missing_ratings = required_ratings - set(ratings.columns)
    missing_grades = required_grades - set(grades.columns)
    if missing_ratings:
        raise ValueError(f"Ratings missing columns: {sorted(missing_ratings)}")
    if missing_grades:
        raise ValueError(f"Grades missing columns: {sorted(missing_grades)}")
    if not 0 <= learning_rate <= 1:
        raise ValueError("learning_rate must be between 0 and 1")
    if team_game_cap <= 0:
        raise ValueError("team_game_cap must be positive")

    result = ratings.copy()
    result["team"] = result["team"].map(normalize_team)
    if result["team"].duplicated().any():
        raise ValueError("Ratings contain duplicate teams")

    components = [
        "base_prior",
        "qb_adjustment",
        "manual_personnel_adjustment",
        "manual_injury_adjustment",
    ]
    for column in components:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["rating_before_performance"] = result[components].sum(axis=1)

    eligible = grades.copy()
    eligible["season"] = pd.to_numeric(eligible["season"], errors="coerce")
    eligible["week"] = pd.to_numeric(eligible["week"], errors="coerce")
    eligible = eligible.loc[
        eligible["season"].eq(int(season))
        & eligible["week"].le(int(through_week))
    ].copy()
    eligible = eligible.drop_duplicates("game_key", keep="last")
    eligible["actual_home_margin"] = pd.to_numeric(
        eligible["actual_home_margin"], errors="coerce"
    )
    eligible["projected_home_margin"] = pd.to_numeric(
        eligible["projected_home_margin"], errors="coerce"
    )
    eligible = eligible.dropna(
        subset=["actual_home_margin", "projected_home_margin"]
    )
    eligible["home_team"] = eligible["home_team"].map(normalize_team)
    eligible["away_team"] = eligible["away_team"].map(normalize_team)
    eligible["margin_residual_home"] = (
        eligible["actual_home_margin"] - eligible["projected_home_margin"]
    )
    eligible["home_adjustment"] = (
        learning_rate * eligible["margin_residual_home"] / 2.0
    ).clip(-team_game_cap, team_game_cap)

    home = eligible[["home_team", "home_adjustment"]].rename(
        columns={"home_team": "team", "home_adjustment": "adjustment"}
    )
    away = eligible[["away_team", "home_adjustment"]].rename(
        columns={"away_team": "team", "home_adjustment": "adjustment"}
    )
    away["adjustment"] = -away["adjustment"]
    adjustments = pd.concat([home, away], ignore_index=True)
    totals = adjustments.groupby("team", as_index=False).agg(
        performance_adjustment=("adjustment", "sum"),
        games_in_performance_update=("adjustment", "size"),
    )

    result = result.drop(
        columns=[
            "performance_adjustment",
            "games_in_performance_update",
            "performance_update_through_week",
            "performance_learning_rate",
            "performance_team_game_cap",
        ],
        errors="ignore",
    ).merge(totals, on="team", how="left", validate="one_to_one")
    result["performance_adjustment"] = result[
        "performance_adjustment"
    ].fillna(0.0)
    result["games_in_performance_update"] = result[
        "games_in_performance_update"
    ].fillna(0).astype(int)
    result["performance_update_through_week"] = int(through_week)
    result["performance_learning_rate"] = float(learning_rate)
    result["performance_team_game_cap"] = float(team_game_cap)
    result["current_team_rating"] = (
        result["rating_before_performance"]
        + result["performance_adjustment"]
    )
    result["rank"] = result["current_team_rating"].rank(
        method="min", ascending=False
    ).astype(int)

    if not np.isclose(result["performance_adjustment"].sum(), 0.0, atol=1e-10):
        raise RuntimeError("Performance adjustments must remain league-wide zero-sum")
    return result.sort_values("team").reset_index(drop=True)
