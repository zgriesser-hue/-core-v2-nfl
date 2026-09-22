from __future__ import annotations

import numpy as np
import pandas as pd

from .projections import game_key, normalize_team


DEFAULT_LEARNING_RATE = 0.10
DEFAULT_TEAM_GAME_CAP = 0.75
DEFAULT_OPPONENT_SHRINKAGE = 0.50


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



def update_team_ratings_opponent_adjusted(
    ratings: pd.DataFrame,
    grades: pd.DataFrame,
    season: int,
    through_week: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    team_game_cap: float = DEFAULT_TEAM_GAME_CAP,
    opponent_shrinkage: float = DEFAULT_OPPONENT_SHRINKAGE,
) -> pd.DataFrame:
    """Shadow rating update that shrinks residuals using opponent reliability.

    The released projection already prices opponent strength, so this function
    deliberately does *not* add opponent rating to the game residual again.
    Instead it downweights a result when the opponent's observed in-season
    performance signal is still weak/noisy, and approaches the normal residual
    update as the opponent accumulates games.

    This is intended for shadow/backtest use until validated.
    """
    if not 0.0 <= opponent_shrinkage <= 1.0:
        raise ValueError("opponent_shrinkage must be between 0 and 1")

    required = {
        "season", "week", "game_key", "away_team", "home_team",
        "actual_home_margin", "projected_home_margin",
    }
    missing = required - set(grades.columns)
    if missing:
        raise ValueError(f"Grades missing columns: {sorted(missing)}")

    result = ratings.copy()
    result["team"] = result["team"].map(normalize_team)
    components = [
        "base_prior", "qb_adjustment",
        "manual_personnel_adjustment", "manual_injury_adjustment",
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
    ].drop_duplicates("game_key", keep="last").copy()
    for column in ["actual_home_margin", "projected_home_margin"]:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    eligible = eligible.dropna(
        subset=["actual_home_margin", "projected_home_margin"]
    ).sort_values(["week", "game_key"])
    eligible["home_team"] = eligible["home_team"].map(normalize_team)
    eligible["away_team"] = eligible["away_team"].map(normalize_team)

    adjustments = {team: 0.0 for team in result["team"]}
    games_seen = {team: 0 for team in result["team"]}

    for row in eligible.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        residual = float(row.actual_home_margin) - float(row.projected_home_margin)

        # Reliability is based only on games available before this observation.
        # With shrinkage=0 this exactly reproduces the current update. At 1.0,
        # an opponent with zero prior games contributes 50% weight, one prior
        # game 75%, two 83.3%, etc. This avoids pretending Week 1 is as
        # informative as a later-season result.
        home_rel = (games_seen.get(home, 0) + 1.0) / (games_seen.get(home, 0) + 2.0)
        away_rel = (games_seen.get(away, 0) + 1.0) / (games_seen.get(away, 0) + 2.0)
        home_weight = 1.0 - opponent_shrinkage * (1.0 - away_rel)
        away_weight = 1.0 - opponent_shrinkage * (1.0 - home_rel)

        raw_home = learning_rate * residual / 2.0 * home_weight
        raw_away = -learning_rate * residual / 2.0 * away_weight

        # Preserve league-wide zero sum game by game while retaining the
        # opponent-reliability shrinkage in the magnitude.
        magnitude = (abs(raw_home) + abs(raw_away)) / 2.0
        magnitude = min(float(team_game_cap), magnitude)
        sign = 1.0 if residual >= 0 else -1.0
        adjustments[home] = adjustments.get(home, 0.0) + sign * magnitude
        adjustments[away] = adjustments.get(away, 0.0) - sign * magnitude
        games_seen[home] = games_seen.get(home, 0) + 1
        games_seen[away] = games_seen.get(away, 0) + 1

    result["performance_adjustment"] = result["team"].map(adjustments).fillna(0.0)
    result["games_in_performance_update"] = result["team"].map(games_seen).fillna(0).astype(int)
    result["performance_update_through_week"] = int(through_week)
    result["performance_learning_rate"] = float(learning_rate)
    result["performance_team_game_cap"] = float(team_game_cap)
    result["performance_opponent_shrinkage"] = float(opponent_shrinkage)
    result["current_team_rating"] = (
        result["rating_before_performance"] + result["performance_adjustment"]
    )
    result["rank"] = result["current_team_rating"].rank(
        method="min", ascending=False
    ).astype(int)
    if not np.isclose(result["performance_adjustment"].sum(), 0.0, atol=1e-10):
        raise RuntimeError("Performance adjustments must remain league-wide zero-sum")
    return result.sort_values("team").reset_index(drop=True)
