from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.ratings import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_TEAM_GAME_CAP,
    latest_fully_graded_week,
    update_team_ratings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Core V2 team ratings through the latest fully graded week"
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through-week", type=int)
    parser.add_argument(
        "--include-partial-latest-week",
        action="store_true",
        help="Include graded games from the newest incomplete week",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--team-game-cap", type=float, default=DEFAULT_TEAM_GAME_CAP)
    args = parser.parse_args()

    schedule = pd.read_csv(args.schedule, low_memory=False)
    grades = pd.read_csv(args.grades, low_memory=False)
    ratings = pd.read_csv(args.ratings)
    through_week = args.through_week
    complete_week = latest_fully_graded_week(schedule, grades, args.season)
    if through_week is None:
        through_week = complete_week
        if args.include_partial_latest_week:
            graded_weeks = pd.to_numeric(
                grades.loc[
                    pd.to_numeric(grades["season"], errors="coerce").eq(
                        args.season
                    ),
                    "week",
                ],
                errors="coerce",
            ).dropna()
            if not graded_weeks.empty:
                through_week = max(through_week, int(graded_weeks.max()))
    if through_week < 1:
        print("No fully graded regular-season week; ratings unchanged")
        return

    updated = update_team_ratings(
        ratings,
        grades,
        season=args.season,
        through_week=through_week,
        learning_rate=args.learning_rate,
        team_game_cap=args.team_game_cap,
    )
    scheduled_week = schedule.loc[
        pd.to_numeric(schedule["season"], errors="coerce").eq(args.season)
        & pd.to_numeric(schedule["week"], errors="coerce").eq(through_week)
    ].copy()
    if "game_type" in scheduled_week.columns:
        scheduled_week = scheduled_week.loc[
            scheduled_week["game_type"].astype(str).eq("REG")
        ]
    graded_week = grades.loc[
        pd.to_numeric(grades["season"], errors="coerce").eq(args.season)
        & pd.to_numeric(grades["week"], errors="coerce").eq(through_week)
    ]
    games_scheduled = len(scheduled_week)
    games_graded = graded_week["game_key"].nunique()
    update_status = (
        "COMPLETE" if games_scheduled and games_graded >= games_scheduled
        else "PARTIAL"
    )
    updated["performance_complete_through_week"] = int(complete_week)
    updated["performance_update_status"] = update_status
    updated["latest_week_games_graded"] = int(games_graded)
    updated["latest_week_games_scheduled"] = int(games_scheduled)
    output = args.output or args.ratings
    atomic_write_csv(updated, output)
    print(
        f"Updated {len(updated)} teams through Week {through_week}; "
        f"mean absolute adjustment={updated['performance_adjustment'].abs().mean():.3f}; "
        f"maximum={updated['performance_adjustment'].abs().max():.3f}; "
        f"status={update_status} ({games_graded}/{games_scheduled} games)"
    )


if __name__ == "__main__":
    main()
