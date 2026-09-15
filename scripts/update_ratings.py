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
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--team-game-cap", type=float, default=DEFAULT_TEAM_GAME_CAP)
    args = parser.parse_args()

    schedule = pd.read_csv(args.schedule, low_memory=False)
    grades = pd.read_csv(args.grades, low_memory=False)
    ratings = pd.read_csv(args.ratings)
    through_week = args.through_week
    if through_week is None:
        through_week = latest_fully_graded_week(schedule, grades, args.season)
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
    output = args.output or args.ratings
    atomic_write_csv(updated, output)
    print(
        f"Updated {len(updated)} teams through Week {through_week}; "
        f"mean absolute adjustment={updated['performance_adjustment'].abs().mean():.3f}; "
        f"maximum={updated['performance_adjustment'].abs().max():.3f}"
    )


if __name__ == "__main__":
    main()
