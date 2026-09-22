from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.efficiency import (
    build_shadow_ratings,
    build_team_game_epa,
    fetch_nflverse_pbp,
    solve_oa_epa,
)
from core_v2.io import atomic_write_csv
from core_v2.projections import build_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2026 OA EPA shadow ratings and projection board")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--locks", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Week N may only use Weeks < N. This is the core no-leakage guard.
    through_week = max(0, int(args.week) - 1)
    ratings = pd.read_csv(args.ratings)
    if "rating_before_performance" not in ratings.columns:
        components = [
            "base_prior",
            "qb_adjustment",
            "manual_personnel_adjustment",
            "manual_injury_adjustment",
        ]
        ratings["rating_before_performance"] = ratings[components].sum(axis=1)

    pbp = fetch_nflverse_pbp(args.season)
    team_games = build_team_game_epa(pbp, args.season, through_week)
    oa = solve_oa_epa(team_games)
    shadow = build_shadow_ratings(ratings, oa)
    shadow["performance_data_through_week"] = through_week

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratings_out = args.output_dir / "oa_epa_shadow_ratings.csv"
    atomic_write_csv(shadow, ratings_out)

    projection_ratings = shadow[["team", "shadow_team_rating"]].rename(
        columns={"shadow_team_rating": "current_team_rating"}
    )
    temp_ratings = args.output_dir / "_shadow_projection_ratings.csv"
    atomic_write_csv(projection_ratings, temp_ratings)
    try:
        board = build_from_files(
            schedule_path=args.schedule,
            ratings_path=temp_ratings,
            personnel_path=args.personnel,
            season=args.season,
            week=args.week,
            locks_path=None,  # shadow must never inherit official historical spread locks
        )
    finally:
        temp_ratings.unlink(missing_ok=True)

    board = board.merge(
        shadow[
            [
                "team",
                "oa_offense",
                "oa_defense",
                "oa_net",
                "games",
                "epa_weight",
                "rating_before_performance",
                "shadow_team_rating",
                "shadow_minus_structural",
            ]
        ].rename(columns=lambda c: f"home_{c}" if c != "team" else "home_team"),
        on="home_team",
        how="left",
        validate="many_to_one",
    )
    board = board.merge(
        shadow[
            [
                "team",
                "oa_offense",
                "oa_defense",
                "oa_net",
                "games",
                "epa_weight",
                "rating_before_performance",
                "shadow_team_rating",
                "shadow_minus_structural",
            ]
        ].rename(columns=lambda c: f"away_{c}" if c != "team" else "away_team"),
        on="away_team",
        how="left",
        validate="many_to_one",
    )
    board["shadow_model"] = "OA_EPA_RAMP_V1"
    board["performance_data_through_week"] = through_week
    atomic_write_csv(board, args.output_dir / "oa_epa_shadow_board.csv")

    print(f"OA EPA shadow built for {args.season} Week {args.week}")
    print(f"Performance data through Week {through_week}")
    print(shadow[["team", "oa_net", "games", "epa_weight", "shadow_team_rating", "shadow_minus_structural"]].to_string(index=False))


if __name__ == "__main__":
    main()
