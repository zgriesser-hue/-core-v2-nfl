from __future__ import annotations

import argparse
from pathlib import Path

from core_v2.io import atomic_write_csv
from core_v2.projections import build_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the current Streamlit projection board")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--locks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("current_live_board.csv"))
    args = parser.parse_args()

    projections = build_from_files(
        args.schedule, args.ratings, args.personnel, args.season, args.week, args.locks
    )
    if {"away_score", "home_score"}.issubset(projections.columns):
        completed = projections["away_score"].notna() & projections["home_score"].notna()
        projections = projections.loc[~completed].copy()
    projections = projections.loc[
        ~projections["projection_display_status"].eq("FINAL_PREGAME")
    ].copy()
    board = projections[
        [
            "season", "week", "game_key", "away_team", "home_team",
            "fair_spread_home", "fair_spread_low", "fair_spread_high",
            "fair_spread_uncertainty_width", "projection_display_status",
            "personnel_projection_status", "used_personnel_fallback",
            "personnel_info_complete", "qb_availability_provisional",
        ]
    ].rename(columns={"fair_spread_home": "fair_home_spread"})
    board["core_v2_home_spread"] = board["fair_home_spread"]
    board["matchup"] = board["away_team"] + " @ " + board["home_team"]
    atomic_write_csv(board, args.output)
    print(f"Published {len(board)} games to {args.output}")


if __name__ == "__main__":
    main()
