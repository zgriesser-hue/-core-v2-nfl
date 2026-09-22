from __future__ import annotations

import argparse
from pathlib import Path

from core_v2.io import atomic_write_csv
from core_v2.qb_efficiency import build_qb_game_panel, fetch_qb_pbp, solve_oa_qb_epa


def main() -> None:
    parser = argparse.ArgumentParser(description="Build walk-forward opponent-adjusted QB EPA research signal")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-dropbacks", type=float, default=120.0)
    parser.add_argument("--defense-prior-dropbacks", type=float, default=250.0)
    args = parser.parse_args()

    through_week = max(0, args.week - 1)
    pbp = fetch_qb_pbp(args.season)
    games = build_qb_game_panel(pbp, args.season, through_week)
    ratings = solve_oa_qb_epa(
        games,
        prior_dropbacks=args.prior_dropbacks,
        defense_prior_dropbacks=args.defense_prior_dropbacks,
    )
    ratings["performance_data_through_week"] = through_week
    ratings["qb_model"] = "OA_QB_EPA_RESEARCH_V1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(games, args.output_dir / "qb_game_efficiency.csv")
    atomic_write_csv(ratings, args.output_dir / "oa_qb_epa_ratings.csv")

    print(f"QB OA EPA research signal built for {args.season} Week {args.week}")
    print(f"Performance data through Week {through_week}")
    print(ratings.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
