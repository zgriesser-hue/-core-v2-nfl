from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.v3 import V3Config, initialize_state, project_week, validate_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen Core V3.0 Week 1 board.")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--hfa", type=float, default=2.20)
    parser.add_argument("--out", type=Path, default=Path("data/research/v3/v3_week1_board_2026.csv"))
    args = parser.parse_args()

    schedule = validate_schedule(pd.read_csv(args.schedule))
    priors = pd.read_csv(args.priors)
    week = schedule.loc[(schedule["season"] == args.season) & (schedule["week"] == 1)].copy()
    teams = sorted(set(week["away_team"]) | set(week["home_team"]))
    config = V3Config(default_hfa=args.hfa, state_version="V3.0_PRESEASON_FOUNDATION")
    state = initialize_state(teams, args.season, priors, config)
    board = project_week(week, state, config)

    market = week[["game_id", "spread_line"]].copy() if "spread_line" in week.columns else pd.DataFrame()
    if not market.empty:
        market["market_home_spread"] = -pd.to_numeric(market["spread_line"], errors="coerce")
        board = board.merge(market[["game_id", "market_home_spread"]], on="game_id", how="left")
        board["v3_minus_market_home"] = board["fair_spread_home"] - board["market_home_spread"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False)
    print(board.to_string(index=False))
    print(f"\nSaved {len(board)} Week 1 projections to {args.out}")


if __name__ == "__main__":
    main()
