from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve the next NFL week whose first kickoff is future")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument(
        "--mode",
        choices=["next", "active"],
        default="next",
        help="next uses first kickoff; active retains a week until its final kickoff",
    )
    parser.add_argument(
        "--floor-board",
        type=Path,
        help="Never resolve an earlier week than the current live board for this season",
    )
    args = parser.parse_args()
    schedule = pd.read_csv(args.schedule, low_memory=False)
    schedule = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == args.season)
        & schedule["game_type"].astype(str).eq("REG")
    ].copy()
    kickoff = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("00:00").astype(str),
        utc=True,
        errors="coerce",
    )
    grouped = kickoff.groupby(schedule["week"])
    boundary = grouped.min() if args.mode == "next" else grouped.max()
    future = boundary.sort_values().loc[lambda values: values > datetime.now(timezone.utc)]
    if future.empty:
        raise RuntimeError("No future regular-season week found")
    resolved_week = int(future.index[0])
    if args.floor_board is not None and args.floor_board.exists():
        live = pd.read_csv(args.floor_board, usecols=["season", "week"])
        live = live.loc[pd.to_numeric(live["season"], errors="coerce").eq(args.season)]
        if not live.empty:
            live_weeks = pd.to_numeric(live["week"], errors="coerce").dropna()
            if live_weeks.empty or live_weeks.nunique() != 1:
                raise ValueError("Current live board has no single valid week")
            resolved_week = max(resolved_week, int(live_weeks.iloc[0]))
    print(resolved_week)


if __name__ == "__main__":
    main()
