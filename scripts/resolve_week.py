from __future__ import annotations

import argparse
from datetime import datetime, timezone

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
    print(int(future.index[0]))


if __name__ == "__main__":
    main()
