from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.market import dashboard_lines, parse_spread_payload, request_spreads
from core_v2.tracking import (
    append_release_records,
    build_release_records,
    schedule_with_kickoffs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock model and market at projection release")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dashboard-lines", type=Path, required=True)
    args = parser.parse_args()

    board = pd.read_csv(args.board)
    schedule = pd.read_csv(args.schedule, low_memory=False)
    games = schedule_with_kickoffs(schedule, args.season, args.week)
    now = pd.Timestamp.now(tz="UTC")
    future_keys = set(games.loc[games["kickoff_utc"].gt(now), "game_key"])
    release_board = board.loc[board["game_key"].isin(future_keys)].copy()
    if release_board.empty:
        print("No future games remain on this board; no API request made")
        return
    keys = set(release_board["game_key"])
    api_key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PARLAY_API_KEY is required to lock a projection release")
    payload, metadata = request_spreads(api_key)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    quotes = parse_spread_payload(payload, args.season, args.week, keys, timestamp)
    records = build_release_records(release_board, quotes, timestamp)
    existing = pd.read_csv(args.history) if args.history.exists() else pd.DataFrame()
    history = append_release_records(existing, records)
    atomic_write_csv(history, args.history)
    fresh_lines = dashboard_lines(quotes)
    if args.dashboard_lines.exists():
        old_lines = pd.read_csv(args.dashboard_lines)
        active_matchups = set(fresh_lines["matchup"])
        old_lines = old_lines.loc[~old_lines["matchup"].isin(active_matchups)].copy()
        fresh_lines = pd.concat([old_lines, fresh_lines], ignore_index=True)
    atomic_write_csv(fresh_lines, args.dashboard_lines)
    print(f"Locked {len(records)} model/market release records")
    print(f"API requests used: {metadata.get('requests_used')}")
    print(f"API requests remaining: {metadata.get('requests_remaining')}")


if __name__ == "__main__":
    main()
