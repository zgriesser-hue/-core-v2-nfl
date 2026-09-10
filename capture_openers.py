from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.market import (
    append_first_seen_openers,
    build_consensus_openers,
    dashboard_lines,
    parse_spread_payload,
    request_spreads,
)
from core_v2.projections import game_key, load_week_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture immutable first-observed NFL openers from one API request"
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dashboard-output", type=Path)
    parser.add_argument("--dashboard-lines-output", type=Path)
    parser.add_argument("--payload", type=Path, help="Offline JSON fixture; skips API")
    args = parser.parse_args()

    schedule = load_week_schedule(args.schedule, args.season, args.week)
    allowed = {
        game_key(args.season, args.week, row.away_team, row.home_team)
        for row in schedule.itertuples()
    }
    if args.payload:
        payload = json.loads(args.payload.read_text())
        metadata = {"requests_used": "offline", "requests_remaining": "offline"}
    else:
        api_key = os.environ.get("PARLAY_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("PARLAY_API_KEY is required")
        payload, metadata = request_spreads(api_key)

    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    quotes = parse_spread_payload(payload, args.season, args.week, allowed, snapshot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    opener_path = args.output_dir / "book_openers.csv"
    existing = pd.read_csv(opener_path) if opener_path.exists() else pd.DataFrame()

    kickoff_column = next(
        (column for column in ("gameday", "kickoff_utc", "start_time") if column in schedule),
        None,
    )
    kickoff_by_game = {}
    for row in schedule.itertuples():
        key = game_key(args.season, args.week, row.away_team, row.home_team)
        kickoff_by_game[key] = getattr(row, kickoff_column) if kickoff_column else pd.NA

    updated = append_first_seen_openers(existing, quotes, kickoff_by_game)
    consensus = build_consensus_openers(updated)
    atomic_write_csv(updated, opener_path)
    atomic_write_csv(consensus, args.output_dir / "consensus_openers.csv")
    if args.dashboard_output:
        active = consensus.loc[consensus["game_key"].isin(allowed)].copy()
        atomic_write_csv(active, args.dashboard_output)
    if args.dashboard_lines_output:
        atomic_write_csv(dashboard_lines(quotes), args.dashboard_lines_output)

    print(f"Valid quotes parsed: {len(quotes)}")
    print(f"Immutable book openers: {len(updated)}")
    print(f"Consensus games: {len(consensus)}")
    print(f"API requests used: {metadata.get('requests_used')}")
    print(f"API requests remaining: {metadata.get('requests_remaining')}")


if __name__ == "__main__":
    main()
