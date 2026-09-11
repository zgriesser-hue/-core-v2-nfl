from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.market import parse_spread_payload, request_spreads
from core_v2.tracking import append_closing_snapshot, due_closing_games, schedule_with_kickoffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture immutable pre-kickoff closing spreads")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracking-consensus",
        type=Path,
        help="Optional season-wide closing consensus history used by grading",
    )
    parser.add_argument("--minimum-lead", type=float, default=5.0)
    parser.add_argument("--maximum-lead", type=float, default=50.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--payload", type=Path, help="Offline JSON fixture")
    args = parser.parse_args()

    schedule = pd.read_csv(args.schedule, low_memory=False)
    games = schedule_with_kickoffs(schedule, args.season, args.week)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus_path = args.output_dir / "closing_consensus.csv"
    books_path = args.output_dir / "closing_book_lines.csv"
    existing_consensus = pd.read_csv(consensus_path) if consensus_path.exists() else pd.DataFrame()
    existing_books = pd.read_csv(books_path) if books_path.exists() else pd.DataFrame()
    due = due_closing_games(
        games, existing_consensus,
        minimum_lead_minutes=args.minimum_lead,
        maximum_lead_minutes=args.maximum_lead,
        force=args.force,
    )
    if due.empty:
        print("No uncovered games are inside the closing capture window; no API request made")
        return

    if args.payload:
        payload = json.loads(args.payload.read_text())
        metadata = {"requests_used": "offline", "requests_remaining": "offline"}
    else:
        api_key = os.environ.get("PARLAY_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("PARLAY_API_KEY is required when games are due")
        payload, metadata = request_spreads(api_key)
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    quotes = parse_spread_payload(
        payload, args.season, args.week, set(due["game_key"]), snapshot
    )
    books, consensus = append_closing_snapshot(
        existing_books, existing_consensus, quotes, due
    )
    atomic_write_csv(books, books_path)
    atomic_write_csv(consensus, consensus_path)
    if args.tracking_consensus:
        tracking = (
            pd.read_csv(args.tracking_consensus)
            if args.tracking_consensus.exists()
            else pd.DataFrame()
        )
        new_rows = consensus.loc[
            ~consensus["game_key"].astype(str).isin(
                set(tracking.get("game_key", pd.Series(dtype=str)).astype(str))
            )
        ].copy()
        tracking = pd.concat([tracking, new_rows], ignore_index=True)
        if tracking.duplicated("game_key").any():
            raise RuntimeError("Tracking closing consensus contains duplicate games")
        atomic_write_csv(
            tracking.sort_values(["season", "week", "game_key"]),
            args.tracking_consensus,
        )
    raw_dir = args.output_dir / "closing_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"closing_{snapshot.replace(':', '').replace('+', '_')}.json"
    raw_path.write_text(json.dumps(payload, indent=2))
    print(f"Due games: {len(due)}; captured consensus closes: {len(consensus) - len(existing_consensus)}")
    print(f"API requests used: {metadata.get('requests_used')}")
    print(f"API requests remaining: {metadata.get('requests_remaining')}")


if __name__ == "__main__":
    main()
