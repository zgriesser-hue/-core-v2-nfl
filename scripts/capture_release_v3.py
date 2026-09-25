"""
capture_release_v3.py - locks V3's board + market together at
projection release time, mirroring Core V2's own capture_release.py
exactly - it's the same operation, just pointed at a different
board.

WHERE THIS FILE GOES: this script imports `core_v2.tracking` and
`core_v2.market` directly, unchanged - so it belongs in the SAME
repo as `core_v2/`, `scripts/`, `app.py` (the "-core-v2-nfl-main"
repo), not in the standalone v3_model package. Copy it into that
repo's `scripts/` folder, alongside `capture_release.py`.

WHY THIS IS NEW CODE, NOT JUST A CALL TO THE EXISTING SCRIPT: the
mechanics (`schedule_with_kickoffs`, `build_release_records`,
`append_release_records`) are 100% reused from `core_v2.tracking`
UNCHANGED - confirmed by reading that module: it only needs a board
with `game_key`/`home_team`/`away_team`/`fair_home_spread` (or
`core_v2_home_spread`), which V3's board.py output already provides
after one rename (`fair_home_spread_v3` -> `fair_home_spread`; V3's
board.py has included `game_key` in Core V2's exact format since
Stage 3). What's new is: a separate `--history` file so V2 and V3
release histories never collide, and - the part that actually
matters - a SIGN FLIP, since `fair_home_spread_v3` is in margin
convention (positive = home favored) while Core V2's whole tracking
stack uses standard bookmaker convention (negative = home favored).
See `prepare_v3_board_for_release`'s docstring for the exact reason
this isn't optional.

WHAT THIS DOES NOT DO: fetch or build V3's board - that's board.py's
job, run first. This script only locks whatever board.py already
wrote, plus a fresh market pull, at release time.

Usage (mirrors capture_release.py exactly):

    python -m scripts.capture_release_v3 \\
        --season 2026 --week 5 \\
        --board week_05_v3_board.csv \\
        --schedule data/schedules/schedule_2026.csv \\
        --history data/tracking/v3_projection_releases.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.market import parse_spread_payload, request_spreads
from core_v2.tracking import (
    append_release_records,
    build_release_records,
    schedule_with_kickoffs,
)


def prepare_v3_board_for_release(v3_board: pd.DataFrame) -> pd.DataFrame:
    """
    Two V3-specific steps.

    1. SIGN FLIP (the important one): V3's fair_home_spread_v3 is in
       margin convention (positive = home favored, matching
       pred_v30/pred_v36 and nflverse's spread_line). Core V2's
       market_home_spread - and everything build_release_records
       compares against - is in standard BOOKMAKER convention
       (negative = home favored), parsed straight from raw sportsbook
       odds in core_v2/market.py. These are exact opposites for the
       same real-world fact; feeding V3's number in unflipped would
       silently invert every model_selected_team/side, edge, and
       downstream CLV sign. bookmaker_spread = -margin_spread exactly.
    2. game_key is already present (board.py, Stage 3) - nothing else
       to adapt.
    """
    board = v3_board.copy()
    if "fair_home_spread" not in board.columns:
        board["fair_home_spread"] = -board["fair_home_spread_v3"]
    return board


def lock_v3_release(
    board_path: Path,
    schedule_path: Path,
    history_path: Path,
    season: int,
    week: int,
    payload: object | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Core logic, separated from argparse/API-calling so it's directly
    testable (pass `payload` to skip the live API call, exactly like
    capture_release.py's own --payload fixture support).
    """
    v3_board = pd.read_csv(board_path)
    board = prepare_v3_board_for_release(v3_board)

    schedule = pd.read_csv(schedule_path, low_memory=False)
    games = schedule_with_kickoffs(schedule, season, week)
    now = pd.Timestamp.now(tz="UTC")
    future_keys = set(games.loc[games["kickoff_utc"].gt(now), "game_key"])
    release_board = board.loc[board["game_key"].isin(future_keys)].copy()

    if release_board.empty:
        return pd.DataFrame()

    keys = set(release_board["game_key"])
    if payload is None:
        api_key = api_key or os.environ.get("PARLAY_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("PARLAY_API_KEY is required to lock a V3 projection release")
        payload, _metadata = request_spreads(api_key)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    quotes = parse_spread_payload(payload, season, week, keys, timestamp)
    records = build_release_records(release_board, quotes, timestamp)

    existing = pd.read_csv(history_path) if Path(history_path).exists() else pd.DataFrame()
    history = append_release_records(existing, records)
    atomic_write_csv(history, history_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--history", type=Path, default=Path("data/tracking/v3_projection_releases.csv")
    )
    args = parser.parse_args()

    records = lock_v3_release(args.board, args.schedule, args.history, args.season, args.week)
    if records.empty:
        print("No future games remain on this board; no API request made")
        return
    print(f"Locked {len(records)} V3 model/market release records")


if __name__ == "__main__":
    main()
