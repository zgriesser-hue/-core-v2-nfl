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
    append_market_history,
    build_consensus_openers,
    build_market_consensus,
    dashboard_lines,
    latest_market_snapshot,
    parse_market_payload,
    request_markets,
)
from core_v2.tracking import schedule_with_kickoffs


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def append_projection_snapshots(
    existing: pd.DataFrame,
    board: pd.DataFrame,
    consensus: pd.DataFrame,
    snapshot_timestamp: str,
) -> pd.DataFrame:
    columns = [
        "season",
        "week",
        "game_key",
        "away_team",
        "home_team",
        "fair_home_spread",
        "core_v2_home_spread",
        "projection_display_status",
        "personnel_projection_status",
        "personnel_info_complete",
    ]
    available = [column for column in columns if column in board.columns]
    captured = board[available].copy()
    if "fair_home_spread" not in captured and "core_v2_home_spread" in captured:
        captured["fair_home_spread"] = captured["core_v2_home_spread"]
    captured["market_snapshot_timestamp_et"] = snapshot_timestamp
    captured["projection_snapshot_timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")
    captured = captured.merge(
        consensus[
            [
                "game_key",
                "market_home_spread",
                "market_total",
                "spread_books",
                "total_books",
            ]
        ],
        on="game_key",
        how="left",
        validate="one_to_one",
    )
    combined = pd.concat([existing, captured], ignore_index=True)
    combined = combined.drop_duplicates(
        ["market_snapshot_timestamp_et", "game_key"], keep="first"
    )
    return combined.sort_values(
        ["market_snapshot_timestamp_et", "game_key"]
    ).reset_index(drop=True)


def preserve_existing_consensus_openers(
    existing: pd.DataFrame,
    captured: pd.DataFrame,
) -> pd.DataFrame:
    """Add first-observed totals without rewriting an existing spread opener."""
    if existing.empty:
        return captured.copy()
    output = existing.copy()
    for column in ("consensus_open_total", "consensus_open_total_books"):
        if column not in output:
            output[column] = pd.NA
    captured_lookup = captured.set_index("game_key")
    for index, row in output.iterrows():
        key = row["game_key"]
        if key not in captured_lookup.index:
            continue
        candidate = captured_lookup.loc[key]
        if isinstance(candidate, pd.DataFrame):
            candidate = candidate.iloc[0]
        for column in ("consensus_open_total", "consensus_open_total_books"):
            if pd.isna(output.loc[index, column]) and pd.notna(candidate.get(column)):
                output.loc[index, column] = candidate[column]
    missing = captured.loc[~captured["game_key"].isin(set(output["game_key"]))]
    return pd.concat([output, missing], ignore_index=True).sort_values(
        "game_key"
    ).reset_index(drop=True)


def latest_remaining_credits(usage: pd.DataFrame) -> float | None:
    if usage.empty or "requests_remaining" not in usage:
        return None
    remaining = pd.to_numeric(usage["requests_remaining"], errors="coerce").dropna()
    return float(remaining.iloc[-1]) if not remaining.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append spread/total history and lock Core V2 at the same timestamp"
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--board", type=Path, default=Path("current_live_board.csv"))
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--market-dir", type=Path, required=True)
    parser.add_argument(
        "--projection-history",
        type=Path,
        default=Path("data/tracking/projection_market_snapshots.csv"),
    )
    parser.add_argument(
        "--usage-history",
        type=Path,
        default=Path("data/market/api_usage.csv"),
    )
    parser.add_argument("--current-lines", type=Path, default=Path("current_book_lines.csv"))
    parser.add_argument("--current-openers", type=Path, default=Path("current_openers.csv"))
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--minimum-remaining-credits", type=int, default=100)
    args = parser.parse_args()

    board = pd.read_csv(args.board, low_memory=False)
    board = board.loc[
        (pd.to_numeric(board["season"], errors="coerce") == args.season)
        & (pd.to_numeric(board["week"], errors="coerce") == args.week)
    ].copy()
    schedule = pd.read_csv(args.schedule, low_memory=False)
    games = schedule_with_kickoffs(schedule, args.season, args.week)
    now = pd.Timestamp.now(tz="UTC")
    future_games = games.loc[games["kickoff_utc"].gt(now)].copy()
    future_keys = set(future_games["game_key"])
    board = board.loc[board["game_key"].isin(future_keys)].copy()
    if board.empty:
        print("No future games remain on the current board; no API request made")
        return

    usage = read_csv(args.usage_history)
    last_remaining = latest_remaining_credits(usage)
    if (
        last_remaining is not None
        and last_remaining <= args.minimum_remaining_credits
    ):
        print(
            f"API guard stopped pull: {last_remaining:.0f} credits remain; "
            f"minimum is {args.minimum_remaining_credits}"
        )
        return

    api_key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PARLAY_API_KEY is required")

    payload, metadata = request_markets(api_key, markets=("spreads", "totals"))
    snapshot_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    quotes = parse_market_payload(
        payload,
        args.season,
        args.week,
        set(board["game_key"]),
        snapshot_timestamp,
    )
    spread_games = set(quotes.loc[quotes["home_spread"].notna(), "game_key"])
    missing_spreads = set(board["game_key"]) - spread_games
    if missing_spreads:
        raise RuntimeError(
            "Market response was missing spread quotes for: "
            + ", ".join(sorted(missing_spreads))
        )

    args.market_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.market_dir / "market_history.csv"
    latest_path = args.market_dir / "latest_book_lines.csv"
    book_openers_path = args.market_dir / "book_openers.csv"
    dashboard_openers_path = args.market_dir / "dashboard_openers.csv"
    consensus_history_path = args.market_dir / "market_consensus_history.csv"

    history = append_market_history(read_csv(history_path), quotes)
    latest = latest_market_snapshot(history)
    consensus = build_market_consensus(latest)

    kickoff_by_game = (
        future_games.set_index("game_key")["kickoff_utc"].astype(str).to_dict()
    )
    book_openers = append_first_seen_openers(
        read_csv(book_openers_path), quotes, kickoff_by_game
    )
    captured_openers = build_consensus_openers(book_openers)
    existing_dashboard_openers = read_csv(dashboard_openers_path)
    if existing_dashboard_openers.empty:
        existing_dashboard_openers = read_csv(args.current_openers)
    dashboard_openers = preserve_existing_consensus_openers(
        existing_dashboard_openers, captured_openers
    )

    consensus_snapshot = consensus.copy()
    consensus_snapshot["market_snapshot_timestamp_et"] = snapshot_timestamp
    consensus_history = pd.concat(
        [read_csv(consensus_history_path), consensus_snapshot], ignore_index=True
    ).drop_duplicates(
        ["market_snapshot_timestamp_et", "game_key"], keep="first"
    )

    projection_history = append_projection_snapshots(
        read_csv(args.projection_history), board, consensus, snapshot_timestamp
    )
    usage_row = pd.DataFrame(
        [
            {
                "captured_at_utc": snapshot_timestamp,
                "season": args.season,
                "week": args.week,
                "markets": "spreads,totals",
                "requests_used": metadata.get("requests_used"),
                "requests_remaining": metadata.get("requests_remaining"),
                "games": quotes["game_key"].nunique(),
                "book_rows": len(quotes),
            }
        ]
    )
    usage = pd.concat([usage, usage_row], ignore_index=True)

    atomic_write_csv(history, history_path)
    atomic_write_csv(dashboard_lines(latest), latest_path)
    atomic_write_csv(book_openers, book_openers_path)
    atomic_write_csv(dashboard_openers, dashboard_openers_path)
    atomic_write_csv(consensus_history, consensus_history_path)
    atomic_write_csv(projection_history, args.projection_history)
    atomic_write_csv(usage, args.usage_history)
    atomic_write_csv(dashboard_lines(latest), args.current_lines)
    atomic_write_csv(dashboard_openers, args.current_openers)

    totals = int(quotes["total_points"].notna().sum())
    print(
        f"Captured {len(quotes)} book rows across {quotes['game_key'].nunique()} games; "
        f"{totals} rows include totals"
    )
    print(f"API requests used: {metadata.get('requests_used')}")
    print(f"API requests remaining: {metadata.get('requests_remaining')}")


if __name__ == "__main__":
    main()
