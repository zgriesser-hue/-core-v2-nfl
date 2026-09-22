from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core_v2.market import (
    append_market_history,
    parse_market_payload,
    request_markets,
)

SCHEDULE = Path("data/schedules/schedule_2026.csv")
MARKET_ROOT = Path("data/market/2026")
ET = ZoneInfo("America/New_York")
TARGET_BEFORE_KICKOFF = timedelta(minutes=12)
LOOKAHEAD = timedelta(minutes=75)


def kickoff_utc(row: pd.Series) -> datetime:
    local = datetime.strptime(
        f"{row['gameday']} {row['gametime']}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=ET)
    return local.astimezone(timezone.utc)


def main() -> None:
    now = datetime.now(timezone.utc)
    schedule = pd.read_csv(SCHEDULE)
    schedule = schedule.loc[
        (schedule["season"].eq(2026))
        & (schedule["game_type"].isin(["REG", "POST"]))
        & schedule["gameday"].notna()
        & schedule["gametime"].notna()
    ].copy()
    schedule["_kickoff"] = schedule.apply(kickoff_utc, axis=1)

    upcoming = schedule.loc[
        (schedule["_kickoff"] > now)
        & (schedule["_kickoff"] <= now + LOOKAHEAD)
    ].copy()
    if upcoming.empty:
        print("No kickoff within 75 minutes; no API request needed.")
        return

    target = upcoming["_kickoff"].min() - TARGET_BEFORE_KICKOFF
    if target > now:
        wait = (target - now).total_seconds()
        print(f"Waiting {wait:.0f}s for 12-minute pre-kickoff capture window.")
        time.sleep(wait)

    capture_now = datetime.now(timezone.utc)
    due = schedule.loc[
        (schedule["_kickoff"] > capture_now)
        & (schedule["_kickoff"] <= capture_now + timedelta(minutes=20))
    ].copy()
    if due.empty:
        print("No game is 0-20 minutes from kickoff after wait; exiting safely.")
        return

    api_key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PARLAY_API_KEY secret is missing or empty")

    payload, metadata = request_markets(api_key, markets=("spreads", "totals"))
    snapshot = capture_now.isoformat(timespec="seconds")
    due["_game_key"] = due.apply(
        lambda r: f"{int(r['season'])}_W{int(r['week'])}_{r['away_team']}_{r['home_team']}",
        axis=1,
    )

    for week, games in due.groupby("week"):
        week = int(week)
        allowed = set(games["_game_key"])
        quotes = parse_market_payload(payload, 2026, week, allowed, snapshot)
        week_dir = MARKET_ROOT / f"week_{week:02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        path = week_dir / "market_history.csv"
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=quotes.columns)
        updated = append_market_history(existing, quotes)
        updated.to_csv(path, index=False)
        print(
            f"Captured {quotes['game_key'].nunique()} game(s), "
            f"{len(quotes)} book rows for Week {week}: {', '.join(sorted(allowed))}"
        )

    print(
        "ParlayAPI usage: "
        f"used={metadata.get('requests_used')} remaining={metadata.get('requests_remaining')}"
    )


if __name__ == "__main__":
    main()
