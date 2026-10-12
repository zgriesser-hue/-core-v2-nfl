"""Wait for upcoming kickoffs and use the canonical close capture writer.

An early GitHub scheduled run can cover later kickoffs when intermediate cron
events are delayed or omitted. The worker exits before the six-hour job limit.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core_v2.tracking import schedule_with_kickoffs


SCHEDULE = Path("data/schedules/schedule_2026.csv")
MARKET_ROOT = Path("data/market/2026")
TRACKING = Path("data/tracking/closing_consensus.csv")
TARGET_BEFORE_KICKOFF = timedelta(minutes=12)
MAX_WORKER_SPAN = timedelta(hours=5, minutes=40)


def target_games(games: pd.DataFrame, now: datetime, deadline: datetime,
                 attempted: set[str]) -> pd.DataFrame:
    """The next uncaptured kickoff reachable within this worker's lifetime."""
    candidates = games.loc[
        games.kickoff_utc.gt(pd.Timestamp(now))
        & games.kickoff_utc.le(pd.Timestamp(deadline) + TARGET_BEFORE_KICKOFF)
        & ~games.game_key.isin(attempted)
    ].sort_values("kickoff_utc")
    if candidates.empty:
        return candidates
    return candidates.loc[candidates.kickoff_utc.eq(candidates.kickoff_utc.iloc[0])]


def capture_due_games(due: pd.DataFrame, runner=subprocess.run) -> set[str]:
    """Attempt each due kickoff and report failures without discarding earlier captures."""
    failed: set[str] = set()
    for week, group in due.groupby("week"):
        week = int(week)
        keys = group.game_key.astype(str).tolist()
        print(f"Capturing Week {week}: {', '.join(keys)}", flush=True)
        try:
            runner([
                sys.executable, "-m", "scripts.capture_closings",
                "--season", "2026", "--week", str(week),
                "--schedule", str(SCHEDULE),
                "--output-dir", str(MARKET_ROOT / f"week_{week:02d}"),
                "--tracking-consensus", str(TRACKING),
                "--minimum-lead", "0", "--maximum-lead", "20",
            ], check=True)
        except subprocess.CalledProcessError:
            failed.update(keys)
            print(f"Closing capture failed for {', '.join(keys)}", flush=True)
    return failed


def main() -> None:
    started = datetime.now(timezone.utc)
    deadline = started + MAX_WORKER_SPAN
    schedule = pd.read_csv(SCHEDULE)
    current = schedule.loc[pd.to_numeric(schedule.season, errors="coerce").eq(2026)]
    weeks = current.loc[current.gameday.between(
        (started - timedelta(days=1)).date().isoformat(),
        (deadline + timedelta(days=1)).date().isoformat()), "week"].unique()
    games = pd.concat([schedule_with_kickoffs(schedule, 2026, int(week)) for week in weeks],
                      ignore_index=True) if len(weeks) else pd.DataFrame()
    if games.empty:
        print("No upcoming scheduled games")
        return
    attempted: set[str] = set()
    failed: set[str] = set()
    while True:
        now = datetime.now(timezone.utc)
        due = target_games(games, now, deadline, attempted)
        if due.empty:
            print("No more reachable kickoffs; backup capture complete")
            if failed:
                raise RuntimeError(f"Closing capture failed for: {', '.join(sorted(failed))}")
            return
        target = due.kickoff_utc.iloc[0].to_pydatetime() - TARGET_BEFORE_KICKOFF
        wait = max(0, (target - now).total_seconds())
        if wait:
            print(f"Waiting {wait:.0f}s until the next pre-kickoff capture", flush=True)
            time.sleep(wait)
        failed.update(capture_due_games(due))
        attempted.update(due.game_key.astype(str))


if __name__ == "__main__":
    main()
