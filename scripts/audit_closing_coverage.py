"""Report recent completed games with no validated pregame closing line."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.tracking import schedule_with_kickoffs


def missing_closes(schedule: pd.DataFrame, closes: pd.DataFrame,
                   now: pd.Timestamp, season: int, days: int = 3) -> pd.DataFrame:
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    weeks = pd.to_numeric(schedule.loc[schedule.season.eq(season), "week"], errors="coerce").dropna().unique()
    games = pd.concat([schedule_with_kickoffs(schedule, season, int(w)) for w in weeks],
                      ignore_index=True) if len(weeks) else pd.DataFrame()
    if games.empty:
        return pd.DataFrame(columns=["season", "week", "game_key", "kickoff_utc"])
    recent = games.loc[games.kickoff_utc.between(now - pd.Timedelta(days=days),
                                                now - pd.Timedelta(minutes=30))].copy()
    if closes.empty:
        recorded: set[str] = set()
    else:
        if not {"game_key", "closing_home_spread", "snapshot_timestamp_utc", "kickoff_utc"}.issubset(closes):
            raise ValueError("Closing consensus schema is incomplete")
        c = closes.copy()
        c["stamp"] = pd.to_datetime(c.snapshot_timestamp_utc, utc=True, errors="coerce", format="mixed")
        c["kickoff"] = pd.to_datetime(c.kickoff_utc, utc=True, errors="coerce", format="mixed")
        recorded = set(c.loc[c.closing_home_spread.notna() & c.stamp.notna()
                             & c.stamp.lt(c.kickoff), "game_key"].astype(str))
    return recent.loc[~recent.game_key.isin(recorded),
                      ["season", "week", "game_key", "kickoff_utc"]].sort_values("kickoff_utc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--closes", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    schedule = pd.read_csv(args.schedule)
    closes = pd.read_csv(args.closes) if args.closes.exists() else pd.DataFrame()
    missing = missing_closes(schedule, closes, pd.Timestamp.now(tz="UTC"), args.season)
    args.report.write_text("\n".join(
        [f"Missing validated pregame closes ({len(missing)} games). Earlier market quotes are not closes.", ""]
        + [f"- Week {int(row.week)} {row.game_key} — kickoff {row.kickoff_utc}" for row in missing.itertuples()]
    ) + "\n" if not missing.empty else "")
    print(f"Recent completed games without validated closes: {len(missing)}")


if __name__ == "__main__":
    main()
