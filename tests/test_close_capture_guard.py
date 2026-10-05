from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.audit_closing_coverage import missing_closes
from scripts.capture_pregame_closes import target_games


def test_early_worker_covers_skipped_afternoon_cron():
    start = datetime(2026, 10, 4, 11, 34, tzinfo=timezone.utc)
    deadline = start + timedelta(hours=5, minutes=40)
    games = pd.DataFrame([
        {"game_key": "early", "kickoff_utc": pd.Timestamp("2026-10-04T13:30Z")},
        {"game_key": "one_et", "kickoff_utc": pd.Timestamp("2026-10-04T17:00Z")},
        {"game_key": "later", "kickoff_utc": pd.Timestamp("2026-10-04T20:25Z")},
    ])
    first = target_games(games, start, deadline, set())
    assert first.game_key.tolist() == ["early"]
    second = target_games(games, datetime(2026, 10, 4, 13, 20, tzinfo=timezone.utc),
                          deadline, {"early"})
    assert second.game_key.tolist() == ["one_et"]
    assert target_games(games, start, deadline, {"early", "one_et"}).empty


def test_audit_flags_missing_and_rejects_postkick_quote():
    schedule = pd.DataFrame([
        {"season": 2026, "week": 4, "game_type": "REG", "gameday": "2026-10-04",
         "gametime": "13:00", "away_team": "A", "home_team": "H"},
        {"season": 2026, "week": 4, "game_type": "REG", "gameday": "2026-10-04",
         "gametime": "16:25", "away_team": "B", "home_team": "J"},
    ])
    now = pd.Timestamp("2026-10-04T21:00Z")
    closes = pd.DataFrame([{"game_key": "2026_W4_A_H", "closing_home_spread": -3.0,
                             "snapshot_timestamp_utc": "2026-10-04T17:01Z",
                             "kickoff_utc": "2026-10-04T17:00Z"}])
    assert missing_closes(schedule, closes, now, 2026).game_key.tolist() == ["2026_W4_A_H"]
    assert missing_closes(schedule, closes.assign(snapshot_timestamp_utc="2026-10-04T16:48Z"),
                          now, 2026).empty
