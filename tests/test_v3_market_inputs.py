import pandas as pd

from scripts.v3_market_snapshot import attach_market_snapshot
from scripts.v3_injury_guard import unresolved_injuries


def test_market_snapshot_uses_only_recent_timestamped_quote_with_correct_sign():
    board = pd.DataFrame([dict(season=2026, week=4, game_key="g", market_home_spread=8.5)])
    quotes = pd.DataFrame([
        dict(season=2026, week=4, game_key="g", market_home_spread=-4.0,
             market_snapshot_timestamp_et="2026-10-03T13:00:00Z"),
        dict(season=2026, week=4, game_key="g", market_home_spread=-5.5,
             market_snapshot_timestamp_et="2026-10-03T17:00:00Z"),
        dict(season=2026, week=4, game_key="g", market_home_spread=-99.0,
             market_snapshot_timestamp_et="2026-10-03T19:00:00Z"),
    ])
    now = pd.Timestamp("2026-10-03T18:00:00Z")
    recent = attach_market_snapshot(board, quotes, now).iloc[0]
    assert recent.market_home_spread == 5.5
    assert recent.market_line_status == "CAPTURED_RECENT"
    stale = attach_market_snapshot(board, quotes.iloc[:1], now).iloc[0]
    assert pd.isna(stale.market_home_spread)
    assert stale.market_line_status == "STALE_QUOTE"
    absent = attach_market_snapshot(board, pd.DataFrame(), now).iloc[0]
    assert pd.isna(absent.market_home_spread)


def test_unreviewed_qb_out_and_continuing_absence_block_until_weekly_review():
    queue = pd.DataFrame([dict(season=2026, week=5, team="CHI", player="Caleb Williams",
                               position_group="QB", signal="OFFICIAL_OUT", review_status="PENDING_REVIEW",
                               source_url="https://nfl.com/injuries")])
    continuing = pd.DataFrame([dict(season=2026, team="NYG", player="Brian Burns",
                                    first_week=4, last_week=18, source_url="https://giants.com/injury")])
    empty = pd.DataFrame(columns=["team", "player"])
    leads = unresolved_injuries(queue, continuing, empty, 2026, 5)
    assert set(leads.player) == {"Caleb Williams", "Brian Burns"}
    reviewed = pd.DataFrame([dict(team="CHI", player="Caleb Williams"),
                             dict(team="NYG", player="Brian Burns")])
    assert unresolved_injuries(queue, continuing, reviewed, 2026, 5).empty
    assert unresolved_injuries(queue, continuing, empty, 2026, 19).empty
