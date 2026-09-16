import pandas as pd

from scripts.refresh_market_history import (
    append_projection_snapshots,
    latest_remaining_credits,
)


def test_projection_snapshots_are_timestamped_and_idempotent():
    board = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 2,
                "game_key": "2026_W2_IND_KC",
                "away_team": "IND",
                "home_team": "KC",
                "fair_home_spread": -3.5,
                "projection_display_status": "EARLY_PROVISIONAL",
            }
        ]
    )
    consensus = pd.DataFrame(
        [
            {
                "game_key": "2026_W2_IND_KC",
                "market_home_spread": -4.0,
                "market_total": 47.5,
                "spread_books": 4,
                "total_books": 4,
            }
        ]
    )
    timestamp = "2026-09-16T12:20:00Z"
    history = append_projection_snapshots(
        pd.DataFrame(), board, consensus, timestamp
    )
    history = append_projection_snapshots(history, board, consensus, timestamp)
    assert len(history) == 1
    assert history.loc[0, "fair_home_spread"] == -3.5
    assert history.loc[0, "market_total"] == 47.5


def test_latest_remaining_credits_ignores_unparseable_values():
    usage = pd.DataFrame({"requests_remaining": ["900", "unknown", "842"]})
    assert latest_remaining_credits(usage) == 842
