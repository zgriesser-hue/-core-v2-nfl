from datetime import datetime, timezone

import pandas as pd

from core_v2.tracking import (
    append_closing_snapshot,
    build_release_records,
    due_closing_games,
    grade_completed_games,
)


GAME_KEY = "2026_W1_WAS_PHI"


def game(kickoff="2026-09-13T17:00:00Z"):
    return pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "game_key": GAME_KEY,
            "away_team": "WAS",
            "home_team": "PHI",
            "kickoff_utc": pd.Timestamp(kickoff),
        }]
    )


def quotes(snapshot="2026-09-13T16:30:00Z"):
    rows = []
    for book, line in [
        ("draftkings", -3.0),
        ("fanduel", -3.5),
        ("bet365", -4.0),
        ("caesars", -3.5),
        ("betmgm", -8.5),
    ]:
        rows.append(
            {
                "season": 2026,
                "week": 1,
                "game_key": GAME_KEY,
                "away_team": "WAS",
                "home_team": "PHI",
                "sportsbook": book,
                "market_snapshot_timestamp_et": snapshot,
                "observed_timestamp_et": snapshot,
                "away_spread": -line,
                "away_price_american": -110,
                "home_spread": line,
                "home_price_american": -110,
                "book_topped_up_clean": False,
            }
        )
    return pd.DataFrame(rows)


def test_due_window_skips_already_captured_games():
    now = datetime(2026, 9, 13, 16, 30, tzinfo=timezone.utc)
    due = due_closing_games(game(), pd.DataFrame(), now_utc=now)
    assert due["game_key"].tolist() == [GAME_KEY]

    existing = pd.DataFrame([{"game_key": GAME_KEY}])
    assert due_closing_games(game(), existing, now_utc=now).empty


def test_closing_snapshot_is_one_time_and_excludes_betmgm():
    books, closes = append_closing_snapshot(
        pd.DataFrame(), pd.DataFrame(), quotes(), game()
    )
    assert "betmgm" not in set(books["sportsbook"])
    assert closes.loc[0, "closing_home_spread"] == -3.5
    assert closes.loc[0, "consensus_books"] == 4

    try:
        append_closing_snapshot(books, closes, quotes(), game())
    except RuntimeError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("A second close should never replace the first close")


def test_grading_uses_last_release_before_close_and_calculates_clv():
    board = pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "game_key": GAME_KEY,
            "away_team": "WAS",
            "home_team": "PHI",
            "fair_home_spread": -6.0,
        }]
    )
    release = build_release_records(board, quotes("2026-09-13T15:00:00Z"), "2026-09-13T15:00:00Z")
    post_kick = release.copy()
    post_kick["release_timestamp_utc"] = "2026-09-13T17:05:00Z"
    post_kick["fair_home_spread"] = 7.0
    releases = pd.concat([release, post_kick], ignore_index=True)
    closings = pd.DataFrame(
        [{
            "game_key": GAME_KEY,
            "closing_home_spread": -4.5,
            "consensus_books": 4,
            "snapshot_timestamp_utc": "2026-09-13T16:30:00Z",
        }]
    )
    results = pd.DataFrame(
        [{"game_key": GAME_KEY, "away_score": 20, "home_score": 24}]
    )
    grades = grade_completed_games(
        releases, closings, results, pd.DataFrame()
    )
    assert len(grades) == 1
    assert grades.loc[0, "model_selected_team"] == "PHI"
    assert grades.loc[0, "ats_result_at_release"] == "Win"
    assert grades.loc[0, "clv_points"] == 1.0
    assert grades.loc[0, "bet_clv_points"] == 1.0
    assert grades.loc[0, "projection_vs_close_points"] == 1.5
    assert grades.loc[0, "model_edge_at_close"] == 1.5
    assert grades.loc[0, "projection_abs_error"] == 2.0


def test_pass_is_not_graded_as_an_away_pick():
    board = pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "game_key": GAME_KEY,
            "away_team": "WAS",
            "home_team": "PHI",
            "fair_home_spread": -3.5,
        }]
    )
    releases = build_release_records(
        board, quotes("2026-09-13T15:00:00Z"), "2026-09-13T15:00:00Z"
    )
    closings = pd.DataFrame(
        [{
            "game_key": GAME_KEY,
            "closing_home_spread": -3.5,
            "consensus_books": 4,
            "snapshot_timestamp_utc": "2026-09-13T16:30:00Z",
        }]
    )
    results = pd.DataFrame(
        [{"game_key": GAME_KEY, "away_score": 20, "home_score": 24}]
    )
    grades = grade_completed_games(
        releases, closings, results, pd.DataFrame()
    )
    assert grades.loc[0, "model_selected_side"] == "PASS"
    assert grades.loc[0, "ats_result_at_release"] == "Pass"
    assert pd.isna(grades.loc[0, "clv_points"])
    assert pd.isna(grades.loc[0, "projection_vs_close_points"])


def test_final_score_is_graded_when_close_was_not_captured():
    board = pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "game_key": GAME_KEY,
            "away_team": "WAS",
            "home_team": "PHI",
            "fair_home_spread": -6.0,
        }]
    )
    releases = build_release_records(
        board, quotes("2026-09-13T15:00:00Z"), "2026-09-13T15:00:00Z"
    )
    results = pd.DataFrame(
        [{"game_key": GAME_KEY, "away_score": 27, "home_score": 7}]
    )

    grades = grade_completed_games(
        releases, pd.DataFrame(), results, pd.DataFrame()
    )

    assert len(grades) == 1
    assert grades.loc[0, "ats_result_at_release"] == "Loss"
    assert grades.loc[0, "projection_abs_error"] == 26.0
    assert grades.loc[0, "closing_status"] == "MISSING_NOT_BACKFILLED"
    assert pd.isna(grades.loc[0, "closing_bet_spread"])
    assert pd.isna(grades.loc[0, "clv_points"])
