import pandas as pd

from core_v2.market import (
    append_market_history,
    append_first_seen_openers,
    build_market_consensus,
    build_consensus_openers,
    latest_market_snapshot,
    parse_market_payload,
    parse_spread_payload,
)
from parlay_live import select_pregame_game_keys


def payload(line=-3.5, topped_up="False"):
    return [
        {
            "away_team": "Washington Commanders",
            "home_team": "Philadelphia Eagles",
            "bookmakers": [
                {
                    "key": "draft_kings",
                    "topped_up": topped_up,
                    "last_update": "2026-09-10T16:00:00Z",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Washington Commanders", "point": -line, "price": 1.91},
                                {"name": "Philadelphia Eagles", "point": line, "price": 1.91},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 47.5, "price": 1.91},
                                {"name": "Under", "point": 47.5, "price": 1.91},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_first_seen_opener_is_immutable_and_boolean_is_clean():
    allowed = {"2026_W1_WAS_PHI"}
    first = parse_spread_payload(payload(-3.5), 2026, 1, allowed, "2026-09-01T12:00:00Z")
    assert first.loc[0, "book_topped_up_clean"] == False  # noqa: E712
    openers = append_first_seen_openers(pd.DataFrame(), first)

    later = parse_spread_payload(payload(-5.5), 2026, 1, allowed, "2026-09-10T12:00:00Z")
    still_open = append_first_seen_openers(openers, later)
    assert len(still_open) == 1
    assert still_open.loc[0, "opening_home_spread"] == -3.5


def test_consensus_uses_configured_core_books():
    rows = []
    for book, line in [("draftkings", -3.5), ("fanduel", -4.5), ("bet365", -4.0), ("betmgm", -9.5)]:
        rows.append(
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_WAS_PHI",
                "away_team": "WAS",
                "home_team": "PHI",
                "book": book,
                "opening_home_spread": line,
                "opening_seen_utc": "2026-09-01T12:00:00Z",
            }
        )
    result = build_consensus_openers(pd.DataFrame(rows))
    assert result.loc[0, "consensus_open_home_spread"] == -4.0
    assert result.loc[0, "consensus_open_books"] == 3


def test_market_parser_and_history_capture_totals_idempotently():
    allowed = {"2026_W1_WAS_PHI"}
    first = parse_market_payload(
        payload(-3.5), 2026, 1, allowed, "2026-09-01T12:00:00Z"
    )
    assert first.loc[0, "total_points"] == 47.5
    assert first.loc[0, "over_price_american"] == -110

    history = append_market_history(pd.DataFrame(), first)
    history = append_market_history(history, first)
    assert len(history) == 1

    later = parse_market_payload(
        payload(-4.5), 2026, 1, allowed, "2026-09-01T18:00:00Z"
    )
    history = append_market_history(history, later)
    assert len(history) == 2
    latest = latest_market_snapshot(history)
    assert latest.loc[0, "home_spread"] == -4.5


def test_current_consensus_contains_spread_and_total():
    rows = []
    for book, line, total in [
        ("draftkings", -3.5, 47.5),
        ("fanduel", -4.5, 48.5),
        ("bet365", -4.0, 48.0),
        ("betmgm", -9.5, 60.0),
    ]:
        rows.append(
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_WAS_PHI",
                "away_team": "WAS",
                "home_team": "PHI",
                "sportsbook": book,
                "home_spread": line,
                "total_points": total,
                "market_snapshot_timestamp_et": "2026-09-01T12:00:00Z",
            }
        )
    result = build_market_consensus(pd.DataFrame(rows))
    assert result.loc[0, "market_home_spread"] == -4.0
    assert result.loc[0, "market_total"] == 48.0
    assert result.loc[0, "spread_books"] == 3
    assert result.loc[0, "total_books"] == 3


def test_live_refresh_excludes_started_and_graded_games():
    board = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_NE_SEA",
                "away_team": "NE",
                "home_team": "SEA",
            },
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_SF_LA",
                "away_team": "SF",
                "home_team": "LA",
            },
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_CHI_CAR",
                "away_team": "CHI",
                "home_team": "CAR",
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "away_team": "NE",
                "home_team": "SEA",
                "gameday": "2026-09-09",
                "gametime": "20:20",
            },
            {
                "season": 2026,
                "week": 1,
                "away_team": "SF",
                "home_team": "LA",
                "gameday": "2026-09-10",
                "gametime": "20:35",
            },
            {
                "season": 2026,
                "week": 1,
                "away_team": "CHI",
                "home_team": "CAR",
                "gameday": "2026-09-13",
                "gametime": "13:00",
            },
        ]
    )

    keys = select_pregame_game_keys(
        board,
        schedule=schedule,
        completed_game_keys={"2026_W1_NE_SEA"},
        now_utc="2026-09-11T14:47:00Z",
    )

    assert keys == ("2026_W1_CHI_CAR",)


def test_live_refresh_keeps_future_missing_api_game_in_validation_set():
    board = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_key": "2026_W1_CHI_CAR",
                "away_team": "CHI",
                "home_team": "CAR",
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "away_team": "CHI",
                "home_team": "CAR",
                "kickoff_utc": "2026-09-13T17:00:00Z",
            }
        ]
    )

    keys = select_pregame_game_keys(
        board,
        schedule=schedule,
        now_utc="2026-09-11T14:47:00Z",
    )

    assert keys == ("2026_W1_CHI_CAR",)
