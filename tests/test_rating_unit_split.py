import pandas as pd
import pytest

from scripts.backtest_rating_unit_split import add_unit_features, normalize_team


def test_historical_team_normalization():
    assert normalize_team("OAK") == "LV"
    assert normalize_team("PHI") == "PHI"


def test_unit_features_are_pregame_and_week_safe():
    games = pd.DataFrame([
        {
            "game_id": "one",
            "season": 2025,
            "week": 1,
            "home_team": "AAA",
            "away_team": "BBB",
            "home_game_off_epa": 0.20,
            "away_game_off_epa": -0.10,
        },
        {
            "game_id": "two",
            "season": 2025,
            "week": 1,
            "home_team": "CCC",
            "away_team": "DDD",
            "home_game_off_epa": 0.80,
            "away_game_off_epa": -0.80,
        },
        {
            "game_id": "three",
            "season": 2025,
            "week": 2,
            "home_team": "AAA",
            "away_team": "BBB",
            "home_game_off_epa": 0.00,
            "away_game_off_epa": 0.00,
        },
    ])

    featured = add_unit_features(games, alpha=1.0, carryover=0.5).set_index("game_id")

    assert featured.loc["one", "offense_difference"] == 0.0
    assert featured.loc["two", "offense_difference"] == 0.0
    assert featured.loc["three", "offense_difference"] == pytest.approx(0.30)
    assert featured.loc["three", "defense_difference"] == pytest.approx(0.30)
