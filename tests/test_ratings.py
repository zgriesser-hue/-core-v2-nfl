import pandas as pd

from core_v2.ratings import latest_fully_graded_week, update_team_ratings


def ratings():
    return pd.DataFrame([
        {"team": team, "base_prior": value, "qb_adjustment": 0.0,
         "manual_personnel_adjustment": 0.0, "manual_injury_adjustment": 0.0,
         "current_team_rating": value, "rank": rank}
        for team, value, rank in [("AWY", -1.0, 2), ("HME", 1.0, 1)]
    ])


def grades(actual_margin=12.0):
    return pd.DataFrame([{
        "season": 2026, "week": 1, "game_key": "2026_W1_AWY_HME",
        "away_team": "AWY", "home_team": "HME",
        "actual_home_margin": actual_margin, "projected_home_margin": 2.0,
    }])


def test_update_is_zero_sum_and_idempotent():
    first = update_team_ratings(ratings(), grades(), 2026, 1)
    second = update_team_ratings(first, grades(), 2026, 1)
    assert first["performance_adjustment"].sum() == 0.0
    assert first.set_index("team").loc["HME", "performance_adjustment"] == 0.5
    pd.testing.assert_series_equal(
        first.set_index("team")["current_team_rating"],
        second.set_index("team")["current_team_rating"],
    )


def test_team_game_adjustment_is_capped():
    updated = update_team_ratings(ratings(), grades(actual_margin=100.0), 2026, 1)
    assert updated["performance_adjustment"].abs().max() == 0.75


def test_latest_week_requires_every_scheduled_game():
    schedule = pd.DataFrame([
        {"season": 2026, "week": 1, "game_type": "REG", "game_key": "one"},
        {"season": 2026, "week": 2, "game_type": "REG", "game_key": "two-a"},
        {"season": 2026, "week": 2, "game_type": "REG", "game_key": "two-b"},
    ])
    grade_rows = pd.DataFrame([
        {"season": 2026, "game_key": "one"},
        {"season": 2026, "game_key": "two-a"},
    ])
    assert latest_fully_graded_week(schedule, grade_rows, 2026) == 1
