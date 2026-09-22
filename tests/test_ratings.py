import pandas as pd

from core_v2.ratings import (\n    latest_fully_graded_week,\n    update_team_ratings,\n    update_team_ratings_opponent_adjusted,\n)


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


def test_partial_recompute_does_not_double_count_games():
    partial = update_team_ratings(ratings(), grades(), 2026, 1)
    full_grades = pd.concat([
        grades(),
        pd.DataFrame([{
            "season": 2026, "week": 2, "game_key": "2026_W2_HME_AWY",
            "away_team": "HME", "home_team": "AWY",
            "actual_home_margin": 1.0, "projected_home_margin": -1.0,
        }]),
    ], ignore_index=True)
    recomputed = update_team_ratings(partial, full_grades, 2026, 2)
    direct = update_team_ratings(ratings(), full_grades, 2026, 2)
    pd.testing.assert_series_equal(
        recomputed.set_index("team")["current_team_rating"],
        direct.set_index("team")["current_team_rating"],
    )



def test_opponent_adjusted_zero_is_current_model():
    current = update_team_ratings(ratings(), grades(), 2026, 1)
    shadow = update_team_ratings_opponent_adjusted(
        ratings(), grades(), 2026, 1, opponent_shrinkage=0.0
    )
    pd.testing.assert_series_equal(
        current.set_index("team")["current_team_rating"],
        shadow.set_index("team")["current_team_rating"],
    )


def test_opponent_adjusted_is_zero_sum_and_shrinks_early_signal():
    current = update_team_ratings(ratings(), grades(), 2026, 1)
    shadow = update_team_ratings_opponent_adjusted(
        ratings(), grades(), 2026, 1, opponent_shrinkage=1.0
    )
    assert abs(shadow["performance_adjustment"].sum()) < 1e-10
    assert (
        shadow["performance_adjustment"].abs().max()
        < current["performance_adjustment"].abs().max()
    )
