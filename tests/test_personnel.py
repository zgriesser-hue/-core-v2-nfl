import pandas as pd

from core_v2.personnel import build_personnel_adjustments, frozen_status_weight
from core_v2.projections import build_projection_board


def test_frozen_status_weights():
    assert frozen_status_weight("Out", "DNP")[0] == 1.00
    assert frozen_status_weight("Doubtful", "Limited")[0] == 0.95
    assert frozen_status_weight("Questionable", "Full")[0] == 0.35
    assert frozen_status_weight("", "Did Not Participate")[0] == 0.20
    assert frozen_status_weight("", "Limited Participation")[0] == 0.05


def test_manual_adjustment_is_applied_while_reports_are_provisional():
    schedule = pd.DataFrame([{"season": 2026, "week": 2, "game_id": "g", "away_team": "SEA", "home_team": "ARI"}])
    roles = pd.DataFrame(columns=["team", "player_name", "position", "expected_role", "name_key"])
    injury = pd.DataFrame(columns=["team", "player_name", "position_reported", "practice_status", "game_status"])
    weekly = pd.DataFrame([
        {"team": "SEA", "adjustment_points": -1.5, "category": "QB", "player": "Sam Darnold", "reason": "out", "exclude_from_auto": False}
    ])
    personnel, _ = build_personnel_adjustments(
        schedule, roles, injury, set(), weekly, 2026, 2
    )
    assert personnel.loc[0, "personnel_correction_current_clean"] == 1.5
    assert not bool(personnel.loc[0, "report_coverage_complete_clean"])

    ratings = pd.DataFrame([
        {"team": "SEA", "current_team_rating": 3.0},
        {"team": "ARI", "current_team_rating": -4.0},
    ])
    board = build_projection_board(schedule, ratings, personnel, 2026, 2)
    # Pre-personnel ARI spread is +4.8; the verified SEA downgrade moves it to +3.3.
    assert board.loc[0, "fair_spread_home"] == 3.3
    assert board.loc[0, "projection_display_status"] == "EARLY_PROVISIONAL"


def test_official_non_qb_exposure_uses_frozen_formula():
    schedule = pd.DataFrame([{"season": 2026, "week": 2, "game_id": "g", "away_team": "PIT", "home_team": "NE"}])
    roles = pd.DataFrame([
        {"team": "NE", "player_name": "Receiver One", "position": "WR", "expected_role": 0.8, "name_key": "receiver one"}
    ])
    injury = pd.DataFrame([
        {"team": "NE", "player_name": "Receiver One", "position_reported": "WR", "practice_status": "DNP", "game_status": "Out"}
    ])
    weekly = pd.DataFrame(columns=["team", "adjustment_points", "category", "player", "reason", "exclude_from_auto"])
    personnel, audit = build_personnel_adjustments(
        schedule, roles, injury, {"PIT", "NE"}, weekly, 2026, 2
    )
    expected = -0.61336750 * 0.8 - 0.00303256
    assert abs(personnel.loc[0, "personnel_correction_current_clean"] - expected) < 1e-10
    assert bool(personnel.loc[0, "report_coverage_complete_clean"])
    assert audit.loc[0, "match_method"] == "EXACT"
