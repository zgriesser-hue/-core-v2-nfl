import pandas as pd

from core_v2.personnel import (
    build_personnel_adjustments,
    frozen_status_weight,
    load_weekly_team_adjustments,
    fetch_nfl_injury_report,
)
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


def test_persistent_injury_adjustment_expires_and_avoids_report_overlap(tmp_path):
    path = tmp_path / "weekly.csv"
    pd.DataFrame([
        {"season": 2026, "week": 2, "end_week": 5, "team": "LA",
         "adjustment_points": -0.56, "category": "INJURY", "player": "Myles Garrett",
         "reason": "IR", "exclude_from_auto": True},
        {"season": 2026, "week": 2, "team": "LA",
         "adjustment_points": -0.2, "category": "INJURY", "player": "Other",
         "reason": "Week 2 only", "exclude_from_auto": False},
    ]).to_csv(path, index=False)
    weekly = load_weekly_team_adjustments(path, 2026, 3)
    assert weekly["player"].tolist() == ["Myles Garrett"]
    assert load_weekly_team_adjustments(path, 2026, 6).empty

    schedule = pd.DataFrame([{"season": 2026, "week": 3,
                              "away_team": "LA", "home_team": "DEN"}])
    roles = pd.DataFrame([{"team": "LA", "player_name": "Myles Garrett",
                           "position": "LB", "expected_role": 0.9,
                           "name_key": "myles garrett"}])
    injury = pd.DataFrame([{"team": "LA", "player_name": "Myles Garrett",
                            "position_reported": "LB", "practice_status": "DNP",
                            "game_status": "Out"}])
    personnel, audit = build_personnel_adjustments(
        schedule, roles, injury, {"LA", "DEN"}, weekly, 2026, 3
    )
    assert audit.loc[0, "excluded_manual_overlap"]
    assert personnel.loc[0, "manual_adjustment_home"] == 0.56
    assert personnel.loc[0, "personnel_correction_current_clean"] == 0.56 - 0.00303256


def test_uncertain_qb_adjustment_changes_center_and_projection_band():
    schedule = pd.DataFrame([{"season": 2026, "week": 3,
                              "away_team": "PHI", "home_team": "CHI"}])
    roles = pd.DataFrame(columns=["team", "player_name", "position", "expected_role", "name_key"])
    injury = pd.DataFrame(columns=["team", "player_name", "position_reported",
                                   "practice_status", "game_status"])
    weekly = pd.DataFrame([{
        "team": "CHI", "adjustment_points": -0.75,
        "category": "QB_PROVISIONAL", "player": "Caleb Williams",
        "reason": "Bagent scenario", "exclude_from_auto": False,
        "out_adjustment_points": -1.5,
    }])
    personnel, _ = build_personnel_adjustments(
        schedule, roles, injury, {"PHI", "CHI"}, weekly, 2026, 3
    )
    assert personnel.loc[0, "qb_availability_provisional"]
    assert abs(personnel.loc[0, "personnel_correction_current_clean"] + 0.75303256) < 1e-10
    assert abs(personnel.loc[0, "personnel_correction_min_clean"] + 1.50303256) < 1e-10
    assert abs(personnel.loc[0, "personnel_correction_max_clean"] + 0.00303256) < 1e-10
    ratings = pd.DataFrame([
        {"team": "PHI", "current_team_rating": 3.0},
        {"team": "CHI", "current_team_rating": 1.0},
    ])
    board = build_projection_board(schedule, ratings, personnel, 2026, 3)
    assert board.loc[0, "projection_display_status"] == "EARLY_PROVISIONAL"
    assert abs(board.loc[0, "fair_spread_high"] - board.loc[0, "fair_spread_low"] - 1.5) < 1e-10


def test_fetch_injury_report_accepts_matching_week_heading(monkeypatch):
    html = """
    <html><head><title>Official NFL Injury Report for Players - Week 3 of the 2026 Season</title></head>
    <body><h1>2026 NFL Injury Report</h1><h2>Injuries - WEEK 3</h2></body></html>
    """

    class Response:
        text = html
        def raise_for_status(self):
            return None

    monkeypatch.setattr("core_v2.personnel.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        "core_v2.personnel.parse_nfl_injury_html",
        lambda html, valid_teams: (
            pd.DataFrame([{
                "team": "ATL", "player_name": "Player One", "position_reported": "DE",
                "injury": "Hamstring", "practice_status": "DNP", "game_status": "Out",
                "source_table": 0,
            }]),
            {"ATL", "GB"},
        ),
    )

    report, teams = fetch_nfl_injury_report(2026, 3, {"ATL", "GB"})
    assert teams == {"ATL", "GB"}
    assert report.loc[0, "source_week"] == 3
    assert report.loc[0, "source_season"] == 2026


def test_fetch_injury_report_rejects_prior_week_heading(monkeypatch):
    html = """
    <html><head><title>Official NFL Injury Report for Players - Week 2 of the 2026 Season</title></head>
    <body><h2>Injuries - WEEK 2</h2></body></html>
    """

    class Response:
        text = html
        def raise_for_status(self):
            return None

    monkeypatch.setattr("core_v2.personnel.requests.get", lambda *args, **kwargs: Response())

    import pytest
    with pytest.raises(ValueError, match="stale prior-week designations"):
        fetch_nfl_injury_report(2026, 3, {"ATL", "GB"})
