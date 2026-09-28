import pandas as pd
import pytest

from scripts.build_v3_team_game_history import assemble_history
from scripts.build_v3_weekly_board import build_week


def test_history_uses_only_final_graded_games():
    schedule = pd.DataFrame([
        dict(season=2026, week=3, game_id="2026_03_A_B", away_team="A", home_team="B", game_type="REG"),
        dict(season=2026, week=3, game_id="2026_03_C_D", away_team="C", home_team="D", game_type="REG"),
    ])
    grades = pd.DataFrame([
        dict(season=2026, week=3, game_key="2026_W3_A_B", away_score=10, home_score=14),
        dict(season=2026, week=3, game_key="2026_W3_C_D", away_score=None, home_score=None),
    ])
    games = pd.DataFrame([
        dict(season=2026, week=3, game_id="2026_03_A_B", team="A", opponent="B", plays=50, epa_per_play=.1),
        dict(season=2026, week=3, game_id="2026_03_A_B", team="B", opponent="A", plays=48, epa_per_play=-.2),
        dict(season=2026, week=3, game_id="2026_03_C_D", team="C", opponent="D", plays=3, epa_per_play=.5),
    ])
    result = assemble_history(games, schedule, grades, 2026, 4)
    assert len(result) == 2
    assert set(result.game_id) == {"2026_03_A_B"}
    assert result.loc[result.team.eq("A"), "def_epa_per_play"].iloc[0] == -.2
    with pytest.raises(ValueError, match="missing from nflverse"):
        assemble_history(games.iloc[0:0], schedule, grades, 2026, 4)


def test_early_board_shows_qb_scenario_but_unknown_defensive_pvar():
    priors = pd.DataFrame([
        dict(team="A", offense_strength=1., defense_strength=0.),
        dict(team="B", offense_strength=0., defense_strength=0.),
    ])
    schedule = pd.DataFrame([dict(season=2026, week=4, game_id="2026_04_A_B",
                                  away_team="A", home_team="B", location="Neutral", spread_line=1., game_type="REG")])
    coverage = pd.DataFrame([dict(team="A", report_available=False), dict(team="B", report_available=False)])
    availability = pd.DataFrame([dict(team="B", position_group="QB", assessment="PROVISIONAL_QB",
                                      adjustment_points=-1.5)])
    board, state = build_week(pd.DataFrame(columns=["season", "week", "game_id", "team", "plays", "epa_per_play", "def_epa_per_play"]),
                              priors, schedule, coverage, pd.DataFrame(), pd.DataFrame(), availability, 2026, 4)
    row = board.iloc[0]
    assert row.persistent_margin == -1.
    assert row.hfa_component == 0.
    assert row.qb_adjustment == -1.5
    assert row.pred_v3_lite == -2.5
    assert pd.isna(row.home_def_loss_pvar)
    assert row.personnel_status == "EARLY_PROVISIONAL_DEF_PVAR_UNAVAILABLE"
    assert len(state) == 2
