import pandas as pd
import pytest

from scripts.build_native_v3_pvar_board import build_board


def fixtures():
    board = pd.DataFrame([dict(game_id="2026_03_BAL_DAL", season=2026, week=3,
        home_team="DAL", away_team="BAL", neutral_site=True,
        market_adjustment=0., persistent_margin=-2., market_home_spread=-2.5)])
    audit = pd.DataFrame([
        dict(team="DAL", player_name="Defender", gsis_id="D1", position_group="DB",
             v2_shadow_eligible=True, status_weight=1., source_season=2026, source_week=3),
        dict(team="BAL", player_name="Offender", gsis_id="O1", position_group="OL",
             v2_shadow_eligible=True, status_weight=1., source_season=2026, source_week=3),
    ])
    ratings = pd.DataFrame([
        dict(gsis_id="D1", team="DAL", pvar_production=2., pvar_source="HISTORY"),
        dict(gsis_id="O1", team="BAL", pvar_production=9., pvar_source="HISTORY"),
    ])
    coverage = pd.DataFrame([dict(team=team, report_available=True) for team in ("BAL", "DAL")])
    return board, audit, ratings, coverage


def test_neutral_defense_only_and_sign():
    result = build_board(*fixtures()).iloc[0]
    assert result.hfa_component == 0
    assert result.home_def_loss_pvar == 2
    assert result.away_def_loss_pvar == 0
    assert result.personnel_adjustment == -4.5
    assert result.pred_v3_lite == -6.5
    assert result.model_side == "BAL"


def test_incomplete_coverage_fails_closed():
    board, audit, ratings, coverage = fixtures()
    coverage.loc[coverage.team.eq("DAL"), "report_available"] = False
    with pytest.raises(ValueError, match="coverage"):
        build_board(board, audit, ratings, coverage)


def test_mismatched_week_fails_closed():
    board, audit, ratings, coverage = fixtures()
    audit["source_week"] = 4
    with pytest.raises(ValueError, match="season/week"):
        build_board(board, audit, ratings, coverage)
