import pandas as pd
import pytest

from scripts.build_v3_team_game_history import assemble_history
from scripts.build_v3_weekly_board import build_week, unreviewed_official_qbs


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


def test_reviewed_defensive_out_is_priced_before_report_and_not_double_counted():
    priors = pd.DataFrame([
        dict(team="A", offense_strength=0., defense_strength=0.),
        dict(team="B", offense_strength=0., defense_strength=0.),
    ])
    schedule = pd.DataFrame([dict(season=2026, week=4, game_id="2026_04_A_B",
                                  away_team="A", home_team="B", location="Neutral",
                                  spread_line=0., game_type="REG")])
    availability = pd.DataFrame([dict(team="B", player="Defender B",
                                      position_group="DB", assessment="CONFIRMED_OUT",
                                      adjustment_points=0.)])
    ratings = pd.DataFrame([dict(team="B", player_name="Defender B", gsis_id="00-1234567",
                                 model_position_v2="CB", pvar_production=.8,
                                 pvar_source="REBASED")])
    history = pd.DataFrame(columns=["season", "week", "game_id", "team", "plays",
                                     "epa_per_play", "def_epa_per_play"])
    missing = pd.DataFrame([dict(team="A", report_available=False),
                            dict(team="B", report_available=False)])
    early, _ = build_week(history, priors, schedule, missing, pd.DataFrame(),
                          ratings, availability, 2026, 4)
    row = early.iloc[0]
    assert row.home_def_loss_pvar == pytest.approx(.8)
    assert row.fair_home_spread_v3 == pytest.approx(-1.8)
    assert row.personnel_status == "EARLY_PROVISIONAL_KNOWN_DEF_PVAR_ONLY"
    assert "Defender B" in row.def_availability_notes

    covered = pd.DataFrame([dict(team="A", report_available=True),
                            dict(team="B", report_available=True)])
    audit = pd.DataFrame([dict(team="B", player_name="Defender B",
                               gsis_id="00-1234567", source_season=2026,
                               source_week=4, position_group="DB",
                               v2_shadow_eligible=True, status_weight=1.)])
    full, _ = build_week(history, priors, schedule, covered, audit,
                         ratings, availability, 2026, 4)
    assert full.iloc[0].home_def_loss_pvar == pytest.approx(.8)
    assert full.iloc[0].fair_home_spread_v3 == pytest.approx(-1.8)

    audit.loc[0, "status_weight"] = .5
    partial, _ = build_week(history, priors, schedule, covered, audit,
                            ratings, availability, 2026, 4)
    assert partial.iloc[0].home_def_loss_pvar == pytest.approx(.8)
    assert partial.iloc[0].fair_home_spread_v3 == pytest.approx(-1.8)


def test_unrated_defender_blocks_only_affected_matchup_edge():
    priors = pd.DataFrame([
        dict(team="JAX", offense_strength=0., defense_strength=0.),
        dict(team="PHI", offense_strength=0., defense_strength=0.),
    ])
    schedule = pd.DataFrame([dict(
        season=2026, week=5, game_id="2026_05_PHI_JAX",
        away_team="PHI", home_team="JAX", location="Home",
        spread_line=0., game_type="REG",
    )])
    coverage = pd.DataFrame([
        dict(team="JAX", report_available=True),
        dict(team="PHI", report_available=True),
    ])
    audit = pd.DataFrame([
        dict(team="JAX", player_name="Jaylon Jones", gsis_id=None,
             source_season=2026, source_week=5, position_group="DB",
             v2_shadow_eligible=True, status_weight=0.05),
        dict(team="PHI", player_name="Known defender", gsis_id="D1",
             source_season=2026, source_week=5, position_group="DB",
             v2_shadow_eligible=True, status_weight=1.0),
    ])
    ratings = pd.DataFrame([
        dict(gsis_id="D1", team="PHI", pvar_production=0.5,
             pvar_source="REBASED"),
    ])
    history = pd.DataFrame(columns=[
        "season", "week", "game_id", "team", "plays", "epa_per_play",
        "def_epa_per_play",
    ])
    availability = pd.DataFrame([
        dict(team="JAX", position_group="QB", assessment="AVAILABLE",
             adjustment_points=0.0),
    ])
    board, _ = build_week(
        history, priors, schedule, coverage, audit, ratings,
        availability, 2026, 5,
    )
    row = board.iloc[0]
    assert row.away_def_loss_pvar == pytest.approx(0.5)
    assert row.home_def_loss_pvar == pytest.approx(0.0)
    assert "Jaylon Jones" in row.unpriced_injury_notes
    assert row.edge_status == "INJURY_REVIEW_REQUIRED"
    assert pd.isna(row.edge_points)
    assert pd.isna(row.model_side)
    assert audit.v2_shadow_eligible.all()


def test_official_starter_qb_designations_require_current_week_review():
    audit = pd.DataFrame([
        dict(team="BAL", player_name="Lamar Jackson", position_group="QB",
             game_status="Out", expected_role_v2=.73),
        dict(team="TB", player_name="Baker Mayfield", position_group="QB",
             game_status="Out", expected_role_v2=.99),
        dict(team="CHI", player_name="Caleb Williams", position_group="QB",
             game_status="Questionable", expected_role_v2=1.0),
        dict(team="CLE", player_name="Dillon Gabriel", position_group="QB",
             game_status="Out", expected_role_v2=.08),
    ])
    reviewed = pd.DataFrame([
        dict(team="BAL", player="Lamar Jackson", assessment="CONFIRMED_OUT"),
    ])
    pending = unreviewed_official_qbs(audit, reviewed)
    assert set(pending.player) == {"Baker Mayfield", "Caleb Williams"}
    assert pending.loc[pending.player.eq("Baker Mayfield"), "reason"].iloc[0] == (
        "official QB Out requires current-week review"
    )

    conflict = reviewed.copy()
    conflict["assessment"] = "AVAILABLE"
    with pytest.raises(ValueError, match="conflicts"):
        unreviewed_official_qbs(audit, conflict)
