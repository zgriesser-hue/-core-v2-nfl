import pandas as pd
import pytest

from core_v2.personnel_v2 import (
    POSITION_GROUPS,
    add_v2_shadow_features,
    build_personnel_v2_shadow,
    load_v2_role_repairs,
    normalize_position_group,
)


def test_position_groups_are_stable():
    assert normalize_position_group("OT") == "OL"
    assert normalize_position_group("TE") == "WR_TE"
    assert normalize_position_group("DE") == "EDGE"
    assert normalize_position_group("S") == "DB"
    assert normalize_position_group("QB") == "QB"
    assert normalize_position_group("ATH") == "OTHER"


def test_shadow_matrix_separates_groups_and_flags_bad_priors():
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_02_NE_NYJ",
                "away_team": "NE",
                "home_team": "NYJ",
            }
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "team": "NE",
                "position_reported": "T",
                "matched_position": "OL",
                "expected_role": 0.60,
                "weighted_role_exposure": 0.60,
                "status_weight": 1.0,
                "roster_status": "ACT",
                "match_method": "EXACT",
                "is_qb": False,
                "excluded_manual_overlap": False,
            },
            {
                "team": "NE",
                "position_reported": "DE",
                "matched_position": "EDGE",
                "expected_role": 0.001,
                "weighted_role_exposure": 0.001,
                "status_weight": 1.0,
                "roster_status": "ACT",
                "match_method": "EXACT",
                "is_qb": False,
                "excluded_manual_overlap": False,
            },
            {
                "team": "NYJ",
                "position_reported": "S",
                "matched_position": None,
                "expected_role": None,
                "weighted_role_exposure": None,
                "status_weight": 0.35,
                "roster_status": None,
                "match_method": "UNMATCHED",
                "is_qb": False,
                "excluded_manual_overlap": False,
            },
        ]
    )

    shadow, decorated = build_personnel_v2_shadow(schedule, audit, 2026, 2)

    assert len(shadow) == 2 * len(POSITION_GROUPS)

    ne_ol = shadow.loc[
        shadow["team"].eq("NE") & shadow["position_group"].eq("OL")
    ].iloc[0]
    assert ne_ol["known_weighted_role_exposure"] == pytest.approx(0.60)

    ne_edge = shadow.loc[
        shadow["team"].eq("NE") & shadow["position_group"].eq("EDGE")
    ].iloc[0]
    assert ne_edge["review_count"] == 1
    assert ne_edge["active_low_role_prior_count"] == 1

    nyj_db = shadow.loc[
        shadow["team"].eq("NYJ") & shadow["position_group"].eq("DB")
    ].iloc[0]
    assert nyj_db["unresolved_status_weight"] == pytest.approx(0.35)
    assert nyj_db["review_count"] == 1

    flags = set(decorated["role_quality_flag"])
    assert "ACTIVE_LOW_ROLE_PRIOR" in flags
    assert "UNRESOLVED_ROLE" in flags


def test_qbs_stay_out_of_non_qb_shadow():
    audit = pd.DataFrame(
        [
            {
                "position_reported": "QB",
                "matched_position": "QB",
                "expected_role": 1.0,
                "is_qb": True,
                "excluded_manual_overlap": False,
            }
        ]
    )
    decorated = add_v2_shadow_features(audit)
    assert decorated.loc[0, "position_group"] == "QB"
    assert decorated.loc[0, "role_quality_flag"] == "EXCLUDED_QB"
    assert not bool(decorated.loc[0, "v2_shadow_eligible"])


def test_stale_roster_status_is_sent_to_review():
    audit = pd.DataFrame(
        [
            {
                "position_reported": "DT",
                "matched_position": "IDL",
                "expected_role": 0.001,
                "roster_status": "EXE",
                "match_method": "EXACT",
                "is_qb": False,
                "excluded_manual_overlap": False,
            }
        ]
    )
    decorated = add_v2_shadow_features(audit)
    assert decorated.loc[0, "role_quality_flag"] == "STALE_ROSTER_STATUS"


def test_reviewed_repair_changes_only_v2_shadow():
    schedule = pd.DataFrame(
        [{"game_id": "2026_02_NE_NYJ", "away_team": "NE", "home_team": "NYJ"}]
    )
    audit = pd.DataFrame(
        [
            {
                "team": "NE",
                "player_name": "Example Edge",
                "name_key": "example edge",
                "position_reported": "LB",
                "matched_position": "LB",
                "expected_role": 0.001,
                "weighted_role_exposure": 0.001,
                "status_weight": 1.0,
                "roster_status": "ACT",
                "match_method": "EXACT",
                "is_qb": False,
                "excluded_manual_overlap": False,
            }
        ]
    )
    repairs = pd.DataFrame(
        [
            {
                "team": "NE",
                "name_key": "example edge",
                "position_group_v2": "EDGE",
                "role_v2": 0.60,
                "repair_method": "validated_test",
                "evidence_source": "test",
                "review_status": "APPROVED_FOR_V2_SHADOW",
            }
        ]
    )

    shadow, decorated = build_personnel_v2_shadow(
        schedule, audit, 2026, 2, repairs
    )
    player = decorated.iloc[0]

    assert player["expected_role"] == pytest.approx(0.001)
    assert player["weighted_role_exposure"] == pytest.approx(0.001)
    assert player["expected_role_v1"] == pytest.approx(0.001)
    assert player["expected_role_v2"] == pytest.approx(0.60)
    assert player["position_group"] == "EDGE"
    assert player["role_quality_flag"] == "V2_ROLE_REPAIRED"

    edge = shadow.loc[
        shadow["team"].eq("NE") & shadow["position_group"].eq("EDGE")
    ].iloc[0]
    assert edge["known_weighted_role_exposure"] == pytest.approx(0.60)
    assert edge["review_count"] == 0


def test_role_repair_loader_respects_week_window(tmp_path):
    path = tmp_path / "repairs.csv"
    pd.DataFrame(
        [
            {
                "season": 2026,
                "start_week": 2,
                "end_week": 18,
                "player_name": "Example Edge",
                "team": "NE",
                "position_group_v2": "EDGE",
                "role_v2": 0.60,
                "repair_method": "validated_test",
                "evidence_source": "test",
                "review_status": "APPROVED_FOR_V2_SHADOW",
            }
        ]
    ).to_csv(path, index=False)

    assert load_v2_role_repairs(path, 2026, 1).empty
    week_two = load_v2_role_repairs(path, 2026, 2)
    assert len(week_two) == 1
    assert week_two.iloc[0]["name_key"] == "example edge"
