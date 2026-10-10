import pandas as pd

from scripts.build_v3_weekly_board import defer_unmatched_limited


def test_only_undesignated_unmatched_limited_defender_is_deferred():
    rows = [
        ("Jaylon Jones", None, 0.05, None, "Limited Participation in Practice", "UNMATCHED"),
        ("Confirmed out", None, 1.0, "Out", "Did Not Participate In Practice", "UNMATCHED"),
        ("High weight", None, 0.50, None, "Limited Participation in Practice", "UNMATCHED"),
        ("Matched limited", "00-1", 0.05, None, "Limited Participation in Practice", "EXACT"),
    ]
    audit = pd.DataFrame([
        dict(team="JAX", player_name=name, gsis_id=gsis_id, status_weight=weight,
             game_status=game_status, practice_status=practice_status,
             match_method=method, position_group="DB", v2_shadow_eligible=True)
        for name, gsis_id, weight, game_status, practice_status, method in rows
    ])
    prepared, deferred = defer_unmatched_limited(audit)

    assert deferred[["team", "player_name"]].to_dict("records") == [
        {"team": "JAX", "player_name": "Jaylon Jones"}
    ]
    assert prepared.v2_shadow_eligible.tolist() == [False, True, True, True]
    assert audit.v2_shadow_eligible.all()  # Preserve the source audit.
