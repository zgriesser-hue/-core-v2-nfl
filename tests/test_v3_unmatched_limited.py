import pandas as pd

from scripts.build_v3_weekly_board import defer_unrated_defenders


def test_unrated_defenders_are_deferred_without_changing_source_audit():
    audit = pd.DataFrame([
        dict(team="JAX", player_name="Jaylon Jones", gsis_id=None,
             position_group="DB", v2_shadow_eligible=True, status_weight=0.05),
        dict(team="JAX", player_name="Confirmed out", gsis_id=None,
             position_group="DB", v2_shadow_eligible=True, status_weight=1.0),
        dict(team="JAX", player_name="Known defender", gsis_id="D1",
             position_group="DB", v2_shadow_eligible=True, status_weight=1.0),
        dict(team="JAX", player_name="Unrated ID", gsis_id="D2",
             position_group="DB", v2_shadow_eligible=True, status_weight=0.5),
        dict(team="JAX", player_name="Unrated OL", gsis_id=None,
             position_group="OL", v2_shadow_eligible=True, status_weight=1.0),
    ])
    ratings = pd.DataFrame([
        dict(gsis_id="D1", pvar_production=0.8),
        dict(gsis_id="D2", pvar_production=float("nan")),
    ])
    prepared, deferred = defer_unrated_defenders(audit, ratings)

    assert deferred.player_name.tolist() == [
        "Jaylon Jones", "Confirmed out", "Unrated ID"
    ]
    assert prepared.v2_shadow_eligible.tolist() == [False, False, True, False, True]
    assert audit.v2_shadow_eligible.all()
