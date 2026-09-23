from __future__ import annotations

import pandas as pd

from core_v2.v3 import V3Config, no_update_state, project_week, run_walk_forward


def _schedule():
    return pd.DataFrame([
        {"season": 2025, "week": 1, "game_id": "g1", "away_team": "A", "home_team": "B"},
        {"season": 2025, "week": 2, "game_id": "g2", "away_team": "B", "home_team": "A"},
    ])


def _priors():
    return pd.DataFrame([
        {"season": 2025, "team": "A", "offense_strength": 2.0, "defense_strength": 1.0},
        {"season": 2025, "team": "B", "offense_strength": -1.0, "defense_strength": -0.5},
    ])


def test_projection_decomposition():
    state = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "A", "offense_strength": 2.0, "defense_strength": 1.0, "qb_value": 0.0, "personnel_adjustment": 0.0, "market_strength": 0.0, "state_version": "V3.0_FOUNDATION"},
        {"season": 2025, "week": 1, "team": "B", "offense_strength": -1.0, "defense_strength": -0.5, "qb_value": 0.0, "personnel_adjustment": 0.0, "market_strength": 0.0, "state_version": "V3.0_FOUNDATION"},
    ])
    board = project_week(_schedule().query("week == 1"), state, V3Config(default_hfa=2.2))
    # B offense - A defense - A offense + B defense = -4.5; plus HFA 2.2.
    assert board.loc[0, "persistent_team_component_home"] == -4.5
    assert board.loc[0, "projected_home_margin"] == -2.3
    assert board.loc[0, "fair_spread_home"] == 2.3


def test_walk_forward_freezes_before_update():
    projections, states = run_walk_forward(
        schedule=_schedule(),
        preseason_priors=_priors(),
        update_state=no_update_state,
        config=V3Config(default_hfa=2.2),
    )
    assert len(projections) == 2
    assert set(states["week"]) == {1, 2}
    a = states.loc[states["team"].eq("A")].sort_values("week")
    assert a["offense_strength"].tolist() == [2.0, 2.0]
    assert projections["state_version"].eq("V3.0_FOUNDATION").all()
