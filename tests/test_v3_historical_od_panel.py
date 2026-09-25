from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_v3_historical_od_panel import build_team_week_measurements, attach_future_unit_targets


def _games():
    return pd.DataFrame([
        {"season": 2025, "week": 1, "game_id": "g1", "team": "A", "opponent": "B", "epa_per_play": 0.20},
        {"season": 2025, "week": 1, "game_id": "g1", "team": "B", "opponent": "A", "epa_per_play": -0.10},
        {"season": 2025, "week": 2, "game_id": "g2", "team": "A", "opponent": "B", "epa_per_play": 0.30},
        {"season": 2025, "week": 2, "game_id": "g2", "team": "B", "opponent": "A", "epa_per_play": 0.00},
    ])


def test_week_one_has_no_current_game_information():
    panel = build_team_week_measurements(_games())
    w1 = panel.loc[panel["week"].eq(1)]
    assert w1["games"].eq(0).all()
    assert w1["oa_offense_measurement"].eq(0).all()
    assert w1["data_through_week"].eq(0).all()


def test_week_two_uses_week_one_only():
    panel = build_team_week_measurements(_games(), shrink_games=0.0)
    a = panel.loc[(panel["week"].eq(2)) & panel["team"].eq("A")].iloc[0]
    assert a["games"] == 1
    assert a["data_through_week"] == 1


def test_future_targets_are_labels_not_features():
    panel = build_team_week_measurements(_games())
    out = attach_future_unit_targets(panel, _games(), horizon_games=1)
    a1 = out.loc[(out["week"].eq(1)) & out["team"].eq("A")].iloc[0]
    assert a1["future_offense_epa"] == pytest.approx(0.20)
    assert a1["future_defense_epa"] == pytest.approx(0.10)
