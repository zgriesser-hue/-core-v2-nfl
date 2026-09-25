from __future__ import annotations

import pandas as pd
import pytest

from core_v2.v3_measurements import update_state_from_measurements


def test_measurement_update_is_reliability_weighted():
    state = pd.DataFrame([{
        "season": 2026, "week": 2, "team": "AAA",
        "offense_strength": 2.0, "defense_strength": 1.0,
        "qb_value": 0.0, "personnel_adjustment": 0.0,
        "market_strength": 0.0, "state_version": "test",
    }])
    measurement = pd.DataFrame([{
        "team": "AAA", "oa_offense_measurement": 0.10,
        "oa_defense_measurement": 0.05, "measurement_weight": 0.25,
    }])
    out = update_state_from_measurements(state, measurement, 20.0, 20.0)
    assert out.loc[0, "offense_strength"] == pytest.approx(2.0)
    assert out.loc[0, "defense_strength"] == pytest.approx(1.0)


def test_zero_weight_preserves_prior():
    state = pd.DataFrame([{
        "season": 2026, "week": 2, "team": "AAA",
        "offense_strength": 3.0, "defense_strength": -1.0,
        "qb_value": 0.0, "personnel_adjustment": 0.0,
        "market_strength": 0.0, "state_version": "test",
    }])
    measurement = pd.DataFrame([{
        "team": "AAA", "oa_offense_measurement": 9.0,
        "oa_defense_measurement": 9.0, "measurement_weight": 0.0,
    }])
    out = update_state_from_measurements(state, measurement, 99.0, 99.0)
    assert out.loc[0, "offense_strength"] == pytest.approx(3.0)
    assert out.loc[0, "defense_strength"] == pytest.approx(-1.0)
