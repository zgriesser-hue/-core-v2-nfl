from pathlib import Path

import numpy as np
import pandas as pd

from core_v2.projections import build_from_files


ROOT = Path(__file__).resolve().parents[1]


def test_week1_projection_parity() -> None:
    actual = build_from_files(
        ROOT / "data/schedules/schedule_2026.csv",
        ROOT / "data/model/team_ratings_2026.csv",
        ROOT / "data/production/2026/week_01/personnel_adjustments.csv",
        2026,
        1,
        ROOT / "data/model/historical_projection_locks.csv",
    )
    expected = pd.read_csv(
        ROOT / "tests/fixtures/week_01_expected_projections.csv"
    )
    comparison = actual.merge(
        expected[
            [
                "game_key",
                "fair_spread_home",
                "personnel_projection_status",
                "used_personnel_fallback",
                "projection_stage",
            ]
        ],
        on="game_key",
        how="outer",
        suffixes=("_actual", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    assert len(comparison) == 16
    assert comparison["_merge"].eq("both").all()
    assert np.allclose(
        comparison["fair_spread_home_actual"],
        comparison["fair_spread_home_expected"],
        atol=0.00051,
    )
    assert comparison["personnel_projection_status_actual"].eq(
        comparison["personnel_projection_status_expected"]
    ).all()
    assert comparison["used_personnel_fallback_actual"].eq(
        comparison["used_personnel_fallback_expected"]
    ).all()
    assert comparison["projection_display_status"].eq(
        comparison["projection_stage"]
    ).all()

