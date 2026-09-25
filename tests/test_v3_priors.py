from __future__ import annotations

import pandas as pd

from core_v2.v3_priors import _status_confidence, _zscore


def test_zscore_centers_league():
    z = _zscore(pd.Series([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-12


def test_confidence_marks_inference_below_verified():
    assert _status_confidence("Complete — includes inference") < _status_confidence("Verified exact")
