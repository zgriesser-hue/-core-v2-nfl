"""
V3.6 - "Pure V3" integration.

Extracted from V3.ipynb, cell "V3.6B - BUILD FROZEN INTEGRATED PURE
V3 HISTORICAL CANDIDATE".

    pred_v32          = pred_v30 + personnel_adjustment
    pred_v33          = pred_v30 + market_adjustment
    pred_v36          = pred_v30 + personnel_adjustment
                                 + market_adjustment
                                 + neutral_hfa_adjustment

This module only does the arithmetic combination; it does not fit
anything. Callers supply pred_v30 (from rating.py + spread.py) and
each adjustment (from personnel.py, market.py, neutral.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_v36(
    pred_v30: pd.Series | np.ndarray,
    personnel_adjustment: pd.Series | np.ndarray,
    market_adjustment: pd.Series | np.ndarray,
    neutral_hfa_adjustment: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Combine V3.0's base projection with the three frozen V3.2/V3.3/
    V3.4 adjustments into the final V3.6 projected home margin.

    All four inputs must be the same length / aligned by game.
    """
    pred_v30 = np.asarray(pred_v30, dtype=float)
    personnel_adjustment = np.asarray(personnel_adjustment, dtype=float)
    market_adjustment = np.asarray(market_adjustment, dtype=float)
    neutral_hfa_adjustment = np.asarray(neutral_hfa_adjustment, dtype=float)

    if not (
        pred_v30.shape
        == personnel_adjustment.shape
        == market_adjustment.shape
        == neutral_hfa_adjustment.shape
    ):
        raise ValueError(
            "All four inputs to predict_v36 must have the same shape; got "
            f"{pred_v30.shape}, {personnel_adjustment.shape}, "
            f"{market_adjustment.shape}, {neutral_hfa_adjustment.shape}"
        )

    return (
        pred_v30 + personnel_adjustment + market_adjustment + neutral_hfa_adjustment
    )
