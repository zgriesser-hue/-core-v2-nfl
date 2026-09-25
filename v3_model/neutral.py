"""
V3.4 - true-neutral-site HFA removal.

Extracted from V3.ipynb, cell "V3.4j - STRUCTURAL NEUTRAL-SITE HFA
CORRECTION".

This is a STRUCTURAL rule, not a fitted one: since
`pred_v30 = season_HFA + season_scale * strength_diff` exactly (by
construction of spread.SpreadCalibration), a true neutral-site game
should simply have the HFA term removed:

    neutral_hfa_adjustment = -HFA   if neutral site
                            =  0     otherwise
    pred_with_neutral_correction = pred_v30 + neutral_hfa_adjustment

The notebook notes this was NOT chosen because it beat V3.0 on MAE
in backtesting (the neutral-game sample is small) - it's an
architectural choice: home-field advantage should not apply when
there is no home field. Kept exactly as designed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutral_hfa_adjustment(
    is_neutral_site: pd.Series | np.ndarray, hfa: float
) -> np.ndarray:
    """
    -hfa where the game is at a neutral site, 0 elsewhere.
    """
    is_neutral = np.asarray(is_neutral_site).astype(bool)
    return np.where(is_neutral, -hfa, 0.0)
