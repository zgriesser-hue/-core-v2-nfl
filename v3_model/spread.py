"""
V3.0 - EPA-strength-to-points margin calibration.

Extracted from V3.ipynb, cell "V3.0 - LOSO GAME-MARGIN CALIBRATION":

    home_margin = HFA + beta * v3_strength_diff

where v3_strength_diff = (home_offense + home_defense) -
(away_offense + away_defense), using the pregame states from
rating.py.

The notebook fit HFA/beta separately per LOSO fold (one value per
held-out season). For production there's no fold to hold out, so
`fit_hfa_beta` here is the same OLS fit, meant to be run ONCE on the
full available history to produce a single (hfa, beta) pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpreadCalibration:
    hfa: float
    beta: float

    def predict(self, strength_diff: pd.Series | np.ndarray) -> np.ndarray:
        return self.hfa + self.beta * np.asarray(strength_diff)


def strength_diff(
    home_offense: pd.Series,
    home_defense: pd.Series,
    away_offense: pd.Series,
    away_defense: pd.Series,
) -> pd.Series:
    """v3_strength_diff, as defined in the notebook's game-design panel."""
    return (home_offense + home_defense) - (away_offense + away_defense)


def fit_hfa_beta(
    home_margin: pd.Series, v3_strength_diff: pd.Series
) -> SpreadCalibration:
    """
    OLS fit of home_margin = HFA + beta * strength_diff.

    Run this once on your full training history (not per-fold) to
    get the single production (hfa, beta) pair. To reproduce the
    notebook's historical LOSO numbers instead, call this once per
    held-out season using only the other seasons' rows.
    """
    X = np.column_stack([np.ones(len(v3_strength_diff)), v3_strength_diff.to_numpy()])
    y = home_margin.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return SpreadCalibration(hfa=float(coef[0]), beta=float(coef[1]))


def loso_fit_hfa_beta(
    game_panel: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """
    Reproduce the notebook's exact LOSO procedure (one HFA/beta pair
    per held-out season, fit on every OTHER season). Useful for
    regression-testing against the historical numbers in
    `v3_0_frozen_states.parquet` / the notebook's printed results,
    NOT for choosing the production single value (use
    `fit_hfa_beta` on full history for that).

    `game_panel` needs columns: season, home_margin, v3_strength_diff.
    """
    rows = []
    for holdout in seasons:
        train = game_panel[game_panel["season"] != holdout].dropna(
            subset=["v3_strength_diff", "home_margin"]
        )
        cal = fit_hfa_beta(train["home_margin"], train["v3_strength_diff"])
        rows.append({"holdout_season": holdout, "hfa": cal.hfa, "beta": cal.beta})

    return pd.DataFrame(rows)
