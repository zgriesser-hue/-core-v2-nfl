"""
V3.2 - major defensive-attrition personnel adjustment.

Extracted from V3.ipynb, cell "V3.2d - FINAL MAJOR DEFENSIVE
ATTRITION SPREAD TEST".

Logic (per LOSO fold in the notebook; here, meant to be fit ONCE on
full history for production):

1. Take each team's `def_loss_pvar` (personnel-value-weighted
   defensive exposure lost to injury) for the game.
2. Flag a team as having a "major" defensive loss if its exposure
   exceeds the 90th percentile of exposure seen in training data
   (frozen threshold quantile = 0.90).
3. `major_def_net = away_major - home_major` (positive => away
   defense is the one depleted, which should help the home team).
4. Fit a single coefficient (no intercept) of
   `base_residual ~ major_def_net`, where base_residual is the
   error of the V3.0 base projection (pred_v30) against actual
   home_margin.
5. Apply: `personnel_adjustment = coefficient * major_def_net`.

The notebook explicitly required the fitted coefficient to come out
POSITIVE (away defense depleted -> higher home margin) and would
NOT silently flip a contradictory sign. `fit_personnel_adjustment`
preserves that check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PersonnelCalibration:
    threshold: float  # def_loss_pvar value above which a loss counts as "major"
    adjustment_points: float  # points per unit of major_def_net

    def major_def_net(
        self, home_def_loss: pd.Series, away_def_loss: pd.Series
    ) -> pd.Series:
        home_major = (home_def_loss > self.threshold).astype(int)
        away_major = (away_def_loss > self.threshold).astype(int)
        return away_major - home_major

    def adjustment(
        self, home_def_loss: pd.Series, away_def_loss: pd.Series
    ) -> pd.Series:
        return self.adjustment_points * self.major_def_net(
            home_def_loss, away_def_loss
        )


def fit_personnel_adjustment(
    home_def_loss: pd.Series,
    away_def_loss: pd.Series,
    base_residual: pd.Series,
    threshold_quantile: float = 0.90,
) -> PersonnelCalibration:
    """
    Fit the major-defensive-attrition adjustment on a training set.

    `base_residual` = actual home_margin - V3.0 base projection
    (pred_v30), i.e. what's left for personnel to explain.

    Raises ValueError if the fitted coefficient comes out negative -
    the notebook treated a contradictory sign as a reason to reject
    the adjustment rather than silently flip it, and this function
    does the same.
    """
    all_losses = pd.concat([home_def_loss, away_def_loss], ignore_index=True)
    threshold = float(all_losses.quantile(threshold_quantile))

    home_major = (home_def_loss > threshold).astype(int)
    away_major = (away_def_loss > threshold).astype(int)
    major_def_net = (away_major - home_major).to_numpy()

    # No-intercept least squares: coefficient = sum(x*y) / sum(x*x)
    denom = float(np.sum(major_def_net**2))
    if denom == 0:
        raise ValueError(
            "No games with a major defensive-loss signal in training data; "
            "cannot fit an adjustment coefficient."
        )

    coefficient = float(np.sum(major_def_net * base_residual.to_numpy()) / denom)

    if coefficient < 0:
        raise ValueError(
            f"Fitted personnel adjustment coefficient is negative "
            f"({coefficient:.4f}). Theory requires away-defense attrition "
            f"to raise the home margin projection; a negative fit "
            f"contradicts that and should be investigated, not flipped."
        )

    return PersonnelCalibration(threshold=threshold, adjustment_points=coefficient)
