"""
V3.2 - major defensive-attrition personnel adjustment.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PersonnelCalibration:
    threshold: float
    adjustment_points: float
    def major_def_net(self,home_def_loss:pd.Series,away_def_loss:pd.Series)->pd.Series:
        home_major=(home_def_loss>self.threshold).astype(int); away_major=(away_def_loss>self.threshold).astype(int)
        return away_major-home_major
    def adjustment(self,home_def_loss:pd.Series,away_def_loss:pd.Series)->pd.Series:
        return self.adjustment_points*self.major_def_net(home_def_loss,away_def_loss)

def fit_personnel_adjustment(home_def_loss:pd.Series,away_def_loss:pd.Series,base_residual:pd.Series,threshold_quantile:float=.90)->PersonnelCalibration:
    all_losses=pd.concat([home_def_loss,away_def_loss],ignore_index=True); threshold=float(all_losses.quantile(threshold_quantile))
    major_def_net=((away_def_loss>threshold).astype(int)-(home_def_loss>threshold).astype(int)).to_numpy()
    denom=float(np.sum(major_def_net**2))
    if denom==0: raise ValueError("No games with a major defensive-loss signal in training data; cannot fit an adjustment coefficient.")
    coefficient=float(np.sum(major_def_net*base_residual.to_numpy())/denom)
    if coefficient<0: raise ValueError(f"Fitted personnel adjustment coefficient is negative ({coefficient:.4f}).")
    return PersonnelCalibration(threshold=threshold,adjustment_points=coefficient)
