"""
V3.2 major defensive-attrition personnel adjustment.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

_POSITION_GROUP_MAP = {
    "C":"OL","G":"OL","OG":"OL","OL":"OL","OT":"OL","T":"OL",
    "WR":"WR_TE","TE":"WR_TE","RB":"RB","FB":"RB",
    "EDGE":"EDGE","DE":"EDGE","IDL":"IDL","DT":"IDL","NT":"IDL",
    "LB":"LB","ILB":"LB","OLB":"LB","CB":"DB","DB":"DB","S":"DB","FS":"DB","SS":"DB",
    "K":"ST","P":"ST","LS":"ST","QB":"QB",
}
DEFENSIVE_POSITION_GROUPS=frozenset({"EDGE","IDL","LB","DB"})

def normalize_position_group(value: object) -> str:
    position=str(value).upper().strip()
    if not position or position=="NAN":
        return "OTHER"
    return _POSITION_GROUP_MAP.get(position,"OTHER")

@dataclass(frozen=True)
class PersonnelCalibration:
    threshold: float
    adjustment_points: float
    def major_def_net(self, home_def_loss: pd.Series, away_def_loss: pd.Series) -> pd.Series:
        return (away_def_loss > self.threshold).astype(int) - (home_def_loss > self.threshold).astype(int)
    def adjustment(self, home_def_loss: pd.Series, away_def_loss: pd.Series) -> pd.Series:
        return self.adjustment_points*self.major_def_net(home_def_loss,away_def_loss)

@dataclass(frozen=True)
class ThresholdCalibrationCheck:
    n:int
    observed_flag_rate:float
    intended_flag_rate:float
    ci_lower:float
    ci_upper:float
    @property
    def intended_rate_within_ci(self)->bool:
        return self.ci_lower <= self.intended_flag_rate <= self.ci_upper

def check_threshold_calibration(observed_exposure: pd.Series, threshold: float, intended_flag_rate: float=0.10)->ThresholdCalibrationCheck:
    exposure=pd.to_numeric(observed_exposure,errors="coerce").dropna()
    n=len(exposure)
    if n==0:
        raise ValueError("observed_exposure must contain at least one numeric value")
    observed_rate=float((exposure>threshold).mean())
    z=1.96
    denom=1+z**2/n
    center=(observed_rate+z**2/(2*n))/denom
    margin=z*((observed_rate*(1-observed_rate)/n+z**2/(4*n**2))**0.5)/denom
    return ThresholdCalibrationCheck(n,observed_rate,intended_flag_rate,max(0.0,center-margin),min(1.0,center+margin))

def fit_personnel_adjustment(home_def_loss: pd.Series, away_def_loss: pd.Series, base_residual: pd.Series, threshold_quantile: float=0.90)->PersonnelCalibration:
    all_losses=pd.concat([home_def_loss,away_def_loss],ignore_index=True)
    threshold=float(all_losses.quantile(threshold_quantile))
    home_major=(home_def_loss>threshold).astype(int)
    away_major=(away_def_loss>threshold).astype(int)
    major_def_net=(away_major-home_major).to_numpy()
    denom=float(np.sum(major_def_net**2))
    if denom==0:
        raise ValueError("No games with a major defensive-loss signal in training data; cannot fit an adjustment coefficient.")
    coefficient=float(np.sum(major_def_net*base_residual.to_numpy())/denom)
    if coefficient<0:
        raise ValueError(f"Fitted personnel adjustment coefficient is negative ({coefficient:.4f}). Theory requires away-defense attrition to raise the home margin projection; a negative fit contradicts that and should be investigated, not flipped.")
    return PersonnelCalibration(threshold=threshold,adjustment_points=coefficient)
