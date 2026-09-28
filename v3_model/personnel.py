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


_POSITION_GROUP_MAP = {"C":"OL","G":"OL","OG":"OL","OL":"OL","OT":"OL","T":"OL","WR":"WR_TE","TE":"WR_TE","RB":"RB","FB":"RB","EDGE":"EDGE","DE":"EDGE","IDL":"IDL","DT":"IDL","NT":"IDL","LB":"LB","ILB":"LB","OLB":"LB","CB":"DB","DB":"DB","S":"DB","FS":"DB","SS":"DB","K":"ST","P":"ST","LS":"ST","QB":"QB"}
DEFENSIVE_POSITION_GROUPS=frozenset({"EDGE","IDL","LB","DB"})

def normalize_position_group(value: object) -> str:
    position=str(value).upper().strip()
    if not position or position=="NAN":
        return "OTHER"
    return _POSITION_GROUP_MAP.get(position,"OTHER")

def defensive_exposure_from_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the weekly player audit into native V3 defensive features."""
    columns=["team","def_loss_pvar","def_max_player_loss","def_high_role_count","def_edge_loss","def_idl_loss","def_lb_loss","def_db_loss","personnel_players_observed"]
    if audit is None or audit.empty:
        return pd.DataFrame(columns=columns)
    df=audit.copy()
    pos_source="position_group" if "position_group" in df.columns else ("matched_position" if "matched_position" in df.columns else "position_reported")
    df["_group"]=df[pos_source].map(normalize_position_group)
    eligible=df["v2_shadow_eligible"].fillna(False).astype(bool) if "v2_shadow_eligible" in df.columns else pd.Series(True,index=df.index)
    df=df.loc[df["_group"].isin(DEFENSIVE_POSITION_GROUPS) & eligible].copy()
    exposure_col="weighted_role_exposure_v2" if "weighted_role_exposure_v2" in df.columns else "weighted_role_exposure"
    df["_exposure"]=pd.to_numeric(df[exposure_col],errors="coerce").fillna(0.0).clip(lower=0.0)
    rows=[]
    for team,g in df.groupby("team",dropna=False):
        rec={"team":team,"def_loss_pvar":float(g["_exposure"].sum()),"def_max_player_loss":float(g["_exposure"].max()) if len(g) else 0.0,"def_high_role_count":int((g["_exposure"]>=0.50).sum()),"personnel_players_observed":int(len(g))}
        for group,col in [("EDGE","def_edge_loss"),("IDL","def_idl_loss"),("LB","def_lb_loss"),("DB","def_db_loss")]:
            rec[col]=float(g.loc[g["_group"].eq(group),"_exposure"].sum())
        rows.append(rec)
    return pd.DataFrame(rows,columns=columns)

def add_personnel_to_matchups(matchups: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Join native defensive exposure features to a weekly matchup table."""
    out=matchups.copy()
    team=defensive_exposure_from_audit(audit)
    feature_cols=[c for c in team.columns if c!="team"]
    for side in ("home","away"):
        renamed=team.rename(columns={"team":f"{side}_team",**{c:f"{side}_{c}" for c in feature_cols}})
        out=out.merge(renamed,on=f"{side}_team",how="left")
        for col in feature_cols:
            out[f"{side}_{col}"]=pd.to_numeric(out[f"{side}_{col}"],errors="coerce").fillna(0.0)
        observed=out[f"{side}_personnel_players_observed"].gt(0)
        out[f"{side}_personnel_provenance"]=np.where(observed,"player_audit_native_v3","player_audit_no_defensive_losses")
        out[f"{side}_def_loss"]=out[f"{side}_def_loss_pvar"]
    return out

@dataclass(frozen=True)
class ThresholdCalibrationCheck:
    n:int
    observed_flag_rate:float
    intended_flag_rate:float
    ci_lower:float
    ci_upper:float
    @property
    def intended_rate_within_ci(self)->bool:
        return self.n>0 and self.ci_lower <= self.intended_flag_rate <= self.ci_upper

def check_threshold_calibration(losses: pd.Series, threshold: float, intended_flag_rate: float=0.10) -> ThresholdCalibrationCheck:
    """Wilson 95% interval for live major-loss frequency; safe for empty input."""
    clean=pd.to_numeric(losses,errors="coerce").dropna()
    n=int(len(clean))
    if n==0:
        return ThresholdCalibrationCheck(0,float("nan"),intended_flag_rate,float("nan"),float("nan"))
    p=float((clean>threshold).mean())
    z=1.959963984540054
    denom=1.0+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/denom
    return ThresholdCalibrationCheck(n,p,intended_flag_rate,max(0.0,center-half),min(1.0,center+half))
