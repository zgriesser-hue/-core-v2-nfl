"""
V3 personnel engine.\n\nHistorical V3.2 used a binary major-defensive-attrition rule. Native V3\nproduction now uses the separately validated fixed continuous defensive-PVAR\nspecification: 2.25 * (away defensive loss - home defensive loss). The old\nV3.2 calibration helpers remain here as frozen research/backtest compatibility.

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
DEFENSIVE_POSITION_GROUPS = frozenset({"EDGE", "IDL", "LB", "DB"})
NATIVE_V3_DEFENSIVE_PVAR_POINTS = 2.25


def continuous_personnel_adjustment(
    home_def_loss: object,
    away_def_loss: object,
    points_per_pvar: float = NATIVE_V3_DEFENSIVE_PVAR_POINTS,
) -> pd.Series:
    """Fixed native-V3 defensive personnel adjustment in home-margin points."""
    home = pd.to_numeric(pd.Series(home_def_loss), errors="coerce")
    away = pd.to_numeric(pd.Series(away_def_loss), errors="coerce")
    if home.isna().any() or away.isna().any():
        raise ValueError(
            "Native V3 personnel loss is missing; refusing to treat missing data as healthy."
        )
    return points_per_pvar * (away - home)

def normalize_position_group(value: object) -> str:
    position=str(value).upper().strip()
    if not position or position=="NAN":
        return "OTHER"
    return _POSITION_GROUP_MAP.get(position,"OTHER")

def attach_player_pvar_to_audit(audit: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative 2026 Stage-2i PVAR values to the weekly audit by GSIS ID."""
    required = {"gsis_id", "pvar_rating_rebased", "pvar_rating_role_only", "pvar_confidence"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"V3 PVAR ratings missing columns: {sorted(missing)}")

    out = audit.copy()
    out["gsis_id"] = out["gsis_id"].astype("string")
    values = ratings.copy()
    values["gsis_id"] = values["gsis_id"].astype("string")
    values = values.loc[values["gsis_id"].notna()].copy()
    if values["gsis_id"].duplicated().any():
        dupes = sorted(values.loc[values["gsis_id"].duplicated(False), "gsis_id"].unique())
        raise ValueError(f"V3 PVAR ratings contain duplicate GSIS IDs: {dupes[:10]}")

    keep = ["gsis_id", "pvar_rating_rebased", "pvar_rating_role_only", "pvar_confidence"]
    out = out.merge(values[keep], on="gsis_id", how="left", validate="many_to_one")
    out["pvar_rating_rebased"] = pd.to_numeric(out["pvar_rating_rebased"], errors="coerce")
    out["pvar_rating_role_only"] = pd.to_numeric(out["pvar_rating_role_only"], errors="coerce")
    return out


def defensive_exposure_from_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate status-weighted player PVAR into native V3 defensive loss."""
    columns = [
        "team", "def_loss_pvar", "def_max_player_loss", "def_high_role_count",
        "def_edge_loss", "def_idl_loss", "def_lb_loss", "def_db_loss",
        "personnel_players_observed",
    ]
    if audit is None or audit.empty:
        return pd.DataFrame(columns=columns)

    df = audit.copy()
    pos_source = (
        "position_group" if "position_group" in df.columns
        else "matched_position" if "matched_position" in df.columns
        else "position_reported"
    )
    df["_group"] = df[pos_source].map(normalize_position_group)

    if "v2_shadow_eligible" in df.columns:
        eligible = df["v2_shadow_eligible"].map(
            lambda x: x if isinstance(x, (bool, np.bool_))
            else str(x).strip().lower() in {"true", "1", "yes", "y"}
        ).fillna(False)
    else:
        eligible = pd.Series(True, index=df.index)

    df = df.loc[df["_group"].isin(DEFENSIVE_POSITION_GROUPS) & eligible].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    required = {"status_weight", "pvar_production"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Native V3 personnel audit missing PVAR inputs: {sorted(missing)}")

    df["_status_weight"] = pd.to_numeric(df["status_weight"], errors="coerce")
    df["_pvar"] = pd.to_numeric(df["pvar_production"], errors="coerce")
    unresolved = df["_status_weight"].isna() | df["_pvar"].isna() | df["gsis_id"].isna()
    if unresolved.any():
        names = df.loc[unresolved, ["team", "player_name"]].astype(str).agg(" ".join, axis=1).tolist()
        raise ValueError(
            "Native V3 eligible defensive players have unresolved status/PVAR/GSIS: "
            + ", ".join(names[:20])
        )

    # This is the established Stage-2i exposure construction. Do NOT multiply
    # by expected_role/weighted_role_exposure_v2: the historical V3 coefficient
    # was trained on status-weighted PVAR loss.
    df["_loss_pvar"] = df["_status_weight"].clip(lower=0.0) * df["_pvar"].clip(lower=0.0)

    rows = []
    for team_name, group_df in df.groupby("team", dropna=False):
        rec = {
            "team": team_name,
            "def_loss_pvar": float(group_df["_loss_pvar"].sum()),
            "def_max_player_loss": float(group_df["_loss_pvar"].max()) if len(group_df) else 0.0,
            "def_high_role_count": int((group_df["_loss_pvar"] >= 0.50).sum()),
            "personnel_players_observed": int(len(group_df)),
        }
        for group, col in [
            ("EDGE", "def_edge_loss"), ("IDL", "def_idl_loss"),
            ("LB", "def_lb_loss"), ("DB", "def_db_loss"),
        ]:
            rec[col] = float(group_df.loc[group_df["_group"].eq(group), "_loss_pvar"].sum())
        rows.append(rec)
    return pd.DataFrame(rows, columns=columns)


def add_personnel_to_matchups(
    matchups: pd.DataFrame,
    audit: pd.DataFrame,
    covered_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Join native defensive exposure features without treating missing coverage as healthy."""
    out = matchups.copy()
    team = defensive_exposure_from_audit(audit)
    feature_cols = [col for col in team.columns if col != "team"]

    if covered_teams is None:
        covered_teams = (
            set(audit["team"].dropna().astype(str))
            if audit is not None and "team" in audit.columns
            else set()
        )
    else:
        covered_teams = {str(team_name) for team_name in covered_teams}

    for side in ("home", "away"):
        renamed = team.rename(
            columns={
                "team": f"{side}_team",
                **{col: f"{side}_{col}" for col in feature_cols},
            }
        )
        out = out.merge(renamed, on=f"{side}_team", how="left")
        covered = out[f"{side}_team"].astype(str).isin(covered_teams)

        for col in feature_cols:
            target = f"{side}_{col}"
            out[target] = pd.to_numeric(out[target], errors="coerce")
            out.loc[covered, target] = out.loc[covered, target].fillna(0.0)

        observed = out[f"{side}_personnel_players_observed"].fillna(0).gt(0)
        out[f"{side}_personnel_provenance"] = np.select(
            [~covered, observed],
            ["missing_team_coverage", "player_audit_native_v3"],
            default="covered_no_defensive_losses",
        )
        out[f"{side}_def_loss"] = out[f"{side}_def_loss_pvar"]

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
