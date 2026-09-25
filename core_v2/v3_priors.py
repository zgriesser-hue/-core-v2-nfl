from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeepDivePriorConfig:
    """Research-only controls for decomposing the whole-team preseason anchor."""

    composition_scale_points: float = 1.0


def _sheet(path: Path, name: str) -> pd.DataFrame:
    # Workbook sheets have a title row followed by the real header.
    frame = pd.read_excel(path, sheet_name=name, header=1)
    if "Team" not in frame.columns:
        raise ValueError(f"{name} sheet is missing Team")
    frame = frame.loc[frame["Team"].notna()].copy()
    frame["Team"] = frame["Team"].astype(str).str.strip().str.upper()
    # Keep only actual NFL team rows; the Power Ratings sheet may contain a
    # league-average/footer row.
    frame = frame.loc[frame["Team"].str.fullmatch(r"[A-Z]{2,3}")].copy()
    if frame["Team"].duplicated().any():
        raise ValueError(f"{name} sheet contains duplicate teams")
    return frame


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - mean) / std


def _status_confidence(status: object) -> float:
    text = str(status).strip().lower()
    if not text or text == "nan":
        return 0.50
    if "inference" in text or "provisional" in text:
        return 0.65
    if "partial" in text or "approx" in text:
        return 0.80
    if "verified" in text or "exact" in text:
        return 1.00
    return 0.75


def load_deep_dive_priors(
    workbook: str | Path,
    season: int,
    config: DeepDivePriorConfig | None = None,
) -> pd.DataFrame:
    """Load Andy/Drew preseason ratings and reconcile them to V3 state.

    The centered Andy/Drew whole-team rating is the authoritative total anchor.
    Unit grades describe *composition* only. They are not allowed to change the
    team's total Week 1 strength.

    offense_strength + defense_strength == overall_prior exactly.

    The offense/defense split uses league-standardized unit composites. Its
    scale is intentionally explicit and research-only so it can later be
    validated rather than silently baked into production.
    """
    config = config or DeepDivePriorConfig()
    path = Path(workbook)

    power = _sheet(path, "Power Ratings")
    offense = _sheet(path, "Offense")
    defense = _sheet(path, "Defense")
    other = _sheet(path, "Other")

    p = power.rename(columns={
        "Team": "team",
        "Team Name": "team_name",
        "Andy Rating": "andy_rating",
        "Drew Rating": "drew_rating",
        "Pod Consensus": "pod_consensus",
        "Andy–Drew Gap": "andy_drew_gap",
        "Centered Week 1 Prior": "overall_prior",
        "Status": "overall_status",
    })
    o = offense.rename(columns={
        "Team": "team",
        "QB/OC Consensus": "qb_oc_prior",
        "Skill Consensus": "skill_prior",
        "OL Consensus": "ol_prior",
        "Extraction Status": "offense_status",
    })
    d = defense.rename(columns={
        "Team": "team",
        "Pass Rush Consensus": "pass_rush_prior",
        "Coverage Consensus": "coverage_prior",
        "Run Defense Consensus": "run_defense_prior",
        "Extraction Status": "defense_status",
    })
    x = other.rename(columns={
        "Team": "team",
        "Coaching Consensus": "coaching_prior",
        "Special Teams Consensus": "special_teams_prior",
        "Extraction Status": "other_status",
    })

    keep_p = ["team", "team_name", "andy_rating", "drew_rating", "pod_consensus",
              "andy_drew_gap", "overall_prior", "overall_status"]
    keep_o = ["team", "qb_oc_prior", "skill_prior", "ol_prior", "offense_status"]
    keep_d = ["team", "pass_rush_prior", "coverage_prior", "run_defense_prior", "defense_status"]
    keep_x = ["team", "coaching_prior", "special_teams_prior", "other_status"]

    out = p[keep_p].merge(o[keep_o], on="team", how="left", validate="one_to_one")
    out = out.merge(d[keep_d], on="team", how="left", validate="one_to_one")
    out = out.merge(x[keep_x], on="team", how="left", validate="one_to_one")

    numeric = [
        "andy_rating", "drew_rating", "pod_consensus", "andy_drew_gap", "overall_prior",
        "qb_oc_prior", "skill_prior", "ol_prior", "pass_rush_prior",
        "coverage_prior", "run_defense_prior", "coaching_prior", "special_teams_prior",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out["overall_prior"].isna().any():
        bad = out.loc[out["overall_prior"].isna(), "team"].tolist()
        raise ValueError(f"Missing whole-team Andy/Drew prior for teams: {bad}")

    # Missing unit cells are filled with the league mean for composition only.
    # The raw columns remain NaN so uncertainty is preserved and auditable.
    offense_cols = ["qb_oc_prior", "skill_prior", "ol_prior"]
    defense_cols = ["pass_rush_prior", "coverage_prior", "run_defense_prior"]
    for column in offense_cols + defense_cols:
        out[f"{column}_filled"] = out[column].fillna(out[column].mean())

    out["offense_unit_composite"] = out[[f"{c}_filled" for c in offense_cols]].mean(axis=1)
    out["defense_unit_composite"] = out[[f"{c}_filled" for c in defense_cols]].mean(axis=1)
    out["offense_unit_z"] = _zscore(out["offense_unit_composite"])
    out["defense_unit_z"] = _zscore(out["defense_unit_composite"])

    # composition_delta is offense minus defense. Split it symmetrically around
    # half the whole-team anchor so the anchor is preserved exactly.
    out["composition_delta"] = (
        out["offense_unit_z"] - out["defense_unit_z"]
    ) * float(config.composition_scale_points)
    out["offense_strength"] = 0.5 * (out["overall_prior"] + out["composition_delta"])
    out["defense_strength"] = 0.5 * (out["overall_prior"] - out["composition_delta"])
    out["reconciled_total"] = out["offense_strength"] + out["defense_strength"]
    if not np.allclose(out["reconciled_total"], out["overall_prior"], atol=1e-12):
        raise AssertionError("V3 preseason decomposition failed to preserve whole-team anchor")

    out["overall_confidence"] = out["overall_status"].map(_status_confidence)
    out["offense_confidence"] = out["offense_status"].map(_status_confidence)
    out["defense_confidence"] = out["defense_status"].map(_status_confidence)
    out["other_confidence"] = out["other_status"].map(_status_confidence)
    out["unit_missing_count"] = out[
        offense_cols + defense_cols + ["coaching_prior", "special_teams_prior"]
    ].isna().sum(axis=1)
    out["andy_drew_disagreement"] = out["andy_drew_gap"].abs()
    out["season"] = int(season)
    out["prior_source"] = "DEEP_DIVE_ANDY_DREW"
    out["prior_version"] = "V3_PRESEASON_V1"

    return out.sort_values("team").reset_index(drop=True)
