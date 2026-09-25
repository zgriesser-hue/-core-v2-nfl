from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .efficiency import build_team_game_epa, solve_oa_epa


@dataclass(frozen=True)
class V3MeasurementConfig:
    """Research-only measurement settings. No point-conversion is assumed."""

    shrink_games: float = 4.0
    prior_equivalent_games: float = 6.0
    max_measurement_weight: float = 0.65


def build_oa_measurements(
    pbp: pd.DataFrame,
    season: int,
    through_week: int,
    config: V3MeasurementConfig | None = None,
) -> pd.DataFrame:
    """Create opponent-adjusted offense/defense measurements from prior weeks.

    EPA is kept in EPA/play units here. V3 does not add EPA directly to spread
    points; the calibration layer must learn how these measurements relate to
    latent team strength.
    """
    config = config or V3MeasurementConfig()
    games = build_team_game_epa(pbp, season, through_week)
    oa = solve_oa_epa(games, shrink_games=config.shrink_games)
    if oa.empty:
        return pd.DataFrame(columns=[
            "team", "oa_offense_measurement", "oa_defense_measurement",
            "games", "measurement_weight", "data_through_week",
        ])
    out = oa.rename(columns={
        "oa_offense": "oa_offense_measurement",
        "oa_defense": "oa_defense_measurement",
    }).copy()
    n = pd.to_numeric(out["games"], errors="coerce").fillna(0.0)
    out["measurement_weight"] = (
        float(config.max_measurement_weight)
        * n / (n + float(config.prior_equivalent_games))
    )
    out["data_through_week"] = int(through_week)
    return out[[
        "team", "oa_offense_measurement", "oa_defense_measurement",
        "games", "measurement_weight", "data_through_week",
    ]].sort_values("team").reset_index(drop=True)


def update_state_from_measurements(
    state: pd.DataFrame,
    measurements: pd.DataFrame,
    offense_points_per_epa: float,
    defense_points_per_epa: float,
) -> pd.DataFrame:
    """Reliability-weight V3 priors toward calibrated OA-EPA measurements.

    The EPA->points coefficients are explicit arguments on purpose. They must
    come from historical walk-forward calibration, never from the current
    season being projected.
    """
    required = {"team", "offense_strength", "defense_strength"}
    missing = required - set(state.columns)
    if missing:
        raise ValueError(f"state missing columns: {sorted(missing)}")
    m_required = {
        "team", "oa_offense_measurement", "oa_defense_measurement",
        "measurement_weight",
    }
    missing = m_required - set(measurements.columns)
    if missing:
        raise ValueError(f"measurements missing columns: {sorted(missing)}")

    out = state.copy()
    m = measurements.copy()
    out = out.merge(m, on="team", how="left", validate="one_to_one")
    out["measurement_weight"] = pd.to_numeric(
        out["measurement_weight"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    w = out["measurement_weight"]

    off_target = (
        pd.to_numeric(out["oa_offense_measurement"], errors="coerce")
        * float(offense_points_per_epa)
    )
    def_target = (
        pd.to_numeric(out["oa_defense_measurement"], errors="coerce")
        * float(defense_points_per_epa)
    )
    off_target = off_target.fillna(out["offense_strength"])
    def_target = def_target.fillna(out["defense_strength"])

    out["offense_strength"] = (1.0 - w) * out["offense_strength"] + w * off_target
    out["defense_strength"] = (1.0 - w) * out["defense_strength"] + w * def_target
    return out[state.columns].copy()
