from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def _require(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _fit_train_mapping(train: pd.DataFrame) -> tuple[float, float]:
    """Map OA offense/defense EPA measurements to point-valued latent strength.

    Fit only on outer-training seasons. Targets are historical offense/defense
    point-strength targets supplied by the research dataset; never use the held
    out season to estimate these coefficients.
    """
    x_off = pd.to_numeric(train["oa_offense_measurement"], errors="coerce")
    y_off = pd.to_numeric(train["offense_target_points"], errors="coerce")
    x_def = pd.to_numeric(train["oa_defense_measurement"], errors="coerce")
    y_def = pd.to_numeric(train["defense_target_points"], errors="coerce")

    def slope(x: pd.Series, y: pd.Series) -> float:
        ok = x.notna() & y.notna()
        if ok.sum() < 20:
            raise ValueError("Not enough training observations for calibration")
        xv = x.loc[ok].to_numpy(float)
        yv = y.loc[ok].to_numpy(float)
        denom = float(np.dot(xv, xv))
        if denom <= 0:
            raise ValueError("Degenerate calibration feature")
        return float(np.dot(xv, yv) / denom)

    return slope(x_off, y_off), slope(x_def, y_def)


def evaluate_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Nested LOSO calibration of prior decay and OA-EPA point mapping.

    Required input is one row per team-week *entering* a game/week. The
    measurement columns must be constructed only from earlier completed weeks.
    """
    _require(data, {
        "season", "week", "team", "games",
        "prior_offense_strength", "prior_defense_strength",
        "oa_offense_measurement", "oa_defense_measurement",
        "offense_target_points", "defense_target_points",
    }, "calibration data")

    d = data.copy()
    d["season"] = pd.to_numeric(d["season"], errors="raise").astype(int)
    d["week"] = pd.to_numeric(d["week"], errors="raise").astype(int)
    seasons = sorted(d["season"].unique())
    rows: list[dict] = []

    prior_games_grid = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    max_weight_grid = [0.25, 0.40, 0.55, 0.70, 0.85]

    for holdout in seasons:
        train = d.loc[d["season"].ne(holdout)].copy()
        test = d.loc[d["season"].eq(holdout)].copy()
        off_beta, def_beta = _fit_train_mapping(train)

        # Inner selection uses only training seasons.
        candidates = []
        for prior_games, max_weight in product(prior_games_grid, max_weight_grid):
            w = max_weight * train["games"] / (train["games"] + prior_games)
            off = (1-w)*train["prior_offense_strength"] + w*(train["oa_offense_measurement"]*off_beta)
            deff = (1-w)*train["prior_defense_strength"] + w*(train["oa_defense_measurement"]*def_beta)
            mae = 0.5 * (
                np.mean(np.abs(off - train["offense_target_points"]))
                + np.mean(np.abs(deff - train["defense_target_points"]))
            )
            candidates.append((float(mae), prior_games, max_weight))
        inner_mae, prior_games, max_weight = min(candidates)

        w = max_weight * test["games"] / (test["games"] + prior_games)
        off = (1-w)*test["prior_offense_strength"] + w*(test["oa_offense_measurement"]*off_beta)
        deff = (1-w)*test["prior_defense_strength"] + w*(test["oa_defense_measurement"]*def_beta)
        prior_mae = 0.5 * (
            np.mean(np.abs(test["prior_offense_strength"] - test["offense_target_points"]))
            + np.mean(np.abs(test["prior_defense_strength"] - test["defense_target_points"]))
        )
        model_mae = 0.5 * (
            np.mean(np.abs(off - test["offense_target_points"]))
            + np.mean(np.abs(deff - test["defense_target_points"]))
        )
        rows.append({
            "holdout_season": holdout,
            "offense_points_per_epa": off_beta,
            "defense_points_per_epa": def_beta,
            "selected_prior_equivalent_games": prior_games,
            "selected_max_measurement_weight": max_weight,
            "inner_mae": inner_mae,
            "holdout_prior_mae": prior_mae,
            "holdout_v3_mae": model_mae,
            "holdout_improvement": prior_mae - model_mae,
            "n": len(test),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested LOSO calibration for Core V3 offense/defense updating.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/research/v3/v3_od_calibration_results.csv"))
    args = parser.parse_args()
    results = evaluate_grid(pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False))
    weighted_prior = np.average(results["holdout_prior_mae"], weights=results["n"])
    weighted_v3 = np.average(results["holdout_v3_mae"], weights=results["n"])
    print(f"\nWeighted prior MAE: {weighted_prior:.6f}")
    print(f"Weighted V3 O/D MAE: {weighted_v3:.6f}")
    print(f"Improvement: {weighted_prior-weighted_v3:+.6f}")
    print("\nResearch gate: do not promote parameters unless gains are stable across holdout seasons.")


if __name__ == "__main__":
    main()
