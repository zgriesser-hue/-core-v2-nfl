from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.ratings import (
    update_team_ratings,
    update_team_ratings_opponent_adjusted,
)


ROOT = Path(__file__).resolve().parents[1]


def project_game(home: str, away: str, ratings: pd.DataFrame, hfa: float = 2.2) -> float:
    lookup = ratings.set_index("team")["current_team_rating"]
    return float(lookup[home] - lookup[away] + hfa)


def walk_forward(
    base_ratings: pd.DataFrame,
    grades: pd.DataFrame,
    shrinkage: float,
    learning_rate: float,
    cap: float,
) -> pd.DataFrame:
    rows = []
    season = int(pd.to_numeric(grades["season"]).max())
    weeks = sorted(pd.to_numeric(grades["week"]).dropna().astype(int).unique())
    for week in weeks:
        prior = grades.loc[pd.to_numeric(grades["week"]) < week].copy()
        current = grades.loc[pd.to_numeric(grades["week"]) == week].copy()
        if prior.empty:
            current_ratings = base_ratings.copy()
            current_ratings["current_team_rating"] = (
                current_ratings["base_prior"]
                + current_ratings["qb_adjustment"]
                + current_ratings["manual_personnel_adjustment"]
                + current_ratings["manual_injury_adjustment"]
            )
        elif shrinkage == 0.0:
            current_ratings = update_team_ratings(
                base_ratings, prior, season, week - 1,
                learning_rate=learning_rate, team_game_cap=cap,
            )
        else:
            current_ratings = update_team_ratings_opponent_adjusted(
                base_ratings, prior, season, week - 1,
                learning_rate=learning_rate, team_game_cap=cap,
                opponent_shrinkage=shrinkage,
            )
        for game in current.itertuples(index=False):
            prediction = project_game(game.home_team, game.away_team, current_ratings)
            actual = float(game.actual_home_margin)
            rows.append({
                "season": season,
                "week": week,
                "game_key": game.game_key,
                "actual_home_margin": actual,
                "predicted_home_margin": prediction,
                "abs_error": abs(actual - prediction),
                "opponent_shrinkage": shrinkage,
                "learning_rate": learning_rate,
                "team_game_cap": cap,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/model/team_ratings_2026.csv")
    parser.add_argument("--grades", default="game_grades.csv")
    parser.add_argument("--output", default="data/experiments/opponent_adjustment_backtest.csv")
    args = parser.parse_args()

    ratings = pd.read_csv(ROOT / args.ratings)
    grades = pd.read_csv(ROOT / args.grades)
    grades = grades.dropna(subset=["actual_home_margin", "projected_home_margin"]).copy()

    # Remove already-learned performance so every candidate starts from the
    # same structural prior and cannot leak future games into earlier weeks.
    for column in [
        "base_prior", "qb_adjustment",
        "manual_personnel_adjustment", "manual_injury_adjustment",
    ]:
        ratings[column] = pd.to_numeric(ratings[column], errors="raise")

    candidates = []
    for learning_rate in [0.05, 0.075, 0.10, 0.125, 0.15]:
        for cap in [0.5, 0.75, 1.0, 1.25]:
            for shrinkage in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = walk_forward(ratings, grades, shrinkage, learning_rate, cap)
                # Week 1 has no in-season information, so tuning is evaluated
                # only where the rating update could have changed a forecast.
                scored = result.loc[result["week"] > result["week"].min()]
                candidates.append({
                    "learning_rate": learning_rate,
                    "team_game_cap": cap,
                    "opponent_shrinkage": shrinkage,
                    "games": len(scored),
                    "mae": scored["abs_error"].mean(),
                    "median_abs_error": scored["abs_error"].median(),
                })

    summary = pd.DataFrame(candidates).sort_values(
        ["mae", "median_abs_error", "opponent_shrinkage"]
    ).reset_index(drop=True)
    baseline_mae = summary.loc[summary["opponent_shrinkage"].eq(0), "mae"].min()
    summary["mae_vs_best_current_form"] = summary["mae"] - baseline_mae
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
