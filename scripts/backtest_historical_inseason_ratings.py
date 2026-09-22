from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run_candidate(
    games: pd.DataFrame,
    learning_rate: float,
    cap: float,
    decay: float,
) -> pd.DataFrame:
    """Walk forward within each season with no future-game leakage.

    The historical Core V2 prediction is the baseline. After each completed
    week, each team receives half of the baseline margin residual, scaled by
    learning_rate and capped per game. The next prediction adds
    home_adjustment - away_adjustment. Because the baseline prediction already
    includes both teams/opponent strength, this tests incremental in-season
    learning rather than double-counting opponent strength.
    """
    rows = []
    for season, season_games in games.groupby("season", sort=True):
        adjustments: dict[str, float] = {}
        for week, week_games in season_games.groupby("week", sort=True):
            # Forecast the whole week using information available before it.
            for game in week_games.itertuples(index=False):
                home_adj = adjustments.get(str(game.home_team), 0.0)
                away_adj = adjustments.get(str(game.away_team), 0.0)
                baseline = float(game.baseline_prediction)
                adjusted = baseline + home_adj - away_adj
                actual = float(game.home_margin)
                rows.append({
                    "season": int(season),
                    "week": int(week),
                    "game_id": game.game_id,
                    "baseline_prediction": baseline,
                    "adjusted_prediction": adjusted,
                    "home_margin": actual,
                    "baseline_abs_error": abs(actual - baseline),
                    "adjusted_abs_error": abs(actual - adjusted),
                })

            # Decay old information once per week, then learn from this week's
            # completed games. Updates are zero-sum game by game.
            for team in list(adjustments):
                adjustments[team] *= decay
            for game in week_games.itertuples(index=False):
                residual = float(game.home_margin) - float(game.baseline_prediction)
                move = float(np.clip(
                    learning_rate * residual / 2.0,
                    -cap,
                    cap,
                ))
                home = str(game.home_team)
                away = str(game.away_team)
                adjustments[home] = adjustments.get(home, 0.0) + move
                adjustments[away] = adjustments.get(away, 0.0) - move
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-column", default="pred_team_plus_qb")
    parser.add_argument(
        "--output",
        default="opponent_adjusted_historical_backtest.csv",
    )
    args = parser.parse_args()

    path = Path(args.input)
    games = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "season", "week", "game_id", "home_team", "away_team",
        "home_margin", args.prediction_column,
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Historical input missing columns: {sorted(missing)}")

    games = games.dropna(subset=["home_margin", args.prediction_column]).copy()
    games["baseline_prediction"] = pd.to_numeric(
        games[args.prediction_column], errors="raise"
    )
    games["home_margin"] = pd.to_numeric(games["home_margin"], errors="raise")
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["week"] = pd.to_numeric(games["week"], errors="raise").astype(int)
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    summaries = []
    details = {}
    for lr in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        for cap in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for decay in [0.75, 0.85, 0.90, 0.95, 1.0]:
                result = run_candidate(games, lr, cap, decay)
                # Week 1 cannot contain an in-season adjustment.
                scored = result.loc[result["week"] > 1].copy()
                baseline_mae = scored["baseline_abs_error"].mean()
                adjusted_mae = scored["adjusted_abs_error"].mean()
                key = (lr, cap, decay)
                details[key] = result
                summaries.append({
                    "learning_rate": lr,
                    "team_game_cap": cap,
                    "weekly_decay": decay,
                    "games": len(scored),
                    "baseline_mae": baseline_mae,
                    "adjusted_mae": adjusted_mae,
                    "mae_improvement": baseline_mae - adjusted_mae,
                })

    summary = pd.DataFrame(summaries).sort_values(
        ["adjusted_mae", "mae_improvement"],
        ascending=[True, False],
    ).reset_index(drop=True)
    best = summary.iloc[0]
    best_key = (
        float(best.learning_rate),
        float(best.team_game_cap),
        float(best.weekly_decay),
    )
    best_detail = details[best_key]
    season = (
        best_detail.loc[best_detail["week"] > 1]
        .groupby("season")
        .agg(
            games=("game_id", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            adjusted_mae=("adjusted_abs_error", "mean"),
        )
        .reset_index()
    )
    season["mae_improvement"] = season["baseline_mae"] - season["adjusted_mae"]

    output = Path(args.output)
    summary.to_csv(output, index=False)
    season.to_csv(output.with_name(output.stem + "_best_by_season.csv"), index=False)

    print("=" * 100)
    print("HISTORICAL WALK-FORWARD IN-SEASON RATING BACKTEST")
    print("=" * 100)
    print(f"Input games: {len(games):,}")
    print(f"Seasons: {games['season'].min()}-{games['season'].max()}")
    print("\nTOP 20")
    print(summary.head(20).to_string(index=False))
    print("\nBEST CANDIDATE BY SEASON")
    print(season.to_string(index=False))
    print("\nPositive MAE improvement means the in-season update beat the frozen baseline.")


if __name__ == "__main__":
    main()
