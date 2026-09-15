"""Walk-forward backtest of independent offense and defense team ratings.

This is a research-only shadow model. It downloads nflverse weekly team stats,
constructs pregame offense and defense EPA-per-play ratings, tunes hyperparameters
on 2020-2022, and reports a locked 2023-2025 test. Production files are never
read or written.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.parquet"
)
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def normalize_team(team: object) -> object:
    """Align historical schedule abbreviations with nflverse stat abbreviations."""
    return {"OAK": "LV"}.get(team, team)


def download_inputs(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for season in range(2018, 2026):
        path = cache_dir / f"stats_team_week_{season}.parquet"
        if not path.exists():
            response = requests.get(STATS_URL.format(season=season), timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
        frames.append(pd.read_parquet(path))

    games_path = cache_dir / "games.csv"
    if not games_path.exists():
        response = requests.get(GAMES_URL, timeout=90)
        response.raise_for_status()
        games_path.write_bytes(response.content)
    return pd.concat(frames, ignore_index=True), pd.read_csv(games_path, low_memory=False)


def prepare_games(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    stats = stats.loc[stats["season_type"].eq("REG")].copy()
    stats["offensive_plays"] = (
        pd.to_numeric(stats["attempts"], errors="coerce").fillna(0)
        + pd.to_numeric(stats["sacks_suffered"], errors="coerce").fillna(0)
        + pd.to_numeric(stats["carries"], errors="coerce").fillna(0)
    )
    stats["off_epa_per_play"] = (
        pd.to_numeric(stats["passing_epa"], errors="coerce").fillna(0)
        + pd.to_numeric(stats["rushing_epa"], errors="coerce").fillna(0)
    ) / stats["offensive_plays"].replace(0, np.nan)

    team_stats = stats[["game_id", "team", "off_epa_per_play"]].drop_duplicates(
        ["game_id", "team"], keep="last"
    )
    schedule = games.loc[
        games["season"].between(2018, 2025)
        & games["game_type"].eq("REG")
        & games["home_score"].notna()
        & games["away_score"].notna(),
        [
            "game_id", "season", "week", "gameday", "away_team", "home_team",
            "away_score", "home_score", "spread_line",
        ],
    ].copy()
    schedule["actual_home_margin"] = schedule["home_score"] - schedule["away_score"]
    schedule["home_team"] = schedule["home_team"].map(normalize_team)
    schedule["away_team"] = schedule["away_team"].map(normalize_team)

    home = team_stats.rename(
        columns={"team": "home_team_stats", "off_epa_per_play": "home_game_off_epa"}
    )
    away = team_stats.rename(
        columns={"team": "away_team_stats", "off_epa_per_play": "away_game_off_epa"}
    )
    result = schedule.merge(home, on="game_id", how="left").merge(
        away, on="game_id", how="left"
    )
    result = result.loc[
        result["home_team_stats"].eq(result["home_team"])
        & result["away_team_stats"].eq(result["away_team"])
    ].copy()
    if result["game_id"].duplicated().any():
        raise RuntimeError("Team-stat merge produced duplicate games")
    return result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def add_unit_features(
    games: pd.DataFrame,
    alpha: float,
    carryover: float,
) -> pd.DataFrame:
    offense: dict[str, float] = {}
    defense: dict[str, float] = {}
    previous_season: int | None = None
    rows: list[dict[str, float | int | str]] = []

    for (season, week), week_games in games.groupby(["season", "week"], sort=True):
        season = int(season)
        if previous_season is not None and season != previous_season:
            offense = {team: value * carryover for team, value in offense.items()}
            defense = {team: value * carryover for team, value in defense.items()}
        previous_season = season

        updates: list[tuple[str, float, float]] = []
        for game in week_games.itertuples(index=False):
            home_off = offense.get(game.home_team, 0.0)
            away_off = offense.get(game.away_team, 0.0)
            home_def = defense.get(game.home_team, 0.0)
            away_def = defense.get(game.away_team, 0.0)
            rows.append({
                "game_id": game.game_id,
                "season": season,
                "week": int(week),
                "offense_difference": home_off - away_off,
                "defense_difference": home_def - away_def,
            })
            # Positive defense means suppressing opponent EPA.
            updates.extend([
                (game.home_team, game.home_game_off_epa, -game.away_game_off_epa),
                (game.away_team, game.away_game_off_epa, -game.home_game_off_epa),
            ])

        # Apply after all games in the week so no same-week result leaks.
        for team, off_value, def_value in updates:
            if pd.notna(off_value):
                offense[team] = (1 - alpha) * offense.get(team, 0.0) + alpha * off_value
            if pd.notna(def_value):
                defense[team] = (1 - alpha) * defense.get(team, 0.0) + alpha * def_value

        # League centering is not game-pair zero-summing; it removes seasonal EPA drift.
        if offense:
            off_mean = float(np.mean(list(offense.values())))
            offense = {team: value - off_mean for team, value in offense.items()}
        if defense:
            def_mean = float(np.mean(list(defense.values())))
            defense = {team: value - def_mean for team, value in defense.items()}

    return games.merge(pd.DataFrame(rows), on=["game_id", "season", "week"], validate="one_to_one")


def fit_predict(
    featured: pd.DataFrame,
    fit_seasons: range,
    score_seasons: range,
    ridge_alpha: float,
    split_units: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    fit = featured["season"].isin(fit_seasons)
    score = featured["season"].isin(score_seasons)
    if split_units:
        columns = ["offense_difference", "defense_difference"]
    else:
        featured = featured.copy()
        featured["combined_difference"] = (
            featured["offense_difference"] + featured["defense_difference"]
        )
        columns = ["combined_difference"]

    fit_x = featured.loc[fit, columns].to_numpy(dtype=float)
    fit_x = np.column_stack([np.ones(len(fit_x)), fit_x])
    fit_y = featured.loc[fit, "actual_home_margin"].to_numpy(dtype=float)
    if ridge_alpha == 0:
        coefficients = np.linalg.lstsq(fit_x, fit_y, rcond=None)[0]
    else:
        penalty = np.eye(fit_x.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0  # Do not penalize home-field intercept.
        coefficients = np.linalg.solve(fit_x.T @ fit_x + penalty, fit_x.T @ fit_y)
    scored = featured.loc[score].copy()
    score_x = np.column_stack([
        np.ones(len(scored)), scored[columns].to_numpy(dtype=float)
    ])
    scored["prediction"] = score_x @ coefficients
    return scored, coefficients


def metrics(scored: pd.DataFrame, prediction: str) -> tuple[float, float]:
    error = scored["actual_home_margin"] - scored[prediction]
    return float(error.abs().mean()), float(np.sqrt(np.mean(error**2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/nflverse"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stats, schedule = download_inputs(args.cache_dir)
    games = prepare_games(stats, schedule)
    print(f"Complete regular-season games: {len(games):,}")

    candidates: list[dict[str, float]] = []
    for alpha in (0.15, 0.25, 0.35, 0.50, 0.70):
        for carryover in (0.25, 0.45, 0.65, 0.85):
            featured = add_unit_features(games, alpha, carryover)
            for ridge_alpha in (0.0, 1.0, 10.0, 100.0):
                for split_units in (False, True):
                    scored, coefficients = fit_predict(
                        featured, range(2018, 2020), range(2020, 2023),
                        ridge_alpha, split_units,
                    )
                    mae, rmse = metrics(scored, "prediction")
                    candidates.append({
                        "alpha": alpha,
                        "carryover": carryover,
                        "ridge_alpha": ridge_alpha,
                        "split_units": split_units,
                        "validation_mae": mae,
                        "validation_rmse": rmse,
                        "intercept": coefficients[0],
                        "offense_coef": coefficients[1],
                        "defense_coef": coefficients[2] if split_units else coefficients[1],
                    })

    tuning = pd.DataFrame(candidates)
    best_rows = []
    test_games = []
    for split_units, label in ((False, "COMBINED_UNIT"), (True, "OFFENSE_DEFENSE_SPLIT")):
        best = tuning.loc[tuning["split_units"].eq(split_units)].sort_values(
            ["validation_mae", "validation_rmse"]
        ).iloc[0]
        featured = add_unit_features(games, best.alpha, best.carryover)
        scored, coefficients = fit_predict(
            featured, range(2018, 2023), range(2023, 2026),
            best.ridge_alpha, split_units,
        )
        scored["model"] = label
        test_games.append(scored)
        mae, rmse = metrics(scored, "prediction")
        best_rows.append({
            "model": label,
            "alpha": best.alpha,
            "carryover": best.carryover,
            "ridge_alpha": best.ridge_alpha,
            "validation_mae": best.validation_mae,
            "test_mae": mae,
            "test_rmse": rmse,
            "refit_intercept": coefficients[0],
            "refit_offense_coef": coefficients[1],
            "refit_defense_coef": coefficients[2] if split_units else coefficients[1],
        })

    comparison = pd.DataFrame(best_rows)
    market = games.loc[games["season"].between(2023, 2025) & games["spread_line"].notna()].copy()
    market_mae, market_rmse = metrics(market.rename(columns={"spread_line": "prediction"}), "prediction")
    comparison = pd.concat([comparison, pd.DataFrame([{
        "model": "CLOSING_MARKET", "test_mae": market_mae, "test_rmse": market_rmse,
    }])], ignore_index=True)

    print("\nLOCKED TEST RESULTS (2023-2025)")
    print("=" * 110)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    by_season = []
    for scored in test_games:
        for season, group in scored.groupby("season"):
            mae, rmse = metrics(group, "prediction")
            by_season.append({"model": group["model"].iloc[0], "season": season, "games": len(group), "mae": mae, "rmse": rmse})
    print("\nTEST RESULTS BY SEASON")
    print("=" * 110)
    print(pd.DataFrame(by_season).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    combined, split = test_games
    paired = combined[["game_id", "actual_home_margin", "prediction"]].merge(
        split[["game_id", "prediction"]],
        on="game_id",
        suffixes=("_combined", "_split"),
        validate="one_to_one",
    )
    combined_loss = (
        paired["actual_home_margin"] - paired["prediction_combined"]
    ).abs()
    split_loss = (
        paired["actual_home_margin"] - paired["prediction_split"]
    ).abs()
    improvement = (combined_loss - split_loss).to_numpy()
    rng = np.random.default_rng(20260915)
    bootstrap = np.array([
        rng.choice(improvement, size=len(improvement), replace=True).mean()
        for _ in range(20_000)
    ])
    print("\nPAIRED SPLIT IMPROVEMENT")
    print("=" * 110)
    print(f"Mean MAE improvement: {improvement.mean():.5f}")
    print(
        "Bootstrap 95% interval: "
        f"[{np.quantile(bootstrap, 0.025):.5f}, "
        f"{np.quantile(bootstrap, 0.975):.5f}]"
    )
    print(f"Bootstrap probability improvement > 0: {(bootstrap > 0).mean():.3%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
        details_path = args.output.with_name(args.output.stem + "_games.csv")
        pd.concat(test_games, ignore_index=True).to_csv(details_path, index=False)
        print(f"\nSaved: {args.output}")
        print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
