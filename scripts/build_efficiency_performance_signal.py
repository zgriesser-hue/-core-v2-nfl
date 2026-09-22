from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = (
    "epa_per_play",
    "success_rate",
    "early_down_epa_per_play",
    "pass_epa_per_play",
)


def weighted_team_game_signal(
    team_games: pd.DataFrame,
    metric_weights: dict[str, float],
    min_history_games: int = 1,
) -> pd.DataFrame:
    """Create a leakage-safe opponent-adjusted efficiency signal.

    Input is one row per team-game. Every metric is expressed from the team's
    perspective. For each game, expected performance is the opponent's PRE-GAME
    rolling defensive allowance; actual-minus-expected metrics are standardized
    using only prior observations. Current-game information is never used in
    the expectation or standardization.
    """
    required = {"season", "week", "game_id", "team", "opponent"} | set(metric_weights)
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"Team-game efficiency input missing: {sorted(missing)}")

    games = team_games.copy().sort_values(["season", "week", "game_id", "team"])
    output = []

    for season, season_games in games.groupby("season", sort=True):
        history: list[dict] = []
        for week, week_games in season_games.groupby("week", sort=True):
            hist = pd.DataFrame(history)
            for row in week_games.to_dict("records"):
                parts = []
                used_weight = 0.0
                for metric, weight in metric_weights.items():
                    actual = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
                    if pd.isna(actual) or hist.empty:
                        continue
                    opp_hist = hist.loc[hist["opponent"].eq(row["opponent"]), metric]
                    league_hist = pd.to_numeric(hist[metric], errors="coerce").dropna()
                    if len(opp_hist) < min_history_games or len(league_hist) < 16:
                        continue
                    expected = pd.to_numeric(opp_hist, errors="coerce").mean()
                    sd = league_hist.std(ddof=0)
                    if not np.isfinite(sd) or sd <= 0:
                        continue
                    parts.append(weight * (float(actual) - float(expected)) / float(sd))
                    used_weight += abs(weight)

                signal = sum(parts) / used_weight if used_weight else np.nan
                output.append({
                    "season": int(season),
                    "week": int(week),
                    "game_id": row["game_id"],
                    "team": row["team"],
                    "opponent": row["opponent"],
                    "efficiency_signal": signal,
                    "signal_available": bool(used_weight),
                })

            # Add the whole completed week only after all of its signals were made.
            history.extend(week_games.to_dict("records"))

    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-games", required=True)
    parser.add_argument("--output", default="efficiency_performance_signal.csv")
    args = parser.parse_args()

    path = Path(args.team_games)
    games = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    weights = {
        "epa_per_play": 0.35,
        "success_rate": 0.25,
        "early_down_epa_per_play": 0.20,
        "pass_epa_per_play": 0.20,
    }
    result = weighted_team_game_signal(games, weights)
    result.to_csv(args.output, index=False)
    print(f"Rows: {len(result):,}")
    print(f"Signal coverage: {result['signal_available'].mean():.1%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
