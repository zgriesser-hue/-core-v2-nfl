from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core_v2.efficiency import solve_oa_epa


def _require(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def build_team_week_measurements(team_games: pd.DataFrame, shrink_games: float = 4.0) -> pd.DataFrame:
    """Strict pregame OA-EPA measurements for every historical team-week.

    A Week N row uses games from Weeks < N only. Targets are deliberately not
    created here; this file is the leakage-safe feature panel.
    """
    _require(team_games, {"season", "week", "game_id", "team", "opponent", "epa_per_play"}, "team_games")
    x = team_games.copy()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
    rows: list[pd.DataFrame] = []

    for season, sx in x.groupby("season", sort=True):
        teams = sorted(set(sx["team"]) | set(sx["opponent"]))
        for week in sorted(sx["week"].unique()):
            history = sx.loc[sx["week"].lt(week)].copy()
            if history.empty:
                oa = pd.DataFrame({"team": teams, "oa_offense": 0.0, "oa_defense": 0.0, "games": 0})
            else:
                oa = solve_oa_epa(history, shrink_games=shrink_games)
                oa = pd.DataFrame({"team": teams}).merge(oa, on="team", how="left")
                oa[["oa_offense", "oa_defense", "games"]] = oa[["oa_offense", "oa_defense", "games"]].fillna(0.0)
            oa["season"] = int(season)
            oa["week"] = int(week)
            oa["data_through_week"] = int(week) - 1
            rows.append(oa.rename(columns={
                "oa_offense": "oa_offense_measurement",
                "oa_defense": "oa_defense_measurement",
            })[["season", "week", "team", "games", "oa_offense_measurement", "oa_defense_measurement", "data_through_week"]])
    return pd.concat(rows, ignore_index=True)


def attach_future_unit_targets(panel: pd.DataFrame, team_games: pd.DataFrame, horizon_games: int = 4) -> pd.DataFrame:
    """Attach forward offense/defense EPA targets for research evaluation only.

    These columns are labels, never pregame features. Offense target is the
    team's next N offensive game EPA/play. Defense target is sign-flipped EPA
    allowed so positive means better defense, matching V3 convention.
    """
    x = team_games.copy().sort_values(["season", "team", "week", "game_id"])
    out = panel.copy()
    labels = []
    for row in out.itertuples(index=False):
        future_off = x.loc[(x["season"].eq(row.season)) & (x["team"].eq(row.team)) & (x["week"].ge(row.week))].head(horizon_games)
        future_def = x.loc[(x["season"].eq(row.season)) & (x["opponent"].eq(row.team)) & (x["week"].ge(row.week))].head(horizon_games)
        labels.append({
            "season": row.season, "week": row.week, "team": row.team,
            "future_offense_epa": future_off["epa_per_play"].mean() if len(future_off) else np.nan,
            "future_defense_epa": -future_def["epa_per_play"].mean() if len(future_def) else np.nan,
            "future_offense_games": len(future_off), "future_defense_games": len(future_def),
        })
    return out.merge(pd.DataFrame(labels), on=["season", "week", "team"], validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe historical V3 team-week O/D panel.")
    parser.add_argument("--team-games", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/research/v3/v3_historical_team_week_od.parquet"))
    parser.add_argument("--shrink-games", type=float, default=4.0)
    parser.add_argument("--target-horizon-games", type=int, default=4)
    args = parser.parse_args()
    games = pd.read_parquet(args.team_games) if args.team_games.suffix == ".parquet" else pd.read_csv(args.team_games)
    panel = build_team_week_measurements(games, args.shrink_games)
    panel = attach_future_unit_targets(panel, games, args.target_horizon_games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        panel.to_parquet(args.out, index=False)
    else:
        panel.to_csv(args.out, index=False)
    print(f"Rows: {len(panel):,}")
    print(f"Seasons: {sorted(panel['season'].unique())}")
    print(f"Leakage check max(data_through_week - (week-1)): {(panel['data_through_week']-(panel['week']-1)).abs().max()}")
    print(panel.groupby("week")[["games"]].mean().head(8).to_string())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
