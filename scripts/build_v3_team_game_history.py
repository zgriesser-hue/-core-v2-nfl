"""Fetch and archive the current-season team-game EPA input for V3.

Only games with final scores in the repository schedule are eligible. A
partially completed week is allowed and recorded; future games cannot enter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.efficiency import build_team_game_epa, fetch_nflverse_pbp
from core_v2.io import atomic_write_csv


def assemble_history(team_games: pd.DataFrame, schedule: pd.DataFrame,
                     grades: pd.DataFrame, season: int, target_week: int) -> pd.DataFrame:
    required = {"season", "week", "game_id", "away_team", "home_team"}
    if required - set(schedule):
        raise ValueError("Schedule lacks game identity columns")
    if {"season", "week", "game_key", "away_score", "home_score"} - set(grades):
        raise ValueError("Grades lack final-score columns")
    past = schedule.loc[schedule.season.eq(season) & schedule.week.lt(target_week)].copy()
    if "game_type" in past:
        past = past.loc[past.game_type.eq("REG")]
    past["game_key"] = past.apply(
        lambda r: f"{season}_W{int(r.week)}_{r.away_team}_{r.home_team}", axis=1)
    done = grades.loc[grades.season.eq(season) & grades.week.lt(target_week)
                      & grades.away_score.notna() & grades.home_score.notna(), ["game_key"]]
    if done.game_key.duplicated().any():
        raise ValueError("Duplicate final game grades")
    finals = past.loc[past.game_key.isin(done.game_key)]
    if finals.game_id.duplicated().any():
        raise ValueError("Duplicate completed schedule games")
    x = team_games.loc[team_games.season.eq(season) & team_games.week.lt(target_week)].copy()
    x = x.loc[x.game_id.isin(finals.game_id)].copy()
    if set(finals.game_id) != set(x.game_id):
        raise ValueError("Final games missing from nflverse play-by-play")
    if x.groupby("game_id").size().ne(2).any() or x.duplicated(["game_id", "team"]).any():
        raise ValueError("Completed games need two unique offensive team rows")
    opponent = x[["game_id", "team", "epa_per_play"]].rename(
        columns={"team": "opponent", "epa_per_play": "def_epa_per_play"})
    result = x.merge(opponent, on=["game_id", "opponent"], how="left", validate="one_to_one")
    if result.def_epa_per_play.isna().any():
        raise ValueError("Missing opposite-team defensive EPA")
    expected = finals[["game_id", "away_team", "home_team"]]
    for game in result.groupby("game_id"):
        row = expected.loc[expected.game_id.eq(game[0])].iloc[0]
        if set(game[1].team) != {row.away_team, row.home_team}:
            raise ValueError(f"Play-by-play team mismatch for {game[0]}")
    return result.sort_values(["week", "game_id", "team"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pbp = fetch_nflverse_pbp(args.season)
    games = build_team_game_epa(pbp, args.season, args.week - 1)
    schedule = pd.read_csv(args.schedule, low_memory=False)
    grades = pd.read_csv(args.grades)
    history = assemble_history(games, schedule, grades, args.season, args.week)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(history, args.out)
    print(f"V3 EPA history: {len(history)//2} completed games strictly before Week {args.week}")


if __name__ == "__main__":
    main()
