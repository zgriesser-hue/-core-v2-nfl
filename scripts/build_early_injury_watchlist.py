"""Create an early-week research watchlist; never infer a current-week absence.

Prior-week designations are leads to verify, not current-week status or PVAR
adjustments. An official current-week row supersedes the prior designation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_watchlist(prior: pd.DataFrame, current: pd.DataFrame,
                    schedule: pd.DataFrame, season: int, week: int,
                    ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    games = schedule.loc[schedule.season.eq(season) & schedule.week.eq(week)].copy()
    if games.empty or games.game_id.duplicated().any():
        raise ValueError("Expected a nonempty, unique target-week schedule")
    if prior.empty or not (prior.source_season.eq(season) & prior.source_week.eq(week - 1)).all():
        raise ValueError("Prior injury audit must be exclusively from the preceding week")
    teams = set(games.home_team) | set(games.away_team)
    candidates = prior.loc[
        prior.team.isin(teams) & prior.game_status.astype(str).str.lower().isin(
            {"out", "doubtful", "questionable"})
    ].copy()
    candidates = candidates.drop_duplicates(["team", "player_name"])
    candidates["prior_game_status"] = candidates.game_status
    candidates["current_game_status"] = pd.NA
    candidates["source_week"] = week - 1
    candidates["verification"] = "PRIOR_WEEK_ONLY_UNVERIFIED"
    if not current.empty:
        if not (current.source_season.eq(season) & current.source_week.eq(week)).all():
            raise ValueError("Current injury report has a different season/week")
        if current.duplicated(["team", "player_name"]).any():
            raise ValueError("Duplicate current-week player report rows")
        current_rows = current[["team", "player_name", "game_status"]].rename(
            columns={"game_status": "official_status"})
        candidates = candidates.merge(current_rows, on=["team", "player_name"], how="left", validate="one_to_one")
        matched = candidates.official_status.notna()
        candidates.loc[matched, "current_game_status"] = candidates.loc[matched, "official_status"]
        candidates.loc[matched, "verification"] = "CURRENT_WEEK_OFFICIAL"
        candidates = candidates.drop(columns="official_status")
        prior_keys = set(zip(candidates.team, candidates.player_name))
        new = current.loc[
            ~pd.Series(list(zip(current.team, current.player_name)), index=current.index).isin(prior_keys)
            & current.team.isin(teams)
        ].copy()
        if not new.empty:
            new["prior_game_status"] = pd.NA
            new["current_game_status"] = new.game_status
            new["verification"] = "CURRENT_WEEK_OFFICIAL"
            new["gsis_id"] = pd.NA
            new["position_group"] = pd.NA
            new["source_week"] = week
            candidates = pd.concat([candidates, new], ignore_index=True)
    if ratings is not None:
        rated = ratings[["gsis_id", "pvar_production"]].dropna(subset=["gsis_id"])
        if rated.gsis_id.duplicated().any():
            raise ValueError("Duplicate player PVAR IDs")
        candidates = candidates.merge(rated, on="gsis_id", how="left", validate="many_to_one")
    else:
        candidates["pvar_production"] = pd.NA
    away = games[["away_team", "home_team", "game_id"]].rename(columns={"away_team": "team"})
    home = games[["home_team", "away_team", "game_id"]].rename(columns={"home_team": "team"})
    candidates = candidates.merge(pd.concat([away, home]), on="team", how="left", validate="many_to_one")
    candidates["season"] = season
    candidates["week"] = week
    candidates["model_adjustment_points"] = pd.NA
    cols = ["season", "week", "game_id", "team", "away_team", "home_team", "player_name",
            "gsis_id", "position_group", "prior_game_status", "current_game_status",
            "verification", "pvar_production", "model_adjustment_points", "source_week"]
    return candidates[cols].sort_values(["game_id", "team", "player_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prior = pd.read_csv(args.prior_audit, dtype={"gsis_id": "string"})
    current = pd.read_csv(args.current_report)
    schedule = pd.read_csv(args.schedule)
    ratings = pd.read_csv(args.ratings, dtype={"gsis_id": "string"}) if args.ratings else None
    result = build_watchlist(prior, current, schedule, args.season, args.week, ratings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} verification leads to {args.out}")


if __name__ == "__main__":
    main()
