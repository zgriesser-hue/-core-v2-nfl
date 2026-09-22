from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.efficiency import (
    build_team_game_epa,
    fetch_nflverse_pbp,
    solve_oa_epa,
)
from core_v2.io import atomic_write_csv
from core_v2.projections import build_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2026 OA EPA shadow ratings and projection board")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--locks", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Week N may only use Weeks < N. This is the core no-leakage guard.
    through_week = max(0, int(args.week) - 1)
    ratings = pd.read_csv(args.ratings)
    if "rating_before_performance" not in ratings.columns:
        components = [
            "base_prior",
            "qb_adjustment",
            "manual_personnel_adjustment",
            "manual_injury_adjustment",
        ]
        ratings["rating_before_performance"] = ratings[components].sum(axis=1)

    pbp = fetch_nflverse_pbp(args.season)
    team_games = build_team_game_epa(pbp, args.season, through_week)
    oa = solve_oa_epa(team_games)
    # Exact live analogue of the validated historical architecture:
    # blend at the matchup-difference level, not by inventing individual
    # EPA-derived team ratings. Historical training-only OA-EPA -> Core
    # difference coefficients were stable around 0.42, so freeze that mapping
    # rather than tuning on 2026 outcomes.
    EPA_TO_CORE_DIFF = 0.42
    MATURE_EPA_WEIGHT = 0.35
    PRIOR_GAMES = 4.0

    oa = oa.copy()
    oa["team"] = oa["team"].astype(str)
    oa_lookup = oa.set_index("team")

    schedule = pd.read_csv(args.schedule, low_memory=False)
    schedule = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(args.season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(args.week))
    ].copy()
    if "game_type" in schedule.columns:
        schedule = schedule.loc[schedule["game_type"].astype(str).eq("REG")]
    from core_v2.projections import normalize_team
    schedule["home_team"] = schedule["home_team"].map(normalize_team)
    schedule["away_team"] = schedule["away_team"].map(normalize_team)

    structural = ratings.set_index("team")["rating_before_performance"].astype(float)
    rows = []
    for game in schedule.itertuples(index=False):
        home = game.home_team
        away = game.away_team
        home_oa = float(oa_lookup.loc[home, "oa_net"]) if home in oa_lookup.index else 0.0
        away_oa = float(oa_lookup.loc[away, "oa_net"]) if away in oa_lookup.index else 0.0
        home_games = int(oa_lookup.loc[home, "games"]) if home in oa_lookup.index else 0
        away_games = int(oa_lookup.loc[away, "games"]) if away in oa_lookup.index else 0
        matchup_games = min(home_games, away_games)
        epa_weight = MATURE_EPA_WEIGHT * matchup_games / (matchup_games + PRIOR_GAMES)
        core_diff = float(structural.loc[home] - structural.loc[away])
        oa_epa_diff = home_oa - away_oa
        epa_implied_core_diff = EPA_TO_CORE_DIFF * oa_epa_diff
        blended_diff = (1.0 - epa_weight) * core_diff + epa_weight * epa_implied_core_diff
        rows.append({
            "game_key": f"{args.season}_W{args.week}_{away}_{home}",
            "home_team": home,
            "away_team": away,
            "core_structural_diff": core_diff,
            "home_oa_net": home_oa,
            "away_oa_net": away_oa,
            "oa_epa_matchup_diff": oa_epa_diff,
            "epa_to_core_diff_beta": EPA_TO_CORE_DIFF,
            "epa_implied_core_diff": epa_implied_core_diff,
            "epa_weight": epa_weight,
            "core_weight": 1.0 - epa_weight,
            "blended_team_strength_diff": blended_diff,
            "shadow_team_strength_delta": blended_diff - core_diff,
        })
    matchup = pd.DataFrame(rows)

    # Build the ordinary structural Core board first, then replace only its
    # team-strength component with the validated matchup-level blend. HFA and
    # personnel are therefore applied exactly once.
    projection_ratings = ratings[["team", "rating_before_performance"]].rename(
        columns={"rating_before_performance": "current_team_rating"}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_ratings = args.output_dir / "_shadow_projection_ratings.csv"
    atomic_write_csv(projection_ratings, temp_ratings)
    try:
        board = build_from_files(
            schedule_path=args.schedule,
            ratings_path=temp_ratings,
            personnel_path=args.personnel,
            season=args.season,
            week=args.week,
            locks_path=None,
        )
    finally:
        temp_ratings.unlink(missing_ok=True)

    board = board.merge(matchup, on=["game_key", "home_team", "away_team"], how="left", validate="one_to_one")
    board["team_strength_component_home"] = board["blended_team_strength_diff"]
    board["projected_home_margin_pre_personnel"] = board["team_strength_component_home"] + board["hfa_component"]
    board["fair_spread_pre_personnel"] = -board["projected_home_margin_pre_personnel"]
    correction = pd.to_numeric(board["personnel_correction_current_clean"], errors="coerce")
    board["fair_spread_home"] = -(board["projected_home_margin_pre_personnel"] + correction.fillna(0.0))
    max_corr = pd.to_numeric(board["personnel_correction_max_clean"], errors="coerce")
    min_corr = pd.to_numeric(board["personnel_correction_min_clean"], errors="coerce")
    board["fair_spread_low"] = -(board["projected_home_margin_pre_personnel"] + max_corr.fillna(0.0))
    board["fair_spread_high"] = -(board["projected_home_margin_pre_personnel"] + min_corr.fillna(0.0))
    board["fair_spread_uncertainty_width"] = board["fair_spread_high"] - board["fair_spread_low"]
    for col in ["fair_spread_pre_personnel", "fair_spread_home", "fair_spread_low", "fair_spread_high", "fair_spread_uncertainty_width"]:
        board[col] = board[col].round(3)
    board["shadow_model"] = "OA_EPA_MATCHUP_BLEND_V2"
    board["performance_data_through_week"] = through_week

    ratings_audit = ratings.merge(oa, on="team", how="left", validate="one_to_one")
    ratings_audit["performance_data_through_week"] = through_week
    ratings_audit["shadow_model"] = "OA_EPA_MATCHUP_BLEND_V2"
    atomic_write_csv(ratings_audit, args.output_dir / "oa_epa_shadow_ratings.csv")
    atomic_write_csv(board, args.output_dir / "oa_epa_shadow_board.csv")

    print(f"OA EPA shadow built for {args.season} Week {args.week}")
    print(f"Performance data through Week {through_week}")
    print(board[["away_team", "home_team", "core_structural_diff", "oa_epa_matchup_diff", "epa_weight", "blended_team_strength_diff", "fair_spread_home"]].to_string(index=False))


if __name__ == "__main__":
    main()
