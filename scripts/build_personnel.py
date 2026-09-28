from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core_v2.io import atomic_write_csv
from core_v2.personnel import (
    build_personnel_adjustments,
    fetch_nfl_injury_report,
    load_role_priors,
    load_weekly_team_adjustments,
)
from core_v2.personnel_v2 import build_personnel_v2_shadow, load_v2_role_repairs
from core_v2.projections import load_week_schedule
from v3_model.personnel import attach_player_pvar_to_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen Core V2 weekly personnel inputs")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--weekly-adjustments", type=Path)
    parser.add_argument("--v2-role-repairs", type=Path)
    parser.add_argument("--v3-pvar-ratings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    schedule = load_week_schedule(args.schedule, args.season, args.week)
    valid_teams = set(schedule["away_team"]) | set(schedule["home_team"])
    try:
        injuries, report_teams = fetch_nfl_injury_report(args.season, args.week, valid_teams)
    except ValueError as exc:
        # Early in a new NFL week, nfl.com may still serve the prior week's
        # report at the new-week URL. The fetcher deliberately rejects that
        # stale data. Publishing should continue with zero official-report
        # coverage so manual weekly adjustments remain available and stale
        # OUT/DOUBTFUL designations never leak into the new projection board.
        if "refusing to reuse potentially stale prior-week designations" not in str(exc):
            raise
        print(f"EARLY_PROVISIONAL: {exc}")
        injuries = pd.DataFrame(
            columns=[
                "team", "player_name", "position_reported", "injury",
                "practice_status", "game_status", "source_table",
                "source_url", "source_season", "source_week",
            ]
        )
        report_teams = set()
    roles = load_role_priors(args.roles)
    weekly = load_weekly_team_adjustments(args.weekly_adjustments, args.season, args.week)
    v2_repairs = load_v2_role_repairs(args.v2_role_repairs, args.season, args.week)
    personnel, audit = build_personnel_adjustments(
        schedule, roles, injuries, report_teams, weekly, args.season, args.week
    )
    shadow, audit = build_personnel_v2_shadow(
        schedule, audit, args.season, args.week, v2_repairs
    )
    if args.v3_pvar_ratings is not None:
        ratings = pd.read_csv(args.v3_pvar_ratings, dtype={"gsis_id": "string"})
        audit = attach_player_pvar_to_audit(audit, ratings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    injuries["captured_at_utc"] = timestamp
    coverage = pd.DataFrame(
        {"team": sorted(valid_teams), "report_available": [team in report_teams for team in sorted(valid_teams)]}
    )
    atomic_write_csv(injuries, args.output_dir / "official_injury_report.csv")
    atomic_write_csv(coverage, args.output_dir / "report_coverage.csv")
    atomic_write_csv(audit, args.output_dir / "personnel_player_audit.csv")
    atomic_write_csv(shadow, args.output_dir / "personnel_v2_shadow.csv")
    atomic_write_csv(personnel, args.output_dir / "personnel_adjustments.csv")
    complete = int(personnel["report_coverage_complete_clean"].sum())
    print(f"Built {len(personnel)} games; {complete} have complete report/role coverage")


if __name__ == "__main__":
    main()
