"""
grade_results_v3.py - grades V3's projections against the closing
market, mirroring Core V2's own grade_results.py exactly, reusing
core_v2.tracking.grade_completed_games UNCHANGED.

WHERE THIS FILE GOES: same as capture_release_v3.py - copy into the
"-core-v2-nfl-main" repo's `scripts/` folder, since it imports
`core_v2.tracking`/`core_v2.io` directly.

WHAT'S SHARED VS. V3-SPECIFIC:
- Closing consensus (--closings): the SAME data/tracking/
  closing_consensus.csv V2 uses. Stage 1 confirmed the capture
  automation is model-agnostic, so there is no separate "V3 closing
  data" to capture - reuse it as-is.
- Final results (--results-url/--results-file): the SAME nflverse
  games.csv feed grade_results.py already fetches. Scores are scores
  regardless of which model is being graded - reuse unchanged.
- Releases (--releases) and output grades (--history/
  --dashboard-output): these ARE V3-specific, since they're what
  V3's own projections were at release time - point these at
  separate v3_ files so V2 and V3 grading histories never collide.

grade_completed_games itself needs no V3-specific version at all;
it operates on releases/closings/results as plain DataFrames with no
V2-specific math anywhere in it.

Usage (mirrors grade_results.py exactly):

    python -m scripts.grade_results_v3 \\
        --season 2026 \\
        --releases data/tracking/v3_projection_releases.csv \\
        --closings data/tracking/closing_consensus.csv \\
        --history data/tracking/v3_game_grades.csv \\
        --dashboard-output v3_game_grades.csv
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from core_v2.io import atomic_write_csv
from core_v2.tracking import grade_completed_games, normalize_results

DEFAULT_RESULTS_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def grade_v3_results(
    season: int,
    releases_path: Path,
    closings_path: Path,
    history_path: Path,
    results: pd.DataFrame | None = None,
    results_url: str = DEFAULT_RESULTS_URL,
    results_file: Path | None = None,
) -> pd.DataFrame:
    """
    Core logic, separated from argparse/network-fetching so it's
    directly testable (pass `results` to skip the live nflverse pull).
    """
    if not Path(releases_path).exists():
        return pd.DataFrame()

    if results is None:
        if results_file:
            raw = pd.read_csv(results_file, low_memory=False)
        else:
            response = requests.get(results_url, timeout=60)
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text), low_memory=False)
        results = normalize_results(raw, season)

    releases = pd.read_csv(releases_path)
    closings = pd.read_csv(closings_path) if Path(closings_path).exists() else pd.DataFrame()
    if releases.empty:
        return pd.DataFrame()

    existing = pd.read_csv(history_path) if Path(history_path).exists() else pd.DataFrame()
    grades = grade_completed_games(releases, closings, results, existing)
    atomic_write_csv(grades, history_path)
    return grades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--releases", type=Path, default=Path("data/tracking/v3_projection_releases.csv")
    )
    parser.add_argument(
        "--closings", type=Path, default=Path("data/tracking/closing_consensus.csv")
    )
    parser.add_argument(
        "--history", type=Path, default=Path("data/tracking/v3_game_grades.csv")
    )
    parser.add_argument("--dashboard-output", type=Path, default=Path("v3_game_grades.csv"))
    parser.add_argument("--results-file", type=Path)
    parser.add_argument("--results-url", default=DEFAULT_RESULTS_URL)
    args = parser.parse_args()

    if not args.releases.exists():
        print("No V3 projection release history is available yet; nothing to grade")
        return

    grades = grade_v3_results(
        args.season, args.releases, args.closings, args.history,
        results_url=args.results_url, results_file=args.results_file,
    )
    if grades.empty:
        print("V3 projection release history is empty; nothing to grade")
        return
    atomic_write_csv(grades, args.dashboard_output)
    print(f"Permanent V3 game grades: {len(grades)}")


if __name__ == "__main__":
    main()
