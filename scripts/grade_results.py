from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from core_v2.io import atomic_write_csv
from core_v2.tracking import grade_completed_games, normalize_results


DEFAULT_RESULTS_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch final scores and append Core V2 grades")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--closings", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--dashboard-output", type=Path, required=True)
    parser.add_argument("--results-file", type=Path)
    parser.add_argument("--results-url", default=DEFAULT_RESULTS_URL)
    args = parser.parse_args()

    if not args.releases.exists() or not args.closings.exists():
        print("No release/closing pair is available yet; nothing to grade")
        return

    if args.results_file:
        raw = pd.read_csv(args.results_file, low_memory=False)
    else:
        response = requests.get(args.results_url, timeout=60)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), low_memory=False)
    results = normalize_results(raw, args.season)
    releases = pd.read_csv(args.releases) if args.releases.exists() else pd.DataFrame()
    closings = pd.read_csv(args.closings) if args.closings.exists() else pd.DataFrame()
    if releases.empty or closings.empty:
        print("Release or closing history is empty; nothing to grade")
        return
    existing = pd.read_csv(args.history) if args.history.exists() else pd.DataFrame()
    grades = grade_completed_games(releases, closings, results, existing)
    atomic_write_csv(grades, args.history)
    atomic_write_csv(grades, args.dashboard_output)
    print(f"Final results available: {len(results)}")
    print(f"Permanent game grades: {len(grades)}")


if __name__ == "__main__":
    main()
