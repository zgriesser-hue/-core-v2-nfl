from __future__ import annotations

import argparse
from pathlib import Path

from core_v2.io import atomic_write_csv
from core_v2.projections import build_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Core V2 weekly projection board")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--locks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    board = build_from_files(
        args.schedule,
        args.ratings,
        args.personnel,
        args.season,
        args.week,
        args.locks,
    )
    atomic_write_csv(board, args.output)
    print(f"Built {len(board)} games: {args.output}")


if __name__ == "__main__":
    main()

