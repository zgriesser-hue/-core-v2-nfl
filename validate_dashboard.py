from __future__ import annotations

import argparse

import pandas as pd

from parlay_live import rebuild_board


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complete Streamlit data release")
    parser.add_argument("--board", required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--openers", required=True)
    args = parser.parse_args()

    board = pd.read_csv(args.board)
    lines = pd.read_csv(args.lines)
    openers = pd.read_csv(args.openers)
    rebuilt = rebuild_board(board, lines)
    opener_keys = openers[["away_team", "home_team"]].drop_duplicates()
    checked = rebuilt.merge(
        opener_keys.assign(has_opener=True),
        on=["away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )
    if not checked["has_opener"].fillna(False).all():
        missing = checked.loc[checked["has_opener"].isna(), "matchup"].tolist()
        raise RuntimeError("Missing dashboard openers for: " + ", ".join(missing))
    print(f"Dashboard release validated: {len(checked)} games")


if __name__ == "__main__":
    main()
