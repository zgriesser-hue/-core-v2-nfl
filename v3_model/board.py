"""
board.py - the offline step that produces a V3 "board" CSV, matching
your existing Core V2 workflow: run the model once (here, or in
Colab), export a CSV, and let the Streamlit app just display + compare
it to live odds (it never runs model code itself - see app.py /
parlay_live.py).

Run this after each week's games complete, pointed at the next week's
slate:

    python -m v3_model.board \\
        --team-game-history path/to/team_game_efficiency_running.csv \\
        --schedule path/to/season_schedule.csv \\
        --market-history path/to/market_history_log.csv \\
        --season 2026 --week 5 \\
        --out week_05_v3_board.csv

Output columns mirror your Core V2 board where the names line up
(matchup, home_team, away_team, market_home_spread, edge_points,
model_side) so the existing app.py display patterns transfer with
minimal changes; fair_home_spread there is Core V2's number, here it
is V3.6's pred_v36 (renamed fair_home_spread_v3 to avoid confusion
if you ever load both boards side by side).

game_key is included in Core V2's exact format
("{season}_W{week}_{away}_{home}"), alongside V3's own game_id (not
replacing it) - this is what lets Stage 4/5 of the automation-wiring
plan reuse core_v2.tracking / core_v2.bet_tracking's grading and
bet-tracking logic UNCHANGED, since both key strictly on that format.

market_home_spread here is filled from your schedule/market-history
input at build time (a snapshot), NOT a live pull - the live
comparison still happens in the Streamlit app via parlay_live.py,
exactly as it does for Core V2 today.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import data_assembly, rating
from .predict_2026 import predict_week


def build_week_board(
    team_game_history_path: str,
    schedule_path: str,
    market_history_path: str | None,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Assemble inputs and run predict_week for one week; return the board."""
    team_game_history = data_assembly.load_team_game_history(
        team_game_history_path, through_season=season, through_week=week
    )
    priors = rating.build_previous_season_priors(team_game_history, season=season)

    market_signal_table = (
        data_assembly.build_market_signal_table(market_history_path)
        if market_history_path
        else None
    )
    matchups = data_assembly.build_matchups(
        schedule_path, season=season, week=week, market_signal_table=market_signal_table
    )

    predictions = predict_week(team_game_history, priors, matchups)

    predictions["matchup"] = predictions["away_team"] + " @ " + predictions["home_team"]
    predictions = predictions.rename(columns={"pred_v36": "fair_home_spread_v3"})

    # game_key, in Core V2's exact format ("{season}_W{week}_{away}_{home}"),
    # not V3's own game_id - this is what lets Stage 4/5 reuse
    # core_v2.tracking / core_v2.bet_tracking unchanged, since both key
    # strictly on this format. Kept alongside game_id, not replacing it -
    # game_id remains V3's own internal join key for everything upstream
    # in this package (data_assembly, predict_2026).
    predictions["game_key"] = predictions.apply(
        lambda r: f"{int(r['season'])}_W{int(r['week'])}_{r['away_team']}_{r['home_team']}", axis=1
    )

    # market_home_spread: latest known line for this game from the market-history
    # log, if supplied (a snapshot at board-build time - the Streamlit app's live
    # refresh, via parlay_live.py, is what keeps this current after that).
    #
    # SIGN CONVENTION, read before touching model_side below: this column
    # (from market_home_margin) and fair_home_spread_v3 are BOTH in margin
    # convention - positive = home favored, matching pred_v30/pred_v36 and
    # nflverse's own spread_line (verified against real moneylines
    # elsewhere in this project). This is the OPPOSITE of standard
    # bookmaker spread notation (negative = home favored), which is what
    # Core V2's own market_home_spread (parsed straight from raw
    # sportsbook odds in core_v2/market.py) uses. The two systems' numbers
    # are NOT directly comparable without a sign flip - see
    # scripts/capture_release_v3.py and pages/1_V3_Dashboard.py for where
    # that flip happens at the V2-integration boundary. Within THIS
    # function, market_home_spread and fair_home_spread_v3 are both
    # margin-convention and consistent with each other.
    if market_history_path:
        mkt = pd.read_csv(market_history_path)
        latest = (
            mkt[(mkt["season"] == season) & (mkt["week"] == week)]
            .sort_values("week")
            .drop_duplicates(subset=["game_id"], keep="last")[["game_id", "market_home_margin"]]
            .rename(columns={"market_home_margin": "market_home_spread"})
        )
        predictions = predictions.merge(latest, on="game_id", how="left")
    else:
        predictions["market_home_spread"] = pd.NA

    predictions["edge_points"] = (
        predictions["fair_home_spread_v3"] - predictions["market_home_spread"]
    ).abs()
    # Margin convention: fair > market means the model projects home as
    # MORE favored than the market prices them - i.e. the market is
    # underpricing home - so the value side is HOME. (This was backwards
    # - fair < market -> HOME - until caught by Stage 4's tests; that
    # direction is only correct under bookmaker convention, which this
    # column is not in. See the sign-convention note above.)
    predictions["model_side"] = predictions.apply(
        lambda r: r["home_team"]
        if pd.notna(r["market_home_spread"]) and r["fair_home_spread_v3"] > r["market_home_spread"]
        else (r["away_team"] if pd.notna(r["market_home_spread"]) else "\u2014"),
        axis=1,
    )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-game-history", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--market-history", default=None)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    board = build_week_board(
        args.team_game_history, args.schedule, args.market_history, args.season, args.week
    )
    board.to_csv(args.out, index=False)
    print(f"Wrote {len(board)} games to {args.out}")


if __name__ == "__main__":
    main()
