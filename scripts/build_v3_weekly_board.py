"""Build an explicitly provisional or fully covered 2026 V3 research board."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_native_v3_pvar_board import build_board as attach_defensive_pvar
from scripts.early_availability import load_early_availability
from v3_model.epa_updater import update_split_state


def build_week(history: pd.DataFrame, priors: pd.DataFrame, schedule: pd.DataFrame,
               coverage: pd.DataFrame, audit: pd.DataFrame, ratings: pd.DataFrame,
               availability: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.loc[history.season.eq(season) & history.week.lt(week)].copy()
    state = update_split_state(priors[["team", "offense_strength", "defense_strength"]], history, season, week)
    games = schedule.loc[schedule.season.eq(season) & schedule.week.eq(week)].copy()
    if "game_type" in games:
        games = games.loc[games.game_type.eq("REG")]
    if games.empty or games.game_id.duplicated().any():
        raise ValueError("Target week must contain unique regular-season games")
    teams = set(games.away_team) | set(games.home_team)
    if len(teams) != 2 * len(games) or state.team.duplicated().any() or not teams.issubset(set(state.team)):
        raise ValueError("V3 state must cover every target-week team exactly once")
    strength = state.set_index("team")[["offense_strength", "defense_strength"]].sum(axis=1)
    board = games[["game_id", "season", "week", "away_team", "home_team", "spread_line", "location"]].copy()
    board["game_key"] = board.apply(lambda r: f"{season}_W{week}_{r.away_team}_{r.home_team}", axis=1)
    board["matchup"] = board.away_team + " @ " + board.home_team
    board["neutral_site"] = board.location.astype(str).str.lower().eq("neutral")
    board["persistent_margin"] = board.home_team.map(strength) - board.away_team.map(strength)
    board["hfa_component"] = (~board.neutral_site).astype(float) * 2.2
    board["market_adjustment"] = 0.0
    board["market_home_spread"] = pd.to_numeric(board.spread_line, errors="coerce")
    board["qb_adjustment"] = 0.0
    for row in availability.itertuples(index=False):
        if row.assessment == "AVAILABLE":
            continue
        if row.position_group != "QB":
            raise ValueError("Non-QB early availability requires separate reviewed PVAR accounting")
        board.loc[board.home_team.eq(row.team), "qb_adjustment"] += float(row.adjustment_points)
        board.loc[board.away_team.eq(row.team), "qb_adjustment"] -= float(row.adjustment_points)
    board["personnel_adjustment"] = board.qb_adjustment
    board["neutral_hfa_adjustment"] = 0.0
    board["pred_v3_lite"] = board.persistent_margin + board.hfa_component + board.personnel_adjustment
    board["fair_home_spread_v3"] = board.pred_v3_lite
    board["market_signal_status"] = "disabled_missing_preserved_v3_pregame_snapshot"
    board["state_version"] = f"V3_2026_EPA_THROUGH_W{week-1}_MARKET_OFF"
    board["research_only"] = True
    full_coverage = (set(coverage.loc[coverage.report_available.astype(str).str.lower().eq("true"), "team"]) == teams)
    if full_coverage:
        if audit.empty:
            raise ValueError("Official reports covered all teams but player audit is empty")
        pvar = attach_defensive_pvar(board, audit, ratings, coverage)
        pvar["personnel_adjustment"] += pvar.qb_adjustment
        pvar["pred_v3_lite"] += pvar.qb_adjustment
        pvar["fair_home_spread_v3"] = pvar.pred_v3_lite
        board = pvar
        board["personnel_status"] = "native_v3_fixed_225_plus_reviewed_qb"
    else:
        board["personnel_status"] = "EARLY_PROVISIONAL_DEF_PVAR_UNAVAILABLE"
        board["home_def_loss_pvar"] = pd.NA
        board["away_def_loss_pvar"] = pd.NA
    board["edge_points"] = (board.fair_home_spread_v3 - board.market_home_spread).abs()
    board["model_side"] = board.apply(
        lambda r: r.home_team if r.fair_home_spread_v3 > r.market_home_spread else r.away_team, axis=1)
    board["data_through_week"] = int(history.week.max()) if not history.empty else 0
    board["games_in_epa_history"] = history.game_id.nunique()
    return board.drop(columns=["location", "spread_line"]), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "priors", "schedule", "coverage", "audit", "ratings", "availability", "out", "state-out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    availability = load_early_availability(args.availability, args.season, args.week)
    board, state = build_week(
        pd.read_csv(args.history), pd.read_csv(args.priors), pd.read_csv(args.schedule),
        pd.read_csv(args.coverage), pd.read_csv(args.audit, dtype={"gsis_id": "string"}),
        pd.read_csv(args.ratings, dtype={"gsis_id": "string"}), availability, args.season, args.week)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False)
    state.to_csv(args.state_out, index=False)
    print(f"V3 board: {len(board)} games; status={board.personnel_status.iloc[0]}; EPA games={board.games_in_epa_history.iloc[0]}")


if __name__ == "__main__":
    main()
