"""Build an explicitly provisional or fully covered 2026 V3 research board."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_native_v3_pvar_board import build_board as attach_defensive_pvar
from scripts.early_availability import load_early_availability, reviewed_defensive_losses
from scripts.v3_market_snapshot import attach_market_snapshot
from scripts.v3_injury_guard import unresolved_injuries
from v3_model.epa_updater import update_split_state


def defer_unmatched_limited(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Defer low-weight, undesignated practice listings lacking a PVAR identity.

    These players receive no invented PVAR. Their team's game is blocked from
    betting edges until the identity or availability is reviewed.
    """
    reviewed = audit.copy()
    if reviewed.empty:
        return reviewed, reviewed.iloc[0:0]
    required = {"gsis_id", "status_weight", "v2_shadow_eligible",
                "position_group", "team", "player_name"}
    if not required.issubset(reviewed):
        raise ValueError(f"Missing defensive audit fields: {sorted(required - set(reviewed))}")
    # Some historical fixture audits omit practice metadata. Without all
    # three fields, no row qualifies for this narrow exception.
    if not {"game_status", "practice_status", "match_method"}.issubset(reviewed):
        return reviewed, reviewed.iloc[0:0][["team", "player_name"]]
    weight = pd.to_numeric(reviewed.status_weight, errors="coerce")
    limited = (
        reviewed.gsis_id.isna()
        & reviewed.game_status.isna()
        & reviewed.practice_status.eq("Limited Participation in Practice")
        & reviewed.match_method.eq("UNMATCHED")
        & reviewed.position_group.isin({"EDGE", "IDL", "LB", "DB"})
        & reviewed.v2_shadow_eligible.astype(str).str.lower().eq("true")
        & weight.ge(0)
        & weight.le(0.05)
    )
    deferred = reviewed.loc[limited, ["team", "player_name"]].copy()
    reviewed.loc[limited, "v2_shadow_eligible"] = False
    return reviewed, deferred


def build_week(history: pd.DataFrame, priors: pd.DataFrame, schedule: pd.DataFrame,
               coverage: pd.DataFrame, audit: pd.DataFrame, ratings: pd.DataFrame,
               availability: pd.DataFrame, season: int, week: int,
               market_snapshots: pd.DataFrame | None = None,
               asof: pd.Timestamp | None = None,
               injury_queue: pd.DataFrame | None = None,
               continuing_absences: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    board = attach_market_snapshot(
        board, market_snapshots if market_snapshots is not None else pd.DataFrame(),
        asof if asof is not None else pd.Timestamp.now(tz="UTC"))
    board["qb_adjustment"] = 0.0
    board["qb_availability_notes"] = ""
    board["def_availability_notes"] = ""
    early_defense = reviewed_defensive_losses(availability, ratings)
    for row in availability.itertuples(index=False):
        if row.assessment == "AVAILABLE":
            continue
        if row.position_group != "QB":
            continue  # Reviewed defensive players are valued from frozen PVAR below.
        board.loc[board.home_team.eq(row.team), "qb_adjustment"] += float(row.adjustment_points)
        board.loc[board.away_team.eq(row.team), "qb_adjustment"] -= float(row.adjustment_points)
        note = f"{getattr(row, 'player', 'QB')}: {row.assessment} ({float(row.adjustment_points):+.2f} team pts)"
        board.loc[board.home_team.eq(row.team) | board.away_team.eq(row.team), "qb_availability_notes"] = note
    board["personnel_adjustment"] = board.qb_adjustment
    board["neutral_hfa_adjustment"] = 0.0
    board["pred_v3_lite"] = board.persistent_margin + board.hfa_component + board.personnel_adjustment
    board["fair_home_spread_v3"] = board.pred_v3_lite
    board["market_signal_status"] = "disabled_missing_preserved_v3_pregame_snapshot"
    board["state_version"] = f"V3_2026_EPA_THROUGH_W{week-1}_MARKET_OFF"
    board["research_only"] = True
    full_coverage = (set(coverage.loc[coverage.report_available.astype(str).str.lower().eq("true"), "team"]) == teams)
    deferred = audit.iloc[0:0]
    audit_for_pvar = audit
    if full_coverage:
        if audit.empty:
            raise ValueError("Official reports covered all teams but player audit is empty")
        audit_for_pvar, deferred = defer_unmatched_limited(audit)
        pvar = attach_defensive_pvar(board, audit_for_pvar, ratings, coverage)
        pvar["personnel_adjustment"] += pvar.qb_adjustment
        pvar["pred_v3_lite"] += pvar.qb_adjustment
        pvar["fair_home_spread_v3"] = pvar.pred_v3_lite
        board = pvar
        board["personnel_status"] = (
            "native_v3_fixed_225_unmatched_limited_review"
            if not deferred.empty else "native_v3_fixed_225_plus_reviewed_qb"
        )
    else:
        board["personnel_status"] = "EARLY_PROVISIONAL_DEF_PVAR_UNAVAILABLE"
        board["home_def_loss_pvar"] = pd.NA
        board["away_def_loss_pvar"] = pd.NA
    if not early_defense.empty:
        # With full reports, add only the portion not already counted in the
        # official player audit. Before full coverage, these are known-only
        # losses, never a claim that all other defenders are healthy.
        known = early_defense.copy()
        if full_coverage:
            official = audit_for_pvar.copy()
            official["gsis_id"] = official.gsis_id.astype("string")
            official["status_weight"] = pd.to_numeric(official.status_weight, errors="coerce")
            eligible = official.v2_shadow_eligible.astype(str).str.lower().eq("true")
            official = official.loc[eligible & official.status_weight.notna()]
            accounted = official.groupby(["team", "gsis_id"]).status_weight.sum()
            known["already_counted"] = known.apply(
                lambda row: min(float(row.loss_pvar),
                                float(row.loss_pvar) * float(accounted.get(
                                    (row.team, str(row.gsis_id)), 0.0))), axis=1
            )
        else:
            known["already_counted"] = 0.0
        known["incremental_loss"] = known.loss_pvar - known.already_counted
        by_team = known.groupby("team").incremental_loss.sum()
        for side in ("home", "away"):
            column = f"{side}_def_loss_pvar"
            board[column] = pd.to_numeric(board[column], errors="coerce").fillna(0.0)
            board[column] += board[f"{side}_team"].map(by_team).fillna(0.0)
        points = 2.25 * (
            board.away_team.map(by_team).fillna(0.0)
            - board.home_team.map(by_team).fillna(0.0)
        )
        board["personnel_adjustment"] += points
        board["pred_v3_lite"] += points
        board["fair_home_spread_v3"] = board.pred_v3_lite
        for row in known.itertuples(index=False):
            mask = board.home_team.eq(row.team) | board.away_team.eq(row.team)
            note = f"{row.player}: CONFIRMED_OUT ({row.points:.2f} pts full PVAR)"
            board.loc[mask, "def_availability_notes"] = (
                board.loc[mask, "def_availability_notes"].map(
                    lambda existing: f"{existing}; {note}" if existing else note
                )
            )
        board["personnel_status"] = (
            "native_v3_fixed_225_unmatched_limited_review"
            if full_coverage and not deferred.empty
            else "native_v3_fixed_225_plus_reviewed_availability" if full_coverage
            else "EARLY_PROVISIONAL_KNOWN_DEF_PVAR_ONLY"
        )
    board["edge_points"] = (board.fair_home_spread_v3 - board.market_home_spread).abs()
    board["model_side"] = board.apply(
        lambda r: (r.home_team if r.fair_home_spread_v3 > r.market_home_spread else r.away_team)
        if pd.notna(r.market_home_spread) else pd.NA, axis=1)
    unresolved = unresolved_injuries(
        injury_queue if injury_queue is not None else pd.DataFrame(),
        continuing_absences if continuing_absences is not None else pd.DataFrame(),
        availability, season, week)
    board["unpriced_injury_notes"] = ""
    for row in unresolved.itertuples(index=False):
        mask = board.home_team.eq(row.team) | board.away_team.eq(row.team)
        note = f"{row.player}: {row.reason}"
        board.loc[mask, "unpriced_injury_notes"] = board.loc[mask, "unpriced_injury_notes"].map(
            lambda current: f"{current}; {note}" if current else note)
    for row in deferred.itertuples(index=False):
        mask = board.home_team.eq(row.team) | board.away_team.eq(row.team)
        note = f"{row.player}: unmatched limited-practice PVAR needs identity review"
        board.loc[mask, "unpriced_injury_notes"] = board.loc[mask, "unpriced_injury_notes"].map(
            lambda current: f"{current}; {note}" if current else note)
    blocked = board.unpriced_injury_notes.ne("")
    board.loc[blocked, "edge_points"] = float("nan")
    board.loc[blocked, "model_side"] = pd.NA
    board["edge_status"] = "READY"
    board.loc[board.market_home_spread.isna(), "edge_status"] = "MARKET_QUOTE_MISSING_OR_STALE"
    board.loc[blocked, "edge_status"] = "INJURY_REVIEW_REQUIRED"
    board["data_through_week"] = int(history.week.max()) if not history.empty else 0
    board["games_in_epa_history"] = history.game_id.nunique()
    return board.drop(columns=["location", "spread_line"]), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "priors", "schedule", "coverage", "audit", "ratings", "availability", "out", "state-out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--market-snapshots", type=Path, required=True)
    parser.add_argument("--injury-queue", type=Path, required=True)
    parser.add_argument("--continuing-absences", type=Path, required=True)
    args = parser.parse_args()
    availability = load_early_availability(args.availability, args.season, args.week)
    board, state = build_week(
        pd.read_csv(args.history), pd.read_csv(args.priors), pd.read_csv(args.schedule),
        pd.read_csv(args.coverage), pd.read_csv(args.audit, dtype={"gsis_id": "string"}),
        pd.read_csv(args.ratings, dtype={"gsis_id": "string"}), availability, args.season, args.week,
        pd.read_csv(args.market_snapshots) if args.market_snapshots.exists() else pd.DataFrame(),
        injury_queue=pd.read_csv(args.injury_queue) if args.injury_queue.exists() else pd.DataFrame(),
        continuing_absences=pd.read_csv(args.continuing_absences) if args.continuing_absences.exists() else pd.DataFrame())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False)
    state.to_csv(args.state_out, index=False)
    print(f"V3 board: {len(board)} games; status={board.personnel_status.iloc[0]}; EPA games={board.games_in_epa_history.iloc[0]}")


if __name__ == "__main__":
    main()
