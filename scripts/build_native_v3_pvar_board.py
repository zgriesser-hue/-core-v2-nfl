"""Build a separate native V3 PVAR board from a frozen weekly research board.

The input board's persistent_margin excludes HFA. Its older neutral-site
adjustment is deliberately recomputed here, so HFA is applied exactly once.
This script never writes the Core V2 live board or release history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEF_GROUPS = {"EDGE", "IDL", "LB", "DB"}
IDENTITY_REPAIRS_2026 = {
    ("MIA", "Rob Beal Jr."): "00-0038603",  # rated as Robert Beal Jr.
    ("MIN", "Harrison Smith"): "00-0029606",
}


def build_board(board: pd.DataFrame, audit: pd.DataFrame,
                ratings: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    if board.empty or board.game_id.duplicated().any():
        raise ValueError("V3 board must contain unique games")
    if board[["season", "week"]].drop_duplicates().shape[0] != 1:
        raise ValueError("V3 board must contain one season and week")
    if coverage.team.duplicated().any():
        raise ValueError("Duplicate injury-report coverage teams")
    if audit[["source_season", "source_week"]].drop_duplicates().shape[0] != 1 or not (
        audit.source_season.eq(board.season.iloc[0]).all()
        and audit.source_week.eq(board.week.iloc[0]).all()
    ):
        raise ValueError("Injury audit season/week does not match V3 board")
    if not board.market_adjustment.fillna(0).eq(0).all():
        raise ValueError("This shadow builder requires the disabled-market V3 base board")
    teams = set(board.home_team) | set(board.away_team)
    if set(audit.team) - teams:
        raise ValueError("Injury audit contains teams outside the V3 board")
    reported = set(coverage.loc[coverage.report_available.astype(str).str.lower().eq("true"), "team"])
    if teams - reported:
        raise ValueError(f"Incomplete official injury-report coverage: {sorted(teams - reported)}")

    audit = audit.copy()
    audit["gsis_id"] = audit["gsis_id"].astype("string")
    ratings = ratings.copy()
    ratings["gsis_id"] = ratings["gsis_id"].astype("string")
    ratings = ratings.loc[ratings.gsis_id.notna()].copy()
    if ratings.gsis_id.duplicated().any():
        raise ValueError("PVAR ratings need unique GSIS IDs")
    for (team, name), player_id in IDENTITY_REPAIRS_2026.items():
        rows = audit.team.eq(team) & audit.player_name.eq(name)
        if rows.any():
            if rows.sum() != 1 or not audit.loc[rows, "gsis_id"].isna().all():
                raise ValueError(f"Identity repair is ambiguous for {team} {name}")
            if not ((ratings.gsis_id.eq(player_id)) & ratings.team.eq(team)).any():
                raise ValueError(f"Verified PVAR row missing for {team} {name}")
            audit.loc[rows, "gsis_id"] = player_id

    eligible = audit.position_group.isin(DEF_GROUPS) & audit.v2_shadow_eligible.astype(str).str.lower().eq("true")
    players = audit.loc[eligible].merge(
        ratings[["gsis_id", "team", "pvar_production", "pvar_source"]].rename(columns={"team": "rating_team"}),
        on="gsis_id", how="left", validate="many_to_one",
    )
    if players[["gsis_id", "status_weight", "pvar_production"]].isna().any().any():
        missing = players.loc[players[["gsis_id", "status_weight", "pvar_production"]].isna().any(axis=1), ["team", "player_name"]]
        raise ValueError("Unresolved defensive PVAR: " + missing.to_string(index=False))
    if not players.team.eq(players.rating_team).all():
        raise ValueError("PVAR rating team does not match weekly audit team")
    if not players.status_weight.ge(0).all() or not players.pvar_production.ge(0).all():
        raise ValueError("Negative PVAR or status weight")
    players["loss_pvar"] = players.status_weight * players.pvar_production
    by_team = players.groupby("team").loss_pvar.sum()
    fallback = players.assign(fallback=players.pvar_source.eq("NO_HISTORY_POSITION_TOP2_PRIOR")).groupby("team").fallback.sum()

    result = board.copy()
    for side in ("home", "away"):
        result[f"{side}_def_loss_pvar"] = result[f"{side}_team"].map(by_team).fillna(0.0)
        result[f"{side}_no_history_players"] = result[f"{side}_team"].map(fallback).fillna(0).astype(int)
    result["hfa_component"] = result.neutral_site.astype(str).str.lower().eq("false").astype(float) * 2.2
    result["neutral_hfa_adjustment"] = 0.0
    result["personnel_adjustment"] = 2.25 * (result.away_def_loss_pvar - result.home_def_loss_pvar)
    result["pred_v3_lite"] = result.persistent_margin + result.hfa_component + result.market_adjustment + result.personnel_adjustment
    result["fair_home_spread_v3"] = result.pred_v3_lite  # margin convention: positive = home favored
    result["edge_points"] = (result.fair_home_spread_v3 - result.market_home_spread).abs()
    result["model_side"] = result.apply(lambda row: row.home_team if row.fair_home_spread_v3 > row.market_home_spread else row.away_team, axis=1)
    result["personnel_status"] = "native_v3_fixed_225"
    result["pvar_snapshot_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["research_only"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("board", "audit", "ratings", "coverage", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    board = pd.read_csv(args.board)
    audit = pd.read_csv(args.audit, dtype={"gsis_id": "string"})
    ratings = pd.read_csv(args.ratings, dtype={"gsis_id": "string"})
    coverage = pd.read_csv(args.coverage)
    result = build_board(board, audit, ratings, coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} V3 PVAR shadow games to {args.out}")


if __name__ == "__main__":
    main()
