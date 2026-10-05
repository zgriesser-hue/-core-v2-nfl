"""Append-only V3 projection releases and pregame performance grading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


RELEASE_COLUMNS = ["season", "week", "game_key", "away_team", "home_team",
                   "release_timestamp_utc", "kickoff_utc", "fair_home_margin_v3",
                   "release_market_home_spread", "market_snapshot_timestamp_utc",
                   "model_side", "personnel_status"]


def _utc(values):
    return pd.to_datetime(values, utc=True, errors="coerce")


def append_releases(board: pd.DataFrame, schedule: pd.DataFrame,
                    existing: pd.DataFrame, now: pd.Timestamp,
                    market_snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Freeze first observed pregame line. Later changed lines are revisions."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    required = {"season", "week", "game_key", "away_team", "home_team", "fair_home_spread_v3"}
    if not required.issubset(board) or board.game_key.duplicated().any():
        raise ValueError("V3 board requires unique games and fair margins")
    if not {"season", "week", "away_team", "home_team", "gameday", "gametime"}.issubset(schedule):
        raise ValueError("Schedule lacks kickoff fields")
    games = schedule.copy()
    if "game_type" in games:
        games = games.loc[games.game_type.eq("REG")]
    keys = ["season", "week", "away_team", "home_team"]
    if games.duplicated(keys).any():
        raise ValueError("Ambiguous schedule game")
    candidate = board.merge(games[keys + ["gameday", "gametime"]], on=keys,
                            how="left", validate="one_to_one")
    # nflverse schedule dates/times are Eastern, including daylight saving time.
    local = pd.to_datetime(candidate.gameday.astype(str) + " " + candidate.gametime.astype(str), errors="coerce")
    if local.isna().any():
        raise ValueError("Missing or invalid kickoff")
    kickoff = local.dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    candidate["kickoff_utc"] = kickoff
    candidate = candidate.loc[candidate.kickoff_utc.gt(now)].copy()
    prior = existing.copy() if not existing.empty else pd.DataFrame(columns=RELEASE_COLUMNS)
    if not prior.empty:
        if not set(RELEASE_COLUMNS).issubset(prior):
            raise ValueError("Existing V3 release schema incomplete")
        if _utc(prior.release_timestamp_utc).isna().any() or _utc(prior.kickoff_utc).isna().any():
            raise ValueError("Invalid stored V3 release timestamps")
        if (_utc(prior.release_timestamp_utc) >= _utc(prior.kickoff_utc)).any():
            raise ValueError("Stored V3 release after kickoff")
        if prior.duplicated(["game_key", "release_timestamp_utc"]).any():
            raise ValueError("Duplicate V3 release")
    snapshots = market_snapshots.copy() if market_snapshots is not None else pd.DataFrame()
    if not snapshots.empty:
        if not {"game_key", "market_home_spread", "market_snapshot_timestamp_et"}.issubset(snapshots):
            raise ValueError("Timestamped market snapshot schema incomplete")
        snapshots["observed_utc"] = _utc(snapshots.market_snapshot_timestamp_et)
    additions = []
    for row in candidate.itertuples(index=False):
        fair = pd.to_numeric(getattr(row, "fair_home_spread_v3"), errors="coerce")
        if pd.isna(fair):
            continue
        # The board's schedule spread has no pregame quote provenance. Only a
        # timestamped market observation within six hours can define a bet line.
        observed = snapshots.loc[snapshots.game_key.eq(row.game_key) &
                                 snapshots.observed_utc.le(now) &
                                 snapshots.observed_utc.ge(now - pd.Timedelta(hours=6))].sort_values("observed_utc") if not snapshots.empty else snapshots
        market = pd.to_numeric(observed.market_home_spread.iloc[-1], errors="coerce") if not observed.empty else float("nan")
        market_time = observed.observed_utc.iloc[-1].isoformat() if not observed.empty and pd.notna(market) else ""
        history = prior.loc[prior.game_key.eq(row.game_key)]
        if not history.empty:
            latest = history.sort_values("release_timestamp_utc").iloc[-1]
            old_market = pd.to_numeric(latest.release_market_home_spread, errors="coerce")
            if abs(float(latest.fair_home_margin_v3) - float(fair)) < 1e-9 and (
                    (pd.isna(old_market) and pd.isna(market)) or
                    (pd.notna(old_market) and pd.notna(market) and abs(float(old_market) - float(market)) < 1e-9)):
                continue
        additions.append({
            "season": row.season, "week": row.week, "game_key": row.game_key,
            "away_team": row.away_team, "home_team": row.home_team,
            "release_timestamp_utc": now.isoformat(), "kickoff_utc": row.kickoff_utc.isoformat(),
            "fair_home_margin_v3": float(fair),
            "release_market_home_spread": float(market) if pd.notna(market) else float("nan"),
            "market_snapshot_timestamp_utc": market_time,
            "model_side": (row.home_team if fair > -market else row.away_team) if pd.notna(market) else "",
            "personnel_status": getattr(row, "personnel_status", ""),
        })
    return pd.concat([prior, pd.DataFrame(additions, columns=RELEASE_COLUMNS)], ignore_index=True)[RELEASE_COLUMNS]


def grade_releases(releases: pd.DataFrame, grades: pd.DataFrame,
                   closes: pd.DataFrame, openers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Grade first V3 release only. Fair margins are positive for home."""
    if releases.empty:
        return pd.DataFrame()
    if not set(RELEASE_COLUMNS).issubset(releases):
        raise ValueError("V3 release schema incomplete")
    eligible = releases.copy()
    market_time = _utc(eligible.market_snapshot_timestamp_utc)
    release_time = _utc(eligible.release_timestamp_utc)
    eligible = eligible.loc[pd.to_numeric(eligible.release_market_home_spread, errors="coerce").notna() &
                            market_time.notna() & market_time.le(release_time) &
                            (eligible.model_side.eq(eligible.home_team) | eligible.model_side.eq(eligible.away_team))]
    if eligible.empty:
        return pd.DataFrame()
    first = eligible.sort_values("release_timestamp_utc").drop_duplicates("game_key").copy()
    if (_utc(first.release_timestamp_utc) >= _utc(first.kickoff_utc)).any():
        raise ValueError("V3 release after kickoff")
    if grades.game_key.duplicated().any() or closes.game_key.duplicated().any():
        raise ValueError("Duplicate grades or closes")
    result = first.merge(grades[["game_key", "away_score", "home_score"]], on="game_key", how="left", validate="one_to_one")
    result = result.merge(closes[["game_key", "closing_home_spread", "snapshot_timestamp_utc", "kickoff_utc"]].rename(columns={"kickoff_utc":"closing_kickoff_utc"}), on="game_key", how="left", validate="one_to_one")
    if openers is None or openers.empty:
        openers = pd.DataFrame(columns=["game_key", "consensus_open_home_spread", "earliest_core_open_seen_utc"])
    if openers.game_key.duplicated().any():
        raise ValueError("Duplicate opener games")
    result = result.merge(openers[["game_key", "consensus_open_home_spread", "earliest_core_open_seen_utc"]],
                          on="game_key", how="left", validate="one_to_one")
    for col in ("fair_home_margin_v3", "release_market_home_spread", "closing_home_spread", "away_score", "home_score"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result["consensus_open_home_spread"] = pd.to_numeric(result.consensus_open_home_spread, errors="coerce")
    result["actual_home_margin"] = result.home_score - result.away_score
    result["absolute_error"] = (result.actual_home_margin - result.fair_home_margin_v3).abs()
    # A captured close must be later than release and earlier than kickoff.
    valid_close = (_utc(result.snapshot_timestamp_utc) >= _utc(result.release_timestamp_utc)) & (
        _utc(result.snapshot_timestamp_utc) < _utc(result.kickoff_utc)) & result.closing_home_spread.notna()
    result.loc[~valid_close, "closing_home_spread"] = float("nan")
    # Openers are first observed core-book quotes, not necessarily the true market open.
    # Grade the saved model side hypothetically even when it was selected after capture.
    valid_open = _utc(result.earliest_core_open_seen_utc).notna() & (
        _utc(result.earliest_core_open_seen_utc) < _utc(result.kickoff_utc)) & result.consensus_open_home_spread.notna()
    result.loc[~valid_open, "consensus_open_home_spread"] = float("nan")
    result["opener_timing"] = "Unavailable"
    result.loc[valid_open, "opener_timing"] = "Hypothetical (pick after opener)"
    result.loc[valid_open & (_utc(result.release_timestamp_utc) <= _utc(result.earliest_core_open_seen_utc)),
               "opener_timing"] = "Pick available at capture"
    # ATS is model side at release, graded at that release's market spread.
    home_side = result.model_side.eq(result.home_team)
    away_side = result.model_side.eq(result.away_team)
    side_margin = result.actual_home_margin + result.release_market_home_spread
    result["ats_margin"] = side_margin.where(home_side, -side_margin.where(away_side))
    result.loc[result.actual_home_margin.isna() | result.release_market_home_spread.isna(), "ats_margin"] = float("nan")
    result["ats_result"] = result.ats_margin.map(lambda x: "Win" if x > 0 else "Loss" if x < 0 else "Push" if pd.notna(x) else "Pending")
    for name, line in (("opener", "consensus_open_home_spread"), ("close", "closing_home_spread")):
        margin = result.actual_home_margin + result[line]
        result[f"{name}_ats_margin"] = margin.where(home_side, -margin.where(away_side))
        result.loc[result.actual_home_margin.isna() | result[line].isna(), f"{name}_ats_margin"] = float("nan")
        result[f"{name}_ats_result"] = result[f"{name}_ats_margin"].map(
            lambda x: "Win" if x > 0 else "Loss" if x < 0 else "Push" if pd.notna(x) else "Pending")
    # Projection vs close is the signed V3 edge at close for the release side.
    close_edge = result.fair_home_margin_v3 + result.closing_home_spread
    result["projection_vs_close_points"] = close_edge.where(home_side, -close_edge.where(away_side))
    result.loc[result.closing_home_spread.isna(), "projection_vs_close_points"] = float("nan")
    # Standard market CLV: release bet line relative to the close, same side.
    result["market_clv_points"] = (result.release_market_home_spread - result.closing_home_spread).where(
        home_side, (result.closing_home_spread - result.release_market_home_spread).where(away_side))
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    for key in ("board", "schedule", "releases", "market-snapshots"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    old = pd.read_csv(args.releases) if args.releases.exists() else pd.DataFrame()
    market = pd.read_csv(args.market_snapshots) if args.market_snapshots.exists() else pd.DataFrame()
    updated = append_releases(pd.read_csv(args.board), pd.read_csv(args.schedule), old,
                              pd.Timestamp.now(tz="UTC"), market)
    args.releases.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.releases, index=False)
    print(f"V3 release rows: {len(updated)}; added {len(updated) - len(old)}")


if __name__ == "__main__":
    main()
