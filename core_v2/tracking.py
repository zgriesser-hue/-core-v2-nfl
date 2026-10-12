from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .config import CORE_CONSENSUS_BOOKS
from .projections import game_key, normalize_team


def schedule_with_kickoffs(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = schedule.loc[
        (pd.to_numeric(schedule["season"], errors="coerce") == int(season))
        & (pd.to_numeric(schedule["week"], errors="coerce") == int(week))
    ].copy()
    if "game_type" in games:
        games = games.loc[games["game_type"].astype(str).eq("REG")]
    games["away_team"] = games["away_team"].map(normalize_team)
    games["home_team"] = games["home_team"].map(normalize_team)
    local = pd.to_datetime(
        games["gameday"].astype(str)
        + " "
        + games["gametime"].fillna("00:00").astype(str),
        errors="coerce",
    )
    games["kickoff_utc"] = local.dt.tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    games["game_key"] = games.apply(
        lambda row: game_key(season, week, row["away_team"], row["home_team"]), axis=1
    )
    if games["kickoff_utc"].isna().any():
        raise ValueError("At least one kickoff could not be parsed")
    return games.reset_index(drop=True)


def due_closing_games(
    games: pd.DataFrame,
    existing_consensus: pd.DataFrame,
    now_utc: datetime | pd.Timestamp | None = None,
    minimum_lead_minutes: float = 5.0,
    maximum_lead_minutes: float = 50.0,
    force: bool = False,
) -> pd.DataFrame:
    now = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    known = (
        set(existing_consensus["game_key"].astype(str))
        if not existing_consensus.empty and "game_key" in existing_consensus
        else set()
    )
    candidates = games.loc[~games["game_key"].isin(known)].copy()
    candidates["minutes_to_kickoff"] = (
        candidates["kickoff_utc"] - now
    ).dt.total_seconds().div(60)
    if force:
        return candidates.loc[candidates["minutes_to_kickoff"].gt(0)].copy()
    return candidates.loc[
        candidates["minutes_to_kickoff"].between(
            minimum_lead_minutes, maximum_lead_minutes, inclusive="both"
        )
    ].copy()


def consensus_from_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    core = quotes.loc[quotes["sportsbook"].isin(CORE_CONSENSUS_BOOKS) & pd.to_numeric(quotes["home_spread"], errors="coerce").notna()].copy()
    for column in ["total_points", "away_moneyline_american", "home_moneyline_american"]:
        if column not in core.columns:
            core[column] = np.nan
    if core.empty:
        return pd.DataFrame()
    return (
        core.groupby(
            ["season", "week", "game_key", "away_team", "home_team"], as_index=False
        )
        .agg(
            market_home_spread=("home_spread", "median"),
            market_total=("total_points", "median"),
            closing_away_moneyline=("away_moneyline_american", "median"),
            closing_home_moneyline=("home_moneyline_american", "median"),
            consensus_books=("sportsbook", "nunique"),
            total_consensus_books=("total_points", "count"),
            moneyline_consensus_books=("away_moneyline_american", "count"),
            snapshot_timestamp_utc=("market_snapshot_timestamp_et", "max"),
        )
    )


def append_closing_snapshot(
    existing_book_lines: pd.DataFrame,
    existing_consensus: pd.DataFrame,
    quotes: pd.DataFrame,
    due_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    due_keys = set(due_games["game_key"])
    captured = quotes.loc[quotes["game_key"].isin(due_keys)].copy()
    captured = captured.loc[~captured["sportsbook"].eq("betmgm")].copy()
    if captured.empty:
        raise RuntimeError("The market response did not contain any due games")
    kickoff_map = due_games.set_index("game_key")["kickoff_utc"].astype(str).to_dict()
    captured["kickoff_utc"] = captured["game_key"].map(kickoff_map)
    captured["market_stage"] = "CLOSE"

    consensus = consensus_from_quotes(captured)
    consensus = consensus.loc[consensus["consensus_books"].ge(2)].copy()
    if consensus.empty:
        raise RuntimeError("No due game had at least two core-book closing quotes")
    valid_keys = set(consensus["game_key"])
    captured = captured.loc[captured["game_key"].isin(valid_keys)].copy()
    consensus["kickoff_utc"] = consensus["game_key"].map(kickoff_map)
    consensus = consensus.rename(columns={
        "market_home_spread": "closing_home_spread",
        "market_total": "closing_total",
    })
    consensus["closing_method"] = "MEDIAN_CORE_BOOKS_PREKICK"

    books_out = pd.concat([existing_book_lines, captured], ignore_index=True)
    closes_out = pd.concat([existing_consensus, consensus], ignore_index=True)
    if books_out.duplicated(["game_key", "sportsbook"]).any():
        raise RuntimeError("Closing book history contains duplicate game/book rows")
    if closes_out.duplicated("game_key").any():
        raise RuntimeError("Closing consensus contains duplicate games")
    return books_out, closes_out


def build_release_records(
    board: pd.DataFrame,
    quotes: pd.DataFrame,
    release_timestamp_utc: str,
) -> pd.DataFrame:
    consensus = consensus_from_quotes(quotes)
    if consensus.empty:
        raise RuntimeError("No core-book consensus was available at model release")
    merged = board.merge(
        consensus[
            ["game_key", "market_home_spread", "consensus_books"]
        ],
        on="game_key",
        how="left",
        validate="one_to_one",
    )
    if merged["market_home_spread"].isna().any():
        missing = merged.loc[merged["market_home_spread"].isna(), "game_key"].tolist()
        raise RuntimeError(f"Release consensus missing games: {missing}")
    fair_column = "fair_home_spread" if "fair_home_spread" in merged else "core_v2_home_spread"
    merged["release_timestamp_utc"] = release_timestamp_utc
    merged["fair_home_spread"] = pd.to_numeric(merged[fair_column], errors="raise")
    merged["model_selected_team"] = np.select(
        [
            merged["fair_home_spread"].lt(merged["market_home_spread"]),
            merged["fair_home_spread"].gt(merged["market_home_spread"]),
        ],
        [merged["home_team"], merged["away_team"]],
        default="PASS",
    )
    merged["model_selected_side"] = np.select(
        [
            merged["model_selected_team"].eq(merged["home_team"]),
            merged["model_selected_team"].eq(merged["away_team"]),
        ],
        ["HOME", "AWAY"],
        default="PASS",
    )
    merged["release_bet_spread"] = np.select(
        [
            merged["model_selected_side"].eq("HOME"),
            merged["model_selected_side"].eq("AWAY"),
        ],
        [merged["market_home_spread"], -merged["market_home_spread"]],
        default=np.nan,
    )
    merged["model_edge_at_release"] = (
        merged["fair_home_spread"] - merged["market_home_spread"]
    ).abs()
    columns = [
        "season", "week", "game_key", "away_team", "home_team",
        "release_timestamp_utc", "fair_home_spread", "market_home_spread",
        "consensus_books", "model_selected_team", "model_selected_side",
        "release_bet_spread", "model_edge_at_release",
    ]
    for optional in ["projection_display_status", "personnel_info_complete"]:
        if optional in merged:
            columns.append(optional)
    return merged[columns].copy()


def append_release_records(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new], ignore_index=True)
    keys = ["game_key", "release_timestamp_utc"]
    if combined.duplicated(keys).any():
        raise RuntimeError("Projection release history contains duplicate game/snapshot rows")
    return combined.sort_values(keys).reset_index(drop=True)


def grade_completed_games(
    releases: pd.DataFrame,
    closings: pd.DataFrame,
    results: pd.DataFrame,
    existing_grades: pd.DataFrame,
) -> pd.DataFrame:
    if releases.empty or results.empty:
        return existing_grades.copy()
    candidates = releases.copy()
    candidates["_release_ts"] = pd.to_datetime(
        candidates["release_timestamp_utc"], utc=True, errors="coerce"
    )
    closing_columns = [
        "game_key",
        "closing_home_spread",
        "consensus_books",
        "snapshot_timestamp_utc",
    ]
    optional_closing_columns = [
        "closing_total", "total_consensus_books",
        "closing_away_moneyline", "closing_home_moneyline", "moneyline_consensus_books",
    ]
    if closings.empty:
        closing_data = pd.DataFrame(columns=closing_columns)
    else:
        missing = set(closing_columns) - set(closings.columns)
        if missing:
            raise ValueError(f"Closing history missing columns: {sorted(missing)}")
        available_closing_columns = closing_columns + [
            col for col in optional_closing_columns if col in closings.columns
        ]
        closing_data = closings[available_closing_columns].copy()
    candidates = candidates.merge(
        closing_data.rename(columns={"consensus_books": "closing_consensus_books"}),
        on="game_key",
        how="left",
        validate="many_to_one",
    )
    candidates["_closing_ts"] = pd.to_datetime(
        candidates["snapshot_timestamp_utc"], utc=True, errors="coerce"
    )
    candidates = candidates.loc[
        candidates["_release_ts"].notna()
        & (
            candidates["_closing_ts"].isna()
            | candidates["_release_ts"].le(candidates["_closing_ts"])
        )
    ].copy()
    latest = candidates.sort_values(["game_key", "_release_ts"]).groupby(
        "game_key", as_index=False
    ).tail(1)
    merged = latest.merge(
        results[["game_key", "away_score", "home_score"]],
        on="game_key",
        how="inner",
        validate="one_to_one",
    )
    known = (
        set(existing_grades["game_key"].astype(str))
        if not existing_grades.empty and "game_key" in existing_grades
        else set()
    )
    merged = merged.loc[~merged["game_key"].isin(known)].copy()
    if merged.empty:
        return existing_grades.copy()

    merged["actual_home_margin"] = merged["home_score"] - merged["away_score"]
    merged["projected_home_margin"] = -merged["fair_home_spread"]
    merged["projection_abs_error"] = (
        merged["actual_home_margin"] - merged["projected_home_margin"]
    ).abs()
    merged["winner_correct"] = np.where(
        merged["actual_home_margin"].eq(0),
        False,
        np.sign(merged["actual_home_margin"]).eq(np.sign(merged["projected_home_margin"])),
    )
    is_home = merged["model_selected_side"].eq("HOME")
    is_away = merged["model_selected_side"].eq("AWAY")
    merged["closing_bet_spread"] = np.select(
        [is_home, is_away],
        [merged["closing_home_spread"], -merged["closing_home_spread"]],
        default=np.nan,
    )
    selected_margin = np.select(
        [is_home, is_away],
        [merged["actual_home_margin"], -merged["actual_home_margin"]],
        default=np.nan,
    )
    merged["ats_margin_at_release"] = selected_margin + merged["release_bet_spread"]
    merged["ats_result_at_release"] = np.select(
        [
            merged["model_selected_side"].eq("PASS"),
            merged["ats_margin_at_release"].gt(0),
            merged["ats_margin_at_release"].lt(0),
        ],
        ["Pass", "Win", "Loss"],
        default="Push",
    )
    merged["bet_clv_points"] = (
        merged["release_bet_spread"] - merged["closing_bet_spread"]
    )
    # Positive means the released Core V2 projection remained beyond the
    # closing market in the direction of the model's selected side. Negative
    # means the closing market crossed past the model projection.
    merged["projection_vs_close_points"] = np.select(
        [is_home, is_away],
        [
            merged["closing_home_spread"] - merged["fair_home_spread"],
            merged["fair_home_spread"] - merged["closing_home_spread"],
        ],
        default=np.nan,
    )
    # Retain the legacy field as traditional bet CLV for downstream users.
    merged["clv_points"] = merged["bet_clv_points"]
    merged["model_edge_at_close"] = (
        merged["fair_home_spread"] - merged["closing_home_spread"]
    ).abs()
    merged["closing_market_abs_error"] = (
        merged["actual_home_margin"] + merged["closing_home_spread"]
    ).abs()
    merged["closing_status"] = np.where(
        merged["_closing_ts"].notna(),
        "CAPTURED",
        "MISSING_NOT_BACKFILLED",
    )
    merged["graded_timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    merged = merged.drop(columns=["_release_ts", "_closing_ts"], errors="ignore")
    combined = pd.concat([existing_grades, merged], ignore_index=True)
    # Backfill the corrected model-versus-close fields for existing grades.
    for column in [
        "fair_home_spread",
        "closing_home_spread",
        "release_bet_spread",
        "actual_home_margin",
    ]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined_home = combined["model_selected_side"].eq("HOME")
    combined_away = combined["model_selected_side"].eq("AWAY")
    combined["closing_bet_spread"] = np.select(
        [combined_home, combined_away],
        [combined["closing_home_spread"], -combined["closing_home_spread"]],
        default=np.nan,
    )
    combined["bet_clv_points"] = (
        combined["release_bet_spread"] - combined["closing_bet_spread"]
    )
    combined["clv_points"] = combined["bet_clv_points"]
    combined["projection_vs_close_points"] = np.select(
        [combined_home, combined_away],
        [
            combined["closing_home_spread"] - combined["fair_home_spread"],
            combined["fair_home_spread"] - combined["closing_home_spread"],
        ],
        default=np.nan,
    )
    combined["model_edge_at_close"] = (
        combined["fair_home_spread"] - combined["closing_home_spread"]
    ).abs()
    combined["closing_market_abs_error"] = (
        combined["actual_home_margin"] + combined["closing_home_spread"]
    ).abs()
    if combined.duplicated("game_key").any():
        raise RuntimeError("Game-grade history contains duplicate games")
    return combined.sort_values(["season", "week", "game_key"]).reset_index(drop=True)


def normalize_results(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {"season", "week", "away_team", "home_team", "away_score", "home_score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Results source missing columns: {sorted(missing)}")
    results = raw.loc[pd.to_numeric(raw["season"], errors="coerce").eq(int(season))].copy()
    results["away_team"] = results["away_team"].map(normalize_team)
    results["home_team"] = results["home_team"].map(normalize_team)
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results = results.loc[results["away_score"].notna() & results["home_score"].notna()].copy()
    results["week"] = pd.to_numeric(results["week"], errors="raise").astype(int)
    results["game_key"] = results.apply(
        lambda row: game_key(season, row["week"], row["away_team"], row["home_team"]), axis=1
    )
    if results["game_key"].duplicated().any():
        raise ValueError("Results source contains duplicate game keys")
    return results
