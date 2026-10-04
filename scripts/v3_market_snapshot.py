"""Use only timestamped, recent market observations for V3 comparison edges."""
from __future__ import annotations

import pandas as pd


MAX_MARKET_AGE = pd.Timedelta(hours=4)


def attach_market_snapshot(board: pd.DataFrame, snapshots: pd.DataFrame,
                           asof: pd.Timestamp) -> pd.DataFrame:
    """Convert bookmaker home spread to V3 home-margin convention."""
    result = board.reset_index(drop=True).copy()
    now = pd.Timestamp(asof)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    result["market_home_spread"] = float("nan")
    result["market_snapshot_timestamp_utc"] = ""
    result["market_line_status"] = "NO_TIMESTAMPED_QUOTE"
    if snapshots.empty:
        return result
    needed = {"season", "week", "game_key", "market_home_spread", "market_snapshot_timestamp_et"}
    if not needed.issubset(snapshots):
        raise ValueError(f"Market snapshots missing {sorted(needed - set(snapshots))}")
    if result.game_key.duplicated().any():
        raise ValueError("V3 board contains duplicate games")
    quotes = snapshots.copy()
    quotes["observed"] = pd.to_datetime(quotes.market_snapshot_timestamp_et,
                                        utc=True, errors="coerce", format="mixed")
    quotes["book_home"] = pd.to_numeric(quotes.market_home_spread, errors="coerce")
    quotes = quotes.loc[quotes.observed.notna() & quotes.book_home.notna()
                        & quotes.observed.le(now)].copy()
    if quotes.empty:
        return result
    quotes = quotes.sort_values("observed").drop_duplicates(
        ["season", "week", "game_key"], keep="last")
    keys = ["season", "week", "game_key"]
    selected = result[keys].merge(quotes[keys + ["observed", "book_home"]],
                                  on=keys, how="left", validate="one_to_one")
    age = now - selected.observed
    fresh = selected.observed.notna() & age.le(MAX_MARKET_AGE)
    result.loc[fresh, "market_home_spread"] = -selected.loc[fresh, "book_home"].to_numpy()
    result.loc[fresh, "market_snapshot_timestamp_utc"] = (
        selected.loc[fresh, "observed"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy())
    result.loc[fresh, "market_line_status"] = "CAPTURED_RECENT"
    result.loc[selected.observed.notna() & ~fresh, "market_line_status"] = "STALE_QUOTE"
    return result
