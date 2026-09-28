"""Source-backed early availability overrides for one game week.

Only reviewed rows for the target week can enter the personnel layer.
An uncertain QB is explicitly a starter scenario, never a confirmed OUT.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED = {"season", "week", "team", "player", "position_group", "assessment", "source_url",
            "source_published_at_utc", "observed_at_utc", "reviewed_by",
            "reason", "adjustment_points", "out_adjustment_points"}
ASSESSMENTS = {"PROVISIONAL_QB", "CONFIRMED_OUT", "AVAILABLE"}


def load_early_availability(path: Path | None, season: int, week: int,
                            asof: datetime | None = None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED))
    ledger = pd.read_csv(path)
    missing = REQUIRED - set(ledger.columns)
    if missing:
        raise ValueError(f"Availability ledger missing columns: {sorted(missing)}")
    ledger = ledger.loc[ledger.season.eq(season) & ledger.week.eq(week)].copy()
    if ledger.empty:
        return ledger
    if ledger.duplicated(["team", "player"]).any():
        raise ValueError("Conflicting availability entries for a player and week")
    if not ledger.assessment.isin(ASSESSMENTS).all():
        raise ValueError("Unknown availability assessment")
    if ledger[["team", "player", "position_group", "reason", "reviewed_by"]].isna().any().any():
        raise ValueError("Availability entries require team, player, reason and reviewer")
    if not ledger.source_url.astype(str).str.startswith("https://").all():
        raise ValueError("Availability entries require an HTTPS source URL")
    published = pd.to_datetime(ledger.source_published_at_utc, utc=True, errors="coerce")
    observed = pd.to_datetime(ledger.observed_at_utc, utc=True, errors="coerce")
    cutoff = pd.Timestamp(asof or datetime.now(timezone.utc))
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    if published.isna().any() or observed.isna().any() or (published > observed).any() or (observed > cutoff).any():
        raise ValueError("Availability source timestamps are missing, reversed or future-dated")
    numbers = ledger[["adjustment_points", "out_adjustment_points"]].apply(pd.to_numeric, errors="coerce")
    if numbers.isna().any().any() or numbers.gt(0).any().any():
        raise ValueError("Availability adjustments must be finite, nonpositive points")
    provisional = ledger.assessment.eq("PROVISIONAL_QB")
    if (numbers.loc[provisional, "adjustment_points"] < numbers.loc[provisional, "out_adjustment_points"]).any():
        raise ValueError("Provisional center must lie between out and available scenarios")
    if (numbers.loc[ledger.assessment.eq("AVAILABLE")] != 0).any().any():
        raise ValueError("Available player cannot have an injury adjustment")
    return ledger


def apply_early_availability(weekly: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Merge reviewed entries; reject double pricing against manual rows."""
    if ledger.empty:
        return weekly
    weekly = weekly.copy()
    keys = set(zip(weekly.team, weekly.player))
    collisions = set(zip(ledger.team, ledger.player)) & keys
    if collisions:
        raise ValueError(f"Availability already priced in weekly adjustments: {sorted(collisions)}")
    active = ledger.loc[~ledger.assessment.eq("AVAILABLE")].copy()
    if active.empty:
        return weekly
    if (active.assessment.eq("PROVISIONAL_QB") & ~active.position_group.eq("QB")).any():
        raise ValueError("Provisional QB scenarios require QB position")
    active["category"] = active.apply(
        lambda row: "QB_PROVISIONAL" if row.assessment == "PROVISIONAL_QB"
        else ("QB" if row.position_group == "QB" else "INJURY"), axis=1
    )
    active["exclude_from_auto"] = True
    for col in weekly.columns:
        if col not in active:
            active[col] = pd.NA
    return pd.concat([weekly, active[weekly.columns]], ignore_index=True)
