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
    non_qb = ~ledger.position_group.eq("QB")
    if (non_qb & ~ledger.assessment.isin({"CONFIRMED_OUT", "AVAILABLE"})).any():
        raise ValueError("Non-QB early availability requires confirmed OUT or AVAILABLE")
    if (non_qb & numbers[["adjustment_points", "out_adjustment_points"]].ne(0).any(axis=1)).any():
        raise ValueError("Non-QB points must be derived from frozen V3 PVAR, not entered manually")
    return ledger


def reviewed_defensive_losses(ledger: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return confirmed defensive losses, failing closed on identity or position mismatch."""
    columns = ["team", "player", "gsis_id", "position_group", "loss_pvar", "points"]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    defense = ledger.loc[ledger.position_group.isin({"EDGE", "IDL", "LB", "DB"})
                         & ledger.assessment.eq("CONFIRMED_OUT")].copy()
    other = ledger.loc[~ledger.position_group.isin({"QB", "EDGE", "IDL", "LB", "DB"})]
    if not other.empty:
        raise ValueError("Early non-QB adjustment requires calibrated defensive PVAR")
    if defense.empty:
        return pd.DataFrame(columns=columns)
    required = {"team", "player_name", "gsis_id", "model_position_v2", "pvar_production"}
    if required - set(ratings):
        raise ValueError("Frozen V3 ratings lack defensive identity/PVAR fields")
    values = ratings.rename(columns={"player_name": "player"})[
        ["team", "player", "gsis_id", "model_position_v2", "pvar_production"]
    ].copy()
    if values.duplicated(["team", "player"]).any():
        raise ValueError("Ambiguous defensive name in frozen PVAR ratings")
    found = defense.merge(values, on=["team", "player"], how="left",
                          validate="one_to_one", indicator=True)
    if not found._merge.eq("both").all():
        raise ValueError("Unmatched reviewed defensive player: "
                         + ", ".join(found.loc[found._merge.ne("both"), "player"]))
    if not found.position_group.eq(found.model_position_v2).all():
        raise ValueError("Reviewed defensive position disagrees with frozen PVAR")
    found["loss_pvar"] = pd.to_numeric(found.pvar_production, errors="coerce")
    if found.gsis_id.isna().any() or found.loss_pvar.isna().any() or found.loss_pvar.lt(0).any():
        raise ValueError("Reviewed defensive player lacks valid GSIS ID or PVAR")
    found["points"] = 2.25 * found.loss_pvar
    return found[columns].copy()


def apply_early_availability(weekly: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Merge reviewed entries; reject double pricing against manual rows."""
    if ledger.empty:
        return weekly
    weekly = weekly.copy()
    keys = set(zip(weekly.team, weekly.player))
    collisions = set(zip(ledger.team, ledger.player)) & keys
    if collisions:
        raise ValueError(f"Availability already priced in weekly adjustments: {sorted(collisions)}")
    # Defensive rows are valued from the frozen V3 player PVAR snapshot.\n    # They must not enter Core V2 as a zero-point manual row or suppress its report join.\n    active = ledger.loc[~ledger.assessment.eq("AVAILABLE") & ledger.position_group.eq("QB")].copy()
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
