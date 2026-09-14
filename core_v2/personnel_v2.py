from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .personnel import normalize_player_name


POSITION_GROUPS = ("OL", "WR_TE", "RB", "EDGE", "IDL", "LB", "DB", "ST", "OTHER")

_POSITION_GROUP_MAP = {
    "C": "OL",
    "G": "OL",
    "OG": "OL",
    "OL": "OL",
    "OT": "OL",
    "T": "OL",
    "WR": "WR_TE",
    "TE": "WR_TE",
    "RB": "RB",
    "FB": "RB",
    "EDGE": "EDGE",
    "DE": "EDGE",
    "IDL": "IDL",
    "DT": "IDL",
    "NT": "IDL",
    "LB": "LB",
    "ILB": "LB",
    "OLB": "LB",
    "CB": "DB",
    "DB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
    "QB": "QB",
}


def normalize_position_group(value: object) -> str:
    """Map roster and injury-report positions into stable modeling groups."""
    position = str(value).upper().strip()
    if not position or position == "NAN":
        return "OTHER"
    return _POSITION_GROUP_MAP.get(position, "OTHER")


def load_v2_role_repairs(
    path: Path | None,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Load reviewed role repairs for a single season/week."""
    columns = [
        "season", "start_week", "end_week", "player_name", "team",
        "position_group_v2", "role_v1", "role_v2", "repair_method",
        "evidence_source", "review_status", "name_key",
    ]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)

    repairs = pd.read_csv(path)
    required = {
        "season", "start_week", "end_week", "player_name", "team",
        "position_group_v2", "role_v2", "repair_method",
        "evidence_source", "review_status",
    }
    missing = required - set(repairs.columns)
    if missing:
        raise ValueError(f"V2 role repairs missing columns: {sorted(missing)}")

    for column in ("season", "start_week", "end_week"):
        repairs[column] = pd.to_numeric(repairs[column], errors="raise").astype(int)
    repairs["role_v2"] = pd.to_numeric(repairs["role_v2"], errors="coerce")
    repairs["team"] = repairs["team"].astype(str).str.upper().str.strip()
    repairs["position_group_v2"] = (
        repairs["position_group_v2"].astype(str).str.upper().str.strip()
    )
    repairs["name_key"] = repairs["player_name"].map(normalize_player_name)
    repairs = repairs.loc[
        repairs["season"].eq(int(season))
        & repairs["start_week"].le(int(week))
        & repairs["end_week"].ge(int(week))
    ].copy()

    approved = repairs["review_status"].eq("APPROVED_FOR_V2_SHADOW")
    if not repairs.loc[approved, "role_v2"].between(0, 1, inclusive="both").all():
        raise ValueError("Approved V2 role repairs must be between 0 and 1")
    if repairs.duplicated(["team", "name_key"]).any():
        raise ValueError("Duplicate active V2 role repairs")
    return repairs


def apply_v2_role_repairs(
    audit: pd.DataFrame,
    repairs: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply reviewed values to shadow columns while preserving all V1 fields."""
    frame = audit.copy()
    frame["expected_role_v1"] = pd.to_numeric(
        frame.get("expected_role"), errors="coerce"
    )
    frame["expected_role_v2"] = frame["expected_role_v1"]
    frame["weighted_role_exposure_v1"] = pd.to_numeric(
        frame.get("weighted_role_exposure"), errors="coerce"
    )
    frame["position_group_v2_override"] = None
    frame["v2_role_repair_applied"] = False
    frame["v2_role_repair_method"] = None
    frame["v2_role_evidence_source"] = None
    frame["v2_role_review_status"] = None

    if not frame.empty and repairs is not None and not repairs.empty:
        if "name_key" not in frame:
            frame["name_key"] = frame["player_name"].map(normalize_player_name)
        lookup = repairs.set_index(["team", "name_key"], drop=False)
        for index, row in frame.iterrows():
            key = (str(row["team"]).upper().strip(), str(row["name_key"]))
            if key not in lookup.index:
                continue
            repair = lookup.loc[key]
            frame.at[index, "position_group_v2_override"] = repair[
                "position_group_v2"
            ]
            frame.at[index, "v2_role_repair_method"] = repair["repair_method"]
            frame.at[index, "v2_role_evidence_source"] = repair["evidence_source"]
            frame.at[index, "v2_role_review_status"] = repair["review_status"]
            approved = (
                repair["review_status"] == "APPROVED_FOR_V2_SHADOW"
                and pd.notna(repair["role_v2"])
            )
            if approved:
                frame.at[index, "expected_role_v2"] = float(repair["role_v2"])
                frame.at[index, "v2_role_repair_applied"] = True

    is_qb = frame.get("is_qb", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    excluded = frame.get(
        "excluded_manual_overlap", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    status_weight = pd.to_numeric(frame.get("status_weight"), errors="coerce")
    eligible = ~is_qb & ~excluded
    frame["weighted_role_exposure_v2"] = np.where(
        eligible & frame["expected_role_v2"].notna(),
        status_weight * frame["expected_role_v2"],
        np.nan,
    )
    return frame


def add_v2_shadow_features(audit: pd.DataFrame) -> pd.DataFrame:
    """Add V2 diagnostics without changing the production personnel correction."""
    frame = audit.copy()
    if frame.empty:
        for column, dtype in {
            "position_group": object,
            "role_quality_flag": object,
            "v2_shadow_eligible": bool,
        }.items():
            frame[column] = pd.Series(dtype=dtype)
        return frame

    matched_position = frame.get(
        "matched_position", pd.Series(index=frame.index, dtype=object)
    )
    reported_position = frame.get(
        "position_reported", pd.Series(index=frame.index, dtype=object)
    )
    preferred_position = matched_position.where(
        matched_position.notna() & matched_position.astype(str).str.strip().ne(""),
        reported_position,
    )
    inferred_group = preferred_position.map(normalize_position_group)
    override_group = frame.get(
        "position_group_v2_override",
        pd.Series(index=frame.index, dtype=object),
    )
    frame["position_group"] = override_group.where(
        override_group.notna() & override_group.astype(str).str.strip().ne(""),
        inferred_group,
    )

    expected_role = pd.to_numeric(
        frame.get("expected_role_v2", frame.get("expected_role")),
        errors="coerce",
    )
    roster_status = (
        frame.get("roster_status", pd.Series(index=frame.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    match_method = (
        frame.get("match_method", pd.Series(index=frame.index, dtype=object))
        .fillna("")
        .astype(str)
    )
    is_qb = frame.get("is_qb", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    excluded = frame.get(
        "excluded_manual_overlap", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)

    repair_applied = frame.get(
        "v2_role_repair_applied", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    conditions = [
        is_qb,
        excluded,
        repair_applied & expected_role.eq(0),
        repair_applied,
        expected_role.isna(),
        expected_role.lt(0) | expected_role.gt(1),
        roster_status.ne("") & ~roster_status.eq("ACT"),
        roster_status.eq("ACT") & expected_role.le(0.02),
        match_method.eq("FUZZY_REVIEW"),
    ]
    labels = [
        "EXCLUDED_QB",
        "EXCLUDED_MANUAL_OVERLAP",
        "REVIEWED_ZERO_ROLE",
        "V2_ROLE_REPAIRED",
        "UNRESOLVED_ROLE",
        "INVALID_ROLE_RANGE",
        "STALE_ROSTER_STATUS",
        "ACTIVE_LOW_ROLE_PRIOR",
        "FUZZY_MATCH_REVIEW",
    ]
    frame["role_quality_flag"] = np.select(conditions, labels, default="OK")
    frame["v2_shadow_eligible"] = ~is_qb & ~excluded
    return frame


def build_personnel_v2_shadow(
    schedule: pd.DataFrame,
    audit: pd.DataFrame,
    season: int,
    week: int,
    role_repairs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a complete game/team/position shadow matrix for later coefficient fitting."""
    repaired = apply_v2_role_repairs(audit, role_repairs)
    decorated = add_v2_shadow_features(repaired)
    eligible = decorated.loc[decorated["v2_shadow_eligible"]].copy()

    aggregate: dict[tuple[str, str], dict[str, float | int]] = {}
    if not eligible.empty:
        role_column = (
            "expected_role_v2" if "expected_role_v2" in eligible else "expected_role"
        )
        exposure_column = (
            "weighted_role_exposure_v2"
            if "weighted_role_exposure_v2" in eligible
            else "weighted_role_exposure"
        )
        eligible["expected_role_numeric"] = pd.to_numeric(
            eligible[role_column], errors="coerce"
        )
        eligible["weighted_role_exposure_numeric"] = pd.to_numeric(
            eligible[exposure_column], errors="coerce"
        )
        eligible["unresolved_status_weight"] = np.where(
            eligible["expected_role_numeric"].isna(),
            pd.to_numeric(eligible["status_weight"], errors="coerce").fillna(0.0),
            0.0,
        )
        eligible["review_required"] = ~eligible["role_quality_flag"].isin(
            {"OK", "V2_ROLE_REPAIRED", "REVIEWED_ZERO_ROLE"}
        )
        eligible["low_role_prior"] = eligible["role_quality_flag"].eq(
            "ACTIVE_LOW_ROLE_PRIOR"
        )

        for (team, group), rows in eligible.groupby(["team", "position_group"]):
            aggregate[(str(team), str(group))] = {
                "known_weighted_role_exposure": float(
                    rows["weighted_role_exposure_numeric"].sum(min_count=1)
                )
                if rows["weighted_role_exposure_numeric"].notna().any()
                else 0.0,
                "unresolved_status_weight": float(rows["unresolved_status_weight"].sum()),
                "player_count": int(len(rows)),
                "known_role_count": int(rows["expected_role_numeric"].notna().sum()),
                "review_count": int(rows["review_required"].sum()),
                "active_low_role_prior_count": int(rows["low_role_prior"].sum()),
            }

    output_rows: list[dict[str, object]] = []
    for game in schedule.itertuples():
        game_id = getattr(
            game,
            "game_id",
            f"{season}_{week:02d}_{game.away_team}_{game.home_team}",
        )
        for side, team, opponent in (
            ("away", game.away_team, game.home_team),
            ("home", game.home_team, game.away_team),
        ):
            for group in POSITION_GROUPS:
                values = aggregate.get(
                    (str(team), group),
                    {
                        "known_weighted_role_exposure": 0.0,
                        "unresolved_status_weight": 0.0,
                        "player_count": 0,
                        "known_role_count": 0,
                        "review_count": 0,
                        "active_low_role_prior_count": 0,
                    },
                )
                output_rows.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "game_id": game_id,
                        "team": team,
                        "opponent": opponent,
                        "side": side,
                        "position_group": group,
                        **values,
                    }
                )
    return pd.DataFrame(output_rows), decorated
