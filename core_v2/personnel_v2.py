from __future__ import annotations

import numpy as np
import pandas as pd


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
    frame["position_group"] = preferred_position.map(normalize_position_group)

    expected_role = pd.to_numeric(frame.get("expected_role"), errors="coerce")
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

    conditions = [
        is_qb,
        excluded,
        expected_role.isna(),
        expected_role.lt(0) | expected_role.gt(1),
        roster_status.eq("ACT") & expected_role.le(0.02),
        match_method.eq("FUZZY_REVIEW"),
    ]
    labels = [
        "EXCLUDED_QB",
        "EXCLUDED_MANUAL_OVERLAP",
        "UNRESOLVED_ROLE",
        "INVALID_ROLE_RANGE",
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a complete game/team/position shadow matrix for later coefficient fitting."""
    decorated = add_v2_shadow_features(audit)
    eligible = decorated.loc[decorated["v2_shadow_eligible"]].copy()

    aggregate: dict[tuple[str, str], dict[str, float | int]] = {}
    if not eligible.empty:
        eligible["expected_role_numeric"] = pd.to_numeric(
            eligible["expected_role"], errors="coerce"
        )
        eligible["weighted_role_exposure_numeric"] = pd.to_numeric(
            eligible["weighted_role_exposure"], errors="coerce"
        )
        eligible["unresolved_status_weight"] = np.where(
            eligible["expected_role_numeric"].isna(),
            pd.to_numeric(eligible["status_weight"], errors="coerce").fillna(0.0),
            0.0,
        )
        eligible["review_required"] = ~eligible["role_quality_flag"].isin({"OK"})
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
