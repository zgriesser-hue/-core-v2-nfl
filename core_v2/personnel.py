from __future__ import annotations

import re
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .projections import load_week_schedule, normalize_team


PERSONNEL_COEFFICIENT = -0.61336750
PERSONNEL_CENTER_OFFSET = -0.00303256


def normalize_player_name(value: object) -> str:
    value = str(value).lower().strip()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", value)
    value = re.sub(r"[^a-z ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def frozen_status_weight(game_status: object, practice_status: object) -> tuple[float, str]:
    game = str(game_status).upper().strip()
    practice = str(practice_status).upper().strip()
    if game in {"OUT", "RESERVE/COVID-19"} or "OUT" in game:
        return 1.00, "OUT"
    if "DOUBTFUL" in game:
        return 0.95, "DOUBTFUL"
    if "QUESTIONABLE" in game:
        return 0.35, "QUESTIONABLE"
    if "DID NOT PARTICIPATE" in practice or practice == "DNP":
        return 0.20, "DNP_NO_STATUS"
    if "LIMITED" in practice:
        return 0.05, "LIMITED_NO_STATUS"
    return 0.00, "NO_EXPOSURE"


def _parse_game_teams(text: str, valid_teams: set[str]) -> tuple[str | None, str | None]:
    aliases = {"LAR": "LA", "AZ": "ARI", "WSH": "WAS", "JAC": "JAX"}
    tokens = re.findall(r"\b[A-Z]{2,3}\b", re.sub(r"\s+", " ", text))
    ordered: list[str] = []
    for token in tokens:
        token = aliases.get(token, token)
        if token in valid_teams and token not in ordered:
            ordered.append(token)
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return None, None


def parse_nfl_injury_html(html: str, valid_teams: set[str]) -> tuple[pd.DataFrame, set[str]]:
    """Parse NFL.com's injury tables and retain the table-to-team relationship."""
    frames = pd.read_html(StringIO(html))
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.find_all("table")
    if not nodes or len(frames) != len(nodes):
        raise ValueError("NFL injury table structure was not recognized")

    rows: list[dict[str, object]] = []
    report_teams: set[str] = set()
    mapped = 0
    for index, (node, frame) in enumerate(zip(nodes, frames)):
        container = node.parent.parent if node.parent is not None else None
        if container is None:
            continue
        away, home = _parse_game_teams(" ".join(container.stripped_strings), valid_teams)
        game_tables = container.find_all("table")
        position = next((i for i, table in enumerate(game_tables) if table is node), None)
        team = away if position == 0 else home if position == 1 else None
        if team is None:
            continue
        mapped += 1
        report_teams.add(team)

        table = frame.copy()
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                " ".join(str(value) for value in key if str(value).lower() != "nan").strip()
                for key in table.columns
            ]
        table.columns = [str(column).strip() for column in table.columns]
        required = {"Player", "Position", "Practice Status", "Game Status"}
        if not required.issubset(table.columns):
            continue
        for _, record in table.iterrows():
            player = str(record.get("Player", "")).strip()
            if not player or player.lower() == "nan":
                continue
            rows.append(
                {
                    "team": team,
                    "player_name": player,
                    "position_reported": str(record.get("Position", "")).strip(),
                    "injury": str(record.get("Injuries", "")).strip(),
                    "practice_status": str(record.get("Practice Status", "")).strip(),
                    "game_status": str(record.get("Game Status", "")).strip(),
                    "source_table": index,
                }
            )

    if mapped / len(nodes) < 0.95:
        raise ValueError("NFL injury table-to-team mapping coverage is below 95%")
    return pd.DataFrame(rows), report_teams


def fetch_nfl_injury_report(season: int, week: int, valid_teams: set[str]) -> tuple[pd.DataFrame, set[str]]:
    url = f"https://www.nfl.com/injuries/league/{int(season)}/reg{int(week)}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 Core-V2-NFL/0.2"},
        timeout=30,
    )
    response.raise_for_status()

    # NFL.com can leave the previous week's injury report behind the next
    # week's URL early in the week.  Never treat those stale final game
    # designations (especially OUT/DOUBTFUL) as current-week statuses.
    soup = BeautifulSoup(response.text, "html.parser")
    heading_text = " ".join(
        node.get_text(" ", strip=True)
        for node in soup.find_all(["title", "h1", "h2"])
    )
    week_patterns = [
        rf"\\bweek\\s*{int(week)}\\b",
        rf"\\breg(?:ular season)?\\s*{int(week)}\\b",
    ]
    if not any(
        re.search(pattern, heading_text, flags=re.IGNORECASE)
        for pattern in week_patterns
    ):
        raise ValueError(
            f"NFL injury page does not identify itself as {season} Week {week}; "
            "refusing to reuse potentially stale prior-week designations"
        )

    report, teams = parse_nfl_injury_html(response.text, valid_teams)
    report["source_url"] = url
    report["source_season"] = int(season)
    report["source_week"] = int(week)
    return report, teams


def load_role_priors(path: Path) -> pd.DataFrame:
    roles = pd.read_csv(path)
    required = {"team", "player_name", "position", "expected_role"}
    missing = required - set(roles.columns)
    if missing:
        raise ValueError(f"Role priors missing columns: {sorted(missing)}")
    roles["team"] = roles["team"].map(normalize_team)
    roles["name_key"] = roles["player_name"].map(normalize_player_name)
    roles["expected_role"] = pd.to_numeric(roles["expected_role"], errors="coerce")
    return roles


def load_weekly_team_adjustments(path: Path | None, season: int, week: int) -> pd.DataFrame:
    columns = ["team", "adjustment_points", "category", "player", "reason", "exclude_from_auto"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)
    adjustments = pd.read_csv(path)
    required = {"season", "week", "team", "adjustment_points"}
    missing = required - set(adjustments.columns)
    if missing:
        raise ValueError(f"Weekly adjustments missing columns: {sorted(missing)}")
    start_week = pd.to_numeric(adjustments["week"], errors="raise")
    end_week = (
        pd.to_numeric(adjustments["end_week"], errors="coerce").fillna(start_week)
        if "end_week" in adjustments
        else start_week
    )
    if (end_week < start_week).any() or end_week.gt(18).any():
        raise ValueError("Weekly adjustment end_week must be between week and 18")
    adjustments = adjustments.loc[
        (pd.to_numeric(adjustments["season"], errors="coerce") == int(season))
        & start_week.le(int(week))
        & end_week.ge(int(week))
    ].copy()
    adjustments["team"] = adjustments["team"].map(normalize_team)
    adjustments["adjustment_points"] = pd.to_numeric(
        adjustments["adjustment_points"], errors="raise"
    )
    if "out_adjustment_points" in adjustments:
        provisional = adjustments["category"].eq("QB_PROVISIONAL")
        out_points = pd.to_numeric(
            adjustments["out_adjustment_points"], errors="coerce"
        )
        if out_points[provisional].isna().any() or (
            out_points[provisional] > 0
        ).any():
            raise ValueError("Provisional QB scenarios require a nonpositive out adjustment")
    for column in columns:
        if column not in adjustments:
            adjustments[column] = "" if column != "exclude_from_auto" else False
    adjustments["exclude_from_auto"] = (
        adjustments["exclude_from_auto"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    return adjustments


def _match_roles(report: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    report["team"] = report["team"].map(normalize_team)
    report["name_key"] = report["player_name"].map(normalize_player_name)
    matched_rows: list[dict[str, object]] = []
    for _, injury in report.iterrows():
        candidates = roles.loc[roles["team"].eq(injury["team"])].copy()
        exact = candidates.loc[candidates["name_key"].eq(injury["name_key"])]
        match = None
        method = "UNMATCHED"
        similarity = np.nan
        if not exact.empty:
            match = exact.sort_values("expected_role", ascending=False).iloc[0]
            method = "EXACT" if len(exact) == 1 else "EXACT_DUPLICATE_MAX_ROLE"
            similarity = 1.0
        elif not candidates.empty:
            scores = candidates["name_key"].map(
                lambda name: SequenceMatcher(None, injury["name_key"], name).ratio()
            )
            best_index = scores.idxmax()
            if scores.loc[best_index] >= 0.88:
                match = candidates.loc[best_index]
                method = "FUZZY_REVIEW"
                similarity = float(scores.loc[best_index])
        output = injury.to_dict()
        output.update(
            {
                "matched_player": None if match is None else match["player_name"],
                "matched_position": None if match is None else match["position"],
                "expected_role": np.nan if match is None else match["expected_role"],
                "role_source": None if match is None else match.get("role_source"),
                "gsis_id": None if match is None else match.get("gsis_id"),
                "roster_status": None if match is None else match.get("roster_status"),
                "match_method": method,
                "name_similarity": similarity,
            }
        )
        matched_rows.append(output)
    return pd.DataFrame(matched_rows)


def build_personnel_adjustments(
    schedule: pd.DataFrame,
    roles: pd.DataFrame,
    injury_report: pd.DataFrame,
    report_teams: set[str],
    weekly_adjustments: pd.DataFrame,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = injury_report.copy()
    required = {"team", "player_name", "position_reported", "practice_status", "game_status"}
    missing = required - set(report.columns)
    if missing and not report.empty:
        raise ValueError(f"Injury report missing columns: {sorted(missing)}")
    if report.empty:
        report = pd.DataFrame(columns=sorted(required))

    weights = report.apply(
        lambda row: frozen_status_weight(row["game_status"], row["practice_status"]), axis=1
    )
    report["status_weight"] = [result[0] for result in weights]
    report["model_status"] = [result[1] for result in weights]
    model_rows = report.loc[report["status_weight"].gt(0)].copy()
    audit = _match_roles(model_rows, roles) if not model_rows.empty else model_rows.assign(
        matched_player=pd.Series(dtype=object), matched_position=pd.Series(dtype=object),
        expected_role=pd.Series(dtype=float), role_source=pd.Series(dtype=object),
        gsis_id=pd.Series(dtype=object), roster_status=pd.Series(dtype=object),
        match_method=pd.Series(dtype=object), name_similarity=pd.Series(dtype=float),
    )
    audit["is_qb"] = (
        audit.get("position_reported", pd.Series(dtype=str)).astype(str).str.upper().eq("QB")
        | audit.get("matched_position", pd.Series(dtype=str)).fillna("").astype(str).str.upper().eq("QB")
    )

    excluded_keys = {
        (row.team, normalize_player_name(row.player))
        for row in weekly_adjustments.itertuples()
        if bool(row.exclude_from_auto) and str(row.player).strip()
    }
    if audit.empty:
        audit["excluded_manual_overlap"] = pd.Series(dtype=bool)
    else:
        audit["excluded_manual_overlap"] = audit.apply(
            lambda row: (row["team"], row["name_key"]) in excluded_keys, axis=1
        )
    eligible = ~audit["is_qb"] & ~audit["excluded_manual_overlap"]
    audit["weighted_role_exposure"] = np.where(
        eligible & audit["expected_role"].notna(),
        audit["status_weight"] * audit["expected_role"],
        np.nan,
    )

    known = audit.loc[eligible & audit["weighted_role_exposure"].notna()].groupby("team")[
        "weighted_role_exposure"
    ].sum().to_dict()
    unresolved = audit.loc[eligible & audit["expected_role"].isna()].groupby("team")[
        "status_weight"
    ].sum().to_dict()
    adjustment_map = weekly_adjustments.groupby("team")["adjustment_points"].sum().to_dict()
    provisional = weekly_adjustments["category"].eq("QB_PROVISIONAL")
    if provisional.any():
        if "out_adjustment_points" not in weekly_adjustments:
            raise ValueError("Provisional QB scenarios require out_adjustment_points")
        out_points = pd.to_numeric(
            weekly_adjustments["out_adjustment_points"], errors="coerce"
        )
        if out_points[provisional].isna().any() or (out_points[provisional] > 0).any():
            raise ValueError("Provisional QB scenarios require a nonpositive out adjustment")
    else:
        out_points = pd.Series(np.nan, index=weekly_adjustments.index)
    low = weekly_adjustments["adjustment_points"].where(
        ~provisional, out_points
    )
    high = weekly_adjustments["adjustment_points"].where(~provisional, 0.0)
    low_map = low.groupby(weekly_adjustments["team"]).sum().to_dict()
    high_map = high.groupby(weekly_adjustments["team"]).sum().to_dict()
    provisional_teams = set(weekly_adjustments.loc[provisional, "team"])
    report_teams = {normalize_team(team) for team in report_teams}

    rows: list[dict[str, object]] = []
    for game in schedule.itertuples():
        away, home = game.away_team, game.home_team
        away_known, home_known = float(known.get(away, 0.0)), float(known.get(home, 0.0))
        away_unresolved, home_unresolved = float(unresolved.get(away, 0.0)), float(unresolved.get(home, 0.0))
        reports_present = away in report_teams and home in report_teams
        roles_resolved = away_unresolved == 0.0 and home_unresolved == 0.0
        complete = reports_present and roles_resolved
        manual_home = float(adjustment_map.get(home, 0.0) - adjustment_map.get(away, 0.0))
        manual_min_home = float(low_map.get(home, 0.0) - high_map.get(away, 0.0))
        manual_max_home = float(high_map.get(home, 0.0) - low_map.get(away, 0.0))
        current_diff = home_known - away_known
        minimum_diff = home_known - (away_known + away_unresolved)
        maximum_diff = (home_known + home_unresolved) - away_known
        auto_current = PERSONNEL_COEFFICIENT * current_diff + PERSONNEL_CENTER_OFFSET
        auto_min = PERSONNEL_COEFFICIENT * maximum_diff + PERSONNEL_CENTER_OFFSET
        auto_max = PERSONNEL_COEFFICIENT * minimum_diff + PERSONNEL_CENTER_OFFSET
        # Before both reports exist, only explicitly reviewed weekly adjustments are applied.
        if not reports_present:
            auto_current = auto_min = auto_max = 0.0
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "game_id": getattr(game, "game_id", f"{season}_{week:02d}_{away}_{home}"),
                "away_team": away,
                "home_team": home,
                "personnel_correction_current_clean": auto_current + manual_home,
                "personnel_correction_min_clean": auto_min + manual_min_home,
                "personnel_correction_max_clean": auto_max + manual_max_home,
                "report_coverage_complete_clean": complete,
                "personnel_info_complete": complete,
                "week1_personnel_status_clean": (
                    "READY_WITH_UNCERTAINTY_BAND" if complete else "EARLY_PROVISIONAL"
                ),
                "away_report_available": away in report_teams,
                "home_report_available": home in report_teams,
                "away_known_exposure": away_known,
                "home_known_exposure": home_known,
                "away_unresolved_exposure_max": away_unresolved,
                "home_unresolved_exposure_max": home_unresolved,
                "manual_adjustment_home": manual_home,
                "qb_availability_provisional": away in provisional_teams or home in provisional_teams,
            }
        )
    return pd.DataFrame(rows), audit


def build_personnel_from_files(
    schedule_path: Path,
    roles_path: Path,
    injury_path: Path,
    report_teams_path: Path,
    weekly_adjustments_path: Path | None,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_week_schedule(schedule_path, season, week)
    roles = load_role_priors(roles_path)
    injuries = pd.read_csv(injury_path)
    coverage = pd.read_csv(report_teams_path)
    report_teams = set(coverage.loc[coverage["report_available"].astype(bool), "team"])
    weekly = load_weekly_team_adjustments(weekly_adjustments_path, season, week)
    return build_personnel_adjustments(
        schedule, roles, injuries, report_teams, weekly, season, week
    )
