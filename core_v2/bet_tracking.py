from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BET_COLUMNS = [
    "bet_id", "season", "week", "game_id", "matchup", "team", "bet_spread",
    "odds_american", "sportsbook", "stake", "entered_at_utc",
    "model_fair_home_spread_at_entry", "market_home_spread_at_entry",
    "model_side_at_entry", "model_edge_points_at_entry",
    "model_selected_team_spread_at_entry", "bet_vs_model_points", "bet_alignment",
    "status", "notes",
]


def load_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    bets = pd.read_csv(path)
    for column in BET_COLUMNS:
        if column not in bets:
            bets[column] = pd.NA
    return bets[BET_COLUMNS].copy()


def make_bet_id(season: int, week: int, game_id: str, team: str, entered_at_utc: str) -> str:
    stamp = pd.Timestamp(entered_at_utc).strftime("%Y%m%dT%H%M%S")
    return f"{int(season)}-{int(week):02d}-{game_id}-{team}-{stamp}"


def append_bet(path: Path, record: dict) -> pd.DataFrame:
    bets = load_bets(path)
    row = {column: record.get(column, pd.NA) for column in BET_COLUMNS}
    if pd.isna(row["entered_at_utc"]):
        row["entered_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if pd.isna(row["bet_id"]):
        row["bet_id"] = make_bet_id(
            int(row["season"]), int(row["week"]), str(row["game_id"]),
            str(row["team"]), str(row["entered_at_utc"]),
        )
    updated = pd.concat([bets, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return updated


def american_profit(stake: float, odds: float) -> float:
    stake, odds = float(stake), float(odds)
    return stake * (100.0 / abs(odds)) if odds < 0 else stake * (odds / 100.0)


def selected_team_close(row: pd.Series, closing_home_spread: float) -> float:
    if pd.isna(closing_home_spread):
        return float("nan")
    if str(row["team"]) == str(row.get("home_team")):
        return float(closing_home_spread)
    return -float(closing_home_spread)


def selected_team_model_spread(team: str, away_team: str, home_team: str, fair_home_spread: float) -> float:
    """Convert the home-team fair spread to the selected team's spread."""
    fair_home_spread = float(fair_home_spread)
    if str(team) == str(home_team):
        return fair_home_spread
    if str(team) == str(away_team):
        return -fair_home_spread
    raise ValueError(f"Bet team {team} is not in {away_team} @ {home_team}")


def classify_bet_alignment(
    bet_spread: float,
    selected_team_model_spread_value: float,
    neutral_tolerance: float = 0.25,
) -> tuple[float, str]:
    """Positive points means the bettor got a better spread than Core V2 fair.

    This is price alignment, not a claim that the model explicitly selected the
    opposite team. A small difference is labeled NEUTRAL.
    """
    points = float(bet_spread) - float(selected_team_model_spread_value)
    if points > neutral_tolerance:
        alignment = "WITH_MODEL"
    elif points < -neutral_tolerance:
        alignment = "AGAINST_MODEL"
    else:
        alignment = "NEUTRAL"
    return points, alignment


def grade_bets_against_closes(bets: pd.DataFrame, game_grades: pd.DataFrame) -> pd.DataFrame:
    """Attach actual-wager closing spread, CLV, ATS result, and P/L.

    CLV is always from the bettor's perspective:
        bet_spread - closing_selected_team_spread
    Positive CLV means the bettor beat the close.
    """
    out = bets.copy()
    for column in ["closing_selected_team_spread", "actual_bet_clv_points", "actual_bet_result", "profit_loss"]:
        out[column] = pd.NA
    if out.empty or game_grades.empty:
        return out

    grades = game_grades.copy()
    grades["matchup"] = grades["away_team"].astype(str) + " @ " + grades["home_team"].astype(str)
    keys = ["season", "week", "matchup"]
    merged = out.merge(
        grades[keys + ["away_team", "home_team", "closing_home_spread", "closing_status", "away_score", "home_score"]],
        on=keys,
        how="left",
    )

    def grade_row(row):
        if row.get("closing_status") != "CAPTURED" or pd.isna(row.get("closing_home_spread")):
            return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])

        close = float(row["closing_home_spread"])
        if str(row["team"]) == str(row["home_team"]):
            selected_close = close
            team_score, opp_score = row.get("home_score"), row.get("away_score")
        elif str(row["team"]) == str(row["away_team"]):
            selected_close = -close
            team_score, opp_score = row.get("away_score"), row.get("home_score")
        else:
            return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])

        clv = float(row["bet_spread"]) - selected_close
        result = pd.NA
        pnl = pd.NA
        if pd.notna(team_score) and pd.notna(opp_score):
            ats_margin = float(team_score) - float(opp_score) + float(row["bet_spread"])
            if ats_margin > 0:
                result = "Win"
                pnl = american_profit(float(row["stake"]), float(row["odds_american"]))
            elif ats_margin < 0:
                result = "Loss"
                pnl = -float(row["stake"])
            else:
                result = "Push"
                pnl = 0.0
        return pd.Series([selected_close, clv, result, pnl])

    merged[["closing_selected_team_spread", "actual_bet_clv_points", "actual_bet_result", "profit_loss"]] = merged.apply(grade_row, axis=1)
    return merged
