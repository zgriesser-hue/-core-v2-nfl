from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .config import CORE_CONSENSUS_BOOKS, EXCLUDED_BOOKS
from .io import clean_bool
from .projections import game_key, normalize_team


API_URL = "https://parlay-api.com/v1/sports/americanfootball_nfl/odds"

FULL_TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

BOOK_MAP = {
    "draft_kings": "draftkings",
    "fan_duel": "fanduel",
    "bet_mgm": "betmgm",
    "williamhill_us": "caesars",
    "bet_parx": "betparx",
    "parx": "betparx",
    "the_score": "thescore",
    "betonlineag": "betonline",
}

USER_AVAILABLE_BOOKS = frozenset(
    {"draftkings", "fanduel", "bet365", "betparx", "caesars", "betmgm",
     "fanatics", "thescore", "betonline"}
)

BOOK_DISPLAY = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "bet365": "Bet365",
    "betparx": "betPARX", "caesars": "Caesars", "betmgm": "BetMGM",
    "fanatics": "Fanatics", "thescore": "theScore Bet", "betonline": "BetOnline",
}


def dashboard_lines(quotes: pd.DataFrame) -> pd.DataFrame:
    lines = quotes.loc[quotes["sportsbook"].isin(USER_AVAILABLE_BOOKS)].copy()
    lines["matchup"] = lines["away_team"] + " @ " + lines["home_team"]
    lines["book_display"] = lines["sportsbook"].map(BOOK_DISPLAY).fillna(lines["sportsbook"])
    lines["book_feed_status"] = np.where(
        lines["sportsbook"].isin(EXCLUDED_BOOKS), "STALE_BOOK_FEED", "ACTIVE"
    )
    snapshot = pd.to_datetime(lines["market_snapshot_timestamp_et"], utc=True, errors="coerce")
    observed = pd.to_datetime(lines["observed_timestamp_et"], utc=True, errors="coerce")
    lines["quote_age_minutes"] = (snapshot - observed).dt.total_seconds().div(60)
    lines["quote_status"] = np.select(
        [
            lines["book_feed_status"].eq("STALE_BOOK_FEED"),
            lines["quote_age_minutes"].isna(),
            lines["quote_age_minutes"].gt(60),
            lines["quote_age_minutes"].gt(15),
        ],
        ["STALE BOOK — VERIFY", "UNKNOWN TIMESTAMP", "OLD — VERIFY", "AGING — VERIFY"],
        default="FRESH",
    )
    return lines


def normalize_book(value: object) -> str | None:
    if pd.isna(value):
        return None
    book = str(value).strip().lower().replace(" ", "")
    return BOOK_MAP.get(book, book)


def normalize_market_team(value: object) -> str:
    raw = str(value).strip()
    return FULL_TEAM_MAP.get(raw, normalize_team(raw))


def decimal_to_american(value: object) -> float:
    try:
        decimal = float(value)
    except (TypeError, ValueError):
        return np.nan
    if decimal <= 1:
        return np.nan
    if decimal >= 2:
        return round((decimal - 1) * 100)
    return round(-100 / (decimal - 1))


def extract_events(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "events"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise RuntimeError("The ParlayAPI response did not contain an event list")


def parse_spread_payload(
    payload: object,
    season: int,
    week: int,
    allowed_game_keys: set[str],
    snapshot_timestamp: str | None = None,
) -> pd.DataFrame:
    """Parse one two-sided spread quote per game/book from a provider response."""
    quotes = parse_market_payload(
        payload,
        season,
        week,
        allowed_game_keys,
        snapshot_timestamp,
    )
    quotes = quotes.loc[
        quotes[["away_spread", "home_spread"]].notna().all(axis=1)
    ].copy()
    if quotes.empty:
        raise RuntimeError("No valid spreads matched the target schedule")
    return quotes.sort_values(["game_key", "sportsbook"]).reset_index(drop=True)


def parse_market_payload(
    payload: object,
    season: int,
    week: int,
    allowed_game_keys: set[str],
    snapshot_timestamp: str | None = None,
) -> pd.DataFrame:
    """Parse spread and total quotes into one nullable row per game/book."""
    snapshot_timestamp = snapshot_timestamp or datetime.now(
        ZoneInfo("America/New_York")
    ).isoformat(timespec="seconds")
    rows: list[dict] = []

    for event in extract_events(payload):
        away = normalize_market_team(event.get("away_team") or event.get("away"))
        home = normalize_market_team(event.get("home_team") or event.get("home"))
        key = game_key(season, week, away, home)
        if key not in allowed_game_keys:
            continue

        for book in event.get("bookmakers") or event.get("books") or []:
            sportsbook = normalize_book(
                book.get("key") or book.get("bookmaker") or book.get("name")
            )
            if not sportsbook:
                continue
            book_update = book.get("last_update") or book.get("lastUpdate")

            row = {
                "season": int(season),
                "week": int(week),
                "game_key": key,
                "away_team": away,
                "home_team": home,
                "sportsbook": sportsbook,
                "market_snapshot_timestamp_et": snapshot_timestamp,
                "observed_timestamp_et": book_update,
                "spread_observed_timestamp_et": None,
                "total_observed_timestamp_et": None,
                "moneyline_observed_timestamp_et": None,
                "away_spread": np.nan,
                "away_price_american": np.nan,
                "home_spread": np.nan,
                "home_price_american": np.nan,
                "total_points": np.nan,
                "over_price_american": np.nan,
                "under_price_american": np.nan,
                "away_moneyline_american": np.nan,
                "home_moneyline_american": np.nan,
                "book_topped_up_clean": clean_bool(book.get("topped_up")),
            }

            for market in book.get("markets") or []:
                market_name = market.get("key") or market.get("market")
                market_name = str(market_name).strip().lower()
                outcomes = market.get("outcomes") or []
                observed = (
                    market.get("last_update")
                    or market.get("lastUpdate")
                    or book_update
                )

                if market_name in {"spread", "spreads"}:
                    by_team = {
                        normalize_market_team(
                            outcome.get("name") or outcome.get("team")
                        ): outcome
                        for outcome in outcomes
                    }
                    away_outcome = by_team.get(away)
                    home_outcome = by_team.get(home)
                    if away_outcome is None or home_outcome is None:
                        continue
                    away_spread = pd.to_numeric(
                        away_outcome.get("point"), errors="coerce"
                    )
                    home_spread = pd.to_numeric(
                        home_outcome.get("point"), errors="coerce"
                    )
                    away_decimal = pd.to_numeric(
                        away_outcome.get("price"), errors="coerce"
                    )
                    home_decimal = pd.to_numeric(
                        home_outcome.get("price"), errors="coerce"
                    )
                    if any(
                        pd.isna(value)
                        for value in (
                            away_spread,
                            home_spread,
                            away_decimal,
                            home_decimal,
                        )
                    ):
                        continue
                    if abs(float(away_spread) + float(home_spread)) > 0.001:
                        raise RuntimeError(
                            f"Non-opposite spread pair for {key} at {sportsbook}"
                        )
                    row.update(
                        {
                            "observed_timestamp_et": observed,
                            "spread_observed_timestamp_et": observed,
                            "away_spread": float(away_spread),
                            "away_price_american": decimal_to_american(away_decimal),
                            "home_spread": float(home_spread),
                            "home_price_american": decimal_to_american(home_decimal),
                        }
                    )

                elif market_name in {"total", "totals"}:
                    by_side = {
                        str(outcome.get("name", "")).strip().lower(): outcome
                        for outcome in outcomes
                    }
                    over = by_side.get("over")
                    under = by_side.get("under")
                    if over is None or under is None:
                        continue
                    over_point = pd.to_numeric(over.get("point"), errors="coerce")
                    under_point = pd.to_numeric(under.get("point"), errors="coerce")
                    over_decimal = pd.to_numeric(over.get("price"), errors="coerce")
                    under_decimal = pd.to_numeric(under.get("price"), errors="coerce")
                    if any(
                        pd.isna(value)
                        for value in (
                            over_point,
                            under_point,
                            over_decimal,
                            under_decimal,
                        )
                    ):
                        continue
                    if abs(float(over_point) - float(under_point)) > 0.001:
                        raise RuntimeError(
                            f"Non-matching total pair for {key} at {sportsbook}"
                        )
                    row.update(
                        {
                            "total_observed_timestamp_et": observed,
                            "total_points": float(over_point),
                            "over_price_american": decimal_to_american(over_decimal),
                            "under_price_american": decimal_to_american(under_decimal),
                        }
                    )

                elif market_name in {"h2h", "moneyline", "moneylines"}:
                    by_team = {
                        normalize_market_team(outcome.get("name") or outcome.get("team")): outcome
                        for outcome in outcomes
                    }
                    away_outcome = by_team.get(away)
                    home_outcome = by_team.get(home)
                    if away_outcome is None or home_outcome is None:
                        continue
                    away_decimal = pd.to_numeric(away_outcome.get("price"), errors="coerce")
                    home_decimal = pd.to_numeric(home_outcome.get("price"), errors="coerce")
                    if pd.isna(away_decimal) or pd.isna(home_decimal):
                        continue
                    row.update(
                        {
                            "moneyline_observed_timestamp_et": observed,
                            "away_moneyline_american": decimal_to_american(away_decimal),
                            "home_moneyline_american": decimal_to_american(home_decimal),
                        }
                    )

            if (
                pd.notna(row["away_spread"])
                or pd.notna(row["total_points"])
                or pd.notna(row["away_moneyline_american"])
            ):
                rows.append(row)

    quotes = pd.DataFrame(rows)
    if quotes.empty:
        raise RuntimeError("No valid spreads or totals matched the target schedule")
    duplicates = quotes.duplicated(["game_key", "sportsbook"], keep=False)
    if duplicates.any():
        bad = quotes.loc[duplicates, ["game_key", "sportsbook"]]
        raise RuntimeError(f"Duplicate game/book quotes parsed: {bad.to_dict('records')}")
    return quotes.sort_values(["game_key", "sportsbook"]).reset_index(drop=True)


def request_spreads(api_key: str, timeout: int = 30) -> tuple[object, dict]:
    return request_markets(api_key, markets=("spreads",), timeout=timeout)


def request_markets(
    api_key: str,
    markets: tuple[str, ...] = ("spreads", "totals", "h2h"),
    timeout: int = 30,
) -> tuple[object, dict]:
    response = requests.get(
        API_URL,
        headers={"X-API-Key": api_key.strip()},
        params={"regions": "us", "markets": ",".join(markets)},
        timeout=timeout,
    )
    response.raise_for_status()
    metadata = {
        "requests_used": response.headers.get("x-requests-used"),
        "requests_remaining": response.headers.get("x-requests-remaining"),
    }
    return response.json(), metadata


def append_market_history(existing: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Append snapshots without mutating prior observations; repeated runs are idempotent."""
    keys = ["market_snapshot_timestamp_et", "game_key", "sportsbook"]
    combined = pd.concat([existing.copy(), quotes.copy()], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(keys, keep="first")
    return combined.sort_values(keys).reset_index(drop=True)


def latest_market_snapshot(history: pd.DataFrame) -> pd.DataFrame:
    """Return the newest quote for each game/book from append-only history."""
    if history.empty:
        return history.copy()
    latest = history.copy()
    latest["_snapshot"] = pd.to_datetime(
        latest["market_snapshot_timestamp_et"], errors="coerce", utc=True
    )
    latest = latest.sort_values("_snapshot").drop_duplicates(
        ["game_key", "sportsbook"], keep="last"
    )
    return latest.drop(columns="_snapshot").sort_values(
        ["game_key", "sportsbook"]
    ).reset_index(drop=True)


def build_market_consensus(quotes: pd.DataFrame) -> pd.DataFrame:
    """Build current spread and total medians from configured core books."""
    core = quotes.loc[quotes["sportsbook"].isin(CORE_CONSENSUS_BOOKS)].copy()
    group = ["season", "week", "game_key", "away_team", "home_team"]
    consensus = core.groupby(group, as_index=False).agg(
        market_home_spread=("home_spread", "median"),
        market_total=("total_points", "median"),
        spread_books=("home_spread", "count"),
        total_books=("total_points", "count"),
        market_snapshot_timestamp_et=("market_snapshot_timestamp_et", "max"),
    )
    consensus["consensus_method"] = "MEDIAN_LATEST_CORE_BOOKS"
    return consensus


def append_first_seen_openers(
    existing: pd.DataFrame,
    quotes: pd.DataFrame,
    kickoff_by_game: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Capture immutable first-seen spread and total values for each game/book."""
    kickoff_by_game = kickoff_by_game or {}
    existing = existing.copy()
    if not existing.empty:
        existing["book"] = existing["book"].map(normalize_book)
    for column in (
        "opening_total",
        "opening_over_price",
        "opening_under_price",
        "total_opening_seen_utc",
    ):
        if column not in existing.columns:
            existing[column] = np.nan

    eligible = quotes.loc[~quotes["sportsbook"].isin(EXCLUDED_BOOKS)].copy()
    known = set(zip(existing.get("game_key", []), existing.get("book", [])))
    new = eligible.loc[
        ~eligible.apply(
            lambda row: (row["game_key"], row["sportsbook"]) in known, axis=1
        )
    ]
    captured = pd.DataFrame(
        {
            "season": new["season"],
            "week": new["week"],
            "game_key": new["game_key"],
            "away_team": new["away_team"],
            "home_team": new["home_team"],
            "kickoff_utc": new["game_key"].map(kickoff_by_game),
            "book": new["sportsbook"],
            "opening_home_spread": new["home_spread"],
            "opening_home_price": new["home_price_american"],
            "opening_seen_utc": new["market_snapshot_timestamp_et"],
            "opening_total": new["total_points"],
            "opening_over_price": new["over_price_american"],
            "opening_under_price": new["under_price_american"],
            "total_opening_seen_utc": new["market_snapshot_timestamp_et"],
            "source": "PARLAYAPI_FIRST_OBSERVED",
            "market_stage": "OPEN",
        }
    )
    combined = pd.concat([existing, captured], ignore_index=True)

    # A sportsbook can publish its spread before its total. Fill only a still-missing
    # total opener from the first later snapshot and never revise a captured value.
    if not combined.empty and not eligible.empty:
        quote_lookup = eligible.set_index(["game_key", "sportsbook"])
        for index, row in combined.loc[combined["opening_total"].isna()].iterrows():
            key = (row["game_key"], row["book"])
            if key not in quote_lookup.index:
                continue
            quote = quote_lookup.loc[key]
            if isinstance(quote, pd.DataFrame):
                quote = quote.iloc[0]
            if pd.isna(quote.get("total_points")):
                continue
            combined.loc[index, "opening_total"] = quote["total_points"]
            combined.loc[index, "opening_over_price"] = quote["over_price_american"]
            combined.loc[index, "opening_under_price"] = quote["under_price_american"]
            combined.loc[index, "total_opening_seen_utc"] = quote[
                "market_snapshot_timestamp_et"
            ]
    if combined.duplicated(["game_key", "book"]).any():
        raise RuntimeError("Opener history contains duplicate game/book pairs")
    return combined.sort_values(["game_key", "book"]).reset_index(drop=True)


def build_consensus_openers(openers: pd.DataFrame) -> pd.DataFrame:
    core = openers.loc[openers["book"].isin(CORE_CONSENSUS_BOOKS)].copy()
    if "opening_total" not in core.columns:
        core["opening_total"] = np.nan
    consensus = (
        core.groupby(
            ["season", "week", "game_key", "away_team", "home_team"], as_index=False
        )
        .agg(
            consensus_open_home_spread=("opening_home_spread", "median"),
            consensus_open_books=("book", "nunique"),
            consensus_open_total=("opening_total", "median"),
            consensus_open_total_books=("opening_total", "count"),
            earliest_core_open_seen_utc=("opening_seen_utc", "min"),
            latest_core_open_seen_utc=("opening_seen_utc", "max"),
        )
    )
    consensus["consensus_method"] = "MEDIAN_FIRST_OBSERVED_CORE_BOOKS"
    return consensus
