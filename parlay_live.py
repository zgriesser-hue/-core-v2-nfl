from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from core_v2.io import clean_bool


API_URL = "https://parlay-api.com/v1/sports/americanfootball_nfl/odds"

USER_AVAILABLE_BOOKS = {
    "draftkings",
    "fanduel",
    "bet365",
    "betparx",
    "caesars",
    "betmgm",
    "fanatics",
    "thescore",
    "betonline",
}

PRIMARY_BOOKS = {
    "draftkings",
    "fanduel",
    "bet365",
    "betparx",
    "caesars",
    "betonline",
}

SECONDARY_BOOKS = {
    "fanatics",
    "thescore",
}

CORE_CONSENSUS_BOOKS = {
    "draftkings",
    "fanduel",
    "bet365",
    "caesars",
}

EXCLUDED_BOOKS = {
    "betmgm",
}

BOOK_DISPLAY = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "bet365": "Bet365",
    "betparx": "betPARX",
    "caesars": "Caesars",
    "betmgm": "BetMGM",
    "fanatics": "Fanatics",
    "thescore": "theScore Bet",
    "betonline": "BetOnline",
}

TEAM_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "JAC": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LA": "LA",
    "LAR": "LA",
    "LAC": "LAC",
    "LV": "LV",
    "LVR": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}

BOOK_MAP = {
    "draft_kings": "draftkings",
    "draftkings": "draftkings",
    "fan_duel": "fanduel",
    "fanduel": "fanduel",
    "bet_mgm": "betmgm",
    "betmgm": "betmgm",
    "williamhill_us": "caesars",
    "caesars": "caesars",
    "bet365": "bet365",
    "bet_parx": "betparx",
    "betparx": "betparx",
    "parx": "betparx",
    "fanatics": "fanatics",
    "the_score": "thescore",
    "thescore": "thescore",
    "betonline": "betonline",
    "betonlineag": "betonline",
}


def norm_team(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    return TEAM_MAP.get(value, value)


def norm_book(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower().replace(" ", "")
    return BOOK_MAP.get(value, value)


def decimal_to_american(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if value <= 1:
        return np.nan

    if value >= 2:
        return round((value - 1) * 100)

    return round(-100 / (value - 1))


def format_spread(value):
    if pd.isna(value):
        return "â"

    value = float(value)

    if abs(value) < 0.001:
        return "PK"

    if value.is_integer():
        return f"{value:+.0f}"

    return f"{value:+.1f}"


def format_american(value):
    if pd.isna(value):
        return "â"

    value = int(round(float(value)))
    return f"+{value}" if value > 0 else str(value)


def normalize_existing_lines(lines):
    lines = lines.copy()
    lines["sportsbook"] = lines["sportsbook"].map(norm_book)

    if "book_display" not in lines.columns:
        lines["book_display"] = (
            lines["sportsbook"]
            .map(BOOK_DISPLAY)
            .fillna(lines["sportsbook"])
        )

    for column in [
        "away_spread",
        "away_price_american",
        "home_spread",
        "home_price_american",
        "quote_age_minutes",
    ]:
        if column in lines.columns:
            lines[column] = pd.to_numeric(
                lines[column],
                errors="coerce",
            )

    if "book_feed_status" not in lines.columns:
        lines["book_feed_status"] = np.where(
            lines["sportsbook"].isin(EXCLUDED_BOOKS),
            "STALE_BOOK_FEED",
            "ACTIVE",
        )

    if "book_topped_up_clean" not in lines.columns:
        lines["book_topped_up_clean"] = False
    else:
        lines["book_topped_up_clean"] = lines[
            "book_topped_up_clean"
        ].map(clean_bool)

    if "matchup" not in lines.columns:
        lines["matchup"] = (
            lines["away_team"]
            + " @ "
            + lines["home_team"]
        )

    return lines


def fetch_live_lines(api_key, season, week, active_game_keys):
    snapshot_dt = datetime.now(
        ZoneInfo("America/New_York")
    )

    snapshot_timestamp = snapshot_dt.isoformat(
        timespec="seconds"
    )

    response = requests.get(
        API_URL,
        headers={"X-API-Key": api_key.strip()},
        params={
            "regions": "us",
            "markets": "spreads",
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        events = payload
    elif (
        isinstance(payload, dict)
        and isinstance(payload.get("data"), list)
    ):
        events = payload["data"]
    elif (
        isinstance(payload, dict)
        and isinstance(payload.get("events"), list)
    ):
        events = payload["events"]
    else:
        raise RuntimeError(
            "The API response did not contain an event list."
        )

    rows = []
    matched_game_keys = set()

    for event in events:
        away = norm_team(
            event.get("away_team")
            or event.get("away")
        )

        home = norm_team(
            event.get("home_team")
            or event.get("home")
        )

        if not away or not home:
            continue

        game_key = f"{season}_W{week}_{away}_{home}"

        if game_key not in active_game_keys:
            continue

        matched_game_keys.add(game_key)

        bookmakers = (
            event.get("bookmakers")
            or event.get("books")
            or []
        )

        for book in bookmakers:
            sportsbook = norm_book(
                book.get("key")
                or book.get("bookmaker")
                or book.get("name")
            )

            if not sportsbook:
                continue

            book_update = (
                book.get("last_update")
                or book.get("lastUpdate")
            )

            for market in (
                book.get("markets")
                or []
            ):
                market_key = (
                    market.get("key")
                    or market.get("market")
                )

                if str(market_key).strip().lower() not in {
                    "spread",
                    "spreads",
                }:
                    continue

                observed_timestamp = (
                    market.get("last_update")
                    or market.get("lastUpdate")
                    or book_update
                )

                away_outcome = None
                home_outcome = None

                for outcome in (
                    market.get("outcomes")
                    or []
                ):
                    outcome_team = norm_team(
                        outcome.get("name")
                        or outcome.get("team")
                    )

                    if outcome_team == away:
                        away_outcome = outcome
                    elif outcome_team == home:
                        home_outcome = outcome

                if away_outcome is None or home_outcome is None:
                    continue

                away_spread = pd.to_numeric(
                    away_outcome.get("point"),
                    errors="coerce",
                )

                home_spread = pd.to_numeric(
                    home_outcome.get("point"),
                    errors="coerce",
                )

                away_decimal = pd.to_numeric(
                    away_outcome.get("price"),
                    errors="coerce",
                )

                home_decimal = pd.to_numeric(
                    home_outcome.get("price"),
                    errors="coerce",
                )

                if any(
                    pd.isna(value)
                    for value in [
                        away_spread,
                        home_spread,
                        away_decimal,
                        home_decimal,
                    ]
                ):
                    continue

                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_key": game_key,
                        "matchup": f"{away} @ {home}",
                        "away_team": away,
                        "home_team": home,
                        "sportsbook": sportsbook,
                        "book_display": BOOK_DISPLAY.get(
                            sportsbook,
                            sportsbook,
                        ),
                        "market_snapshot_timestamp_et":
                            snapshot_timestamp,
                        "observed_timestamp_et":
                            observed_timestamp,
                        "away_spread": float(away_spread),
                        "away_price_american":
                            decimal_to_american(
                                away_decimal
                            ),
                        "home_spread": float(home_spread),
                        "home_price_american":
                            decimal_to_american(
                                home_decimal
                            ),
                        "book_topped_up_clean": clean_bool(
                            book.get("topped_up")
                        ),
                        "book_feed_status": (
                            "STALE_BOOK_FEED"
                            if sportsbook in EXCLUDED_BOOKS
                            else "ACTIVE"
                        ),
                    }
                )

    quotes = pd.DataFrame(rows)

    if quotes.empty:
        raise RuntimeError(
            "No valid spread quotes were parsed."
        )

    missing_games = (
        set(active_game_keys)
        - matched_game_keys
    )

    if missing_games:
        raise RuntimeError(
            "The API response was missing games: "
            + ", ".join(sorted(missing_games))
        )

    duplicate_count = quotes.duplicated(
        ["game_key", "sportsbook"]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"Parsed {duplicate_count} duplicate game/book rows."
        )

    bad_pairs = (
        quotes["away_spread"]
        + quotes["home_spread"]
    ).abs() > 0.001

    if bad_pairs.any():
        raise RuntimeError(
            "The response contained non-opposite spread pairs."
        )

    quotes["market_snapshot_timestamp_et"] = pd.to_datetime(
        quotes["market_snapshot_timestamp_et"],
        errors="coerce",
        utc=True,
    )

    quotes["observed_timestamp_et"] = pd.to_datetime(
        quotes["observed_timestamp_et"],
        errors="coerce",
        utc=True,
    )

    quotes["quote_age_minutes"] = (
        (
            quotes["market_snapshot_timestamp_et"]
            - quotes["observed_timestamp_et"]
        )
        .dt.total_seconds()
        .div(60)
    )

    quotes["quote_status"] = np.select(
        [
            quotes["book_feed_status"]
            .eq("STALE_BOOK_FEED"),

            quotes["quote_age_minutes"].isna(),

            quotes["quote_age_minutes"] > 60,

            quotes["quote_age_minutes"] > 15,
        ],
        [
            "STALE BOOK â VERIFY",
            "UNKNOWN TIMESTAMP",
            "OLD â VERIFY",
            "AGING â VERIFY",
        ],
        default="FRESH",
    )

    user_quotes = quotes.loc[
        quotes["sportsbook"].isin(
            USER_AVAILABLE_BOOKS
        )
    ].copy()

    core_counts = (
        user_quotes.loc[
            user_quotes["sportsbook"].isin(
                CORE_CONSENSUS_BOOKS
            )
        ]
        .groupby("game_key")["sportsbook"]
        .nunique()
    )

    insufficient = [
        key
        for key in active_game_keys
        if core_counts.get(key, 0) < 3
    ]

    if insufficient:
        raise RuntimeError(
            "Fewer than three core books were available for: "
            + ", ".join(sorted(insufficient))
        )

    metadata = {
        "snapshot_timestamp": snapshot_timestamp,
        "parsed_rows": len(quotes),
        "user_rows": len(user_quotes),
        "games": quotes["game_key"].nunique(),
        "books": quotes["sportsbook"].nunique(),
        "requests_used": response.headers.get(
            "x-requests-used",
            response.headers.get("X-Requests-Used"),
        ),
        "requests_remaining": response.headers.get(
            "x-requests-remaining",
            response.headers.get("X-Requests-Remaining"),
        ),
    }

    return user_quotes, metadata


def rebuild_board(base_board, lines):
    board = base_board.copy()
    lines = normalize_existing_lines(lines)

    recalculated_columns = [
        "market_home_spread",
        "model_side",
        "edge_points",
        "model_market_gap",
        "edge_bucket",
        "best_available_bet",
        "best_book",
        "best_book_edge_points",
        "best_price_american",
        "line_deviation_from_consensus",
        "best_quote_age_minutes",
        "best_quote_topped_up",
        "best_quote_status",
        "consensus_books_available",
        "direct_core_books",
        "topped_up_core_books",
        "shopping_books_available",
        "recommendation_eligibility",
        "dashboard_signal",
    ]

    board = board.drop(
        columns=[
            column
            for column in recalculated_columns
            if column in board.columns
        ],
        errors="ignore",
    )

    active = lines.loc[
        ~lines["sportsbook"].isin(EXCLUDED_BOOKS)
    ].copy()

    core = active.loc[
        active["sportsbook"].isin(
            CORE_CONSENSUS_BOOKS
        )
    ].copy()

    core["direct_core_row"] = (
        ~core["book_topped_up_clean"].fillna(False)
    )

    consensus = (
        core.groupby(
            ["away_team", "home_team"],
            as_index=False,
        )
        .agg(
            market_home_spread=(
                "home_spread",
                "median",
            ),
            consensus_books_available=(
                "sportsbook",
                "nunique",
            ),
            direct_core_books=(
                "direct_core_row",
                "sum",
            ),
            topped_up_core_books=(
                "book_topped_up_clean",
                "sum",
            ),
        )
    )

    board = board.merge(
        consensus,
        on=["away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )

    if board["market_home_spread"].isna().any():
        missing = board.loc[
            board["market_home_spread"].isna(),
            "matchup",
        ].tolist()

        raise RuntimeError(
            "Missing consensus for: "
            + ", ".join(missing)
        )

    board["model_side"] = np.select(
        [
            board["fair_home_spread"]
            < board["market_home_spread"],

            board["fair_home_spread"]
            > board["market_home_spread"],
        ],
        [
            board["home_team"],
            board["away_team"],
        ],
        default="PASS",
    )

    board["edge_points"] = (
        board["fair_home_spread"]
        - board["market_home_spread"]
    ).abs()

    board["model_market_gap"] = (
        board["edge_points"]
    )

    board["edge_bucket"] = pd.cut(
        board["edge_points"],
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["<1", "1â2", "2â3", "3+"],
        right=False,
    ).astype(str)

    shopping = active.merge(
        board[
            [
                "away_team",
                "home_team",
                "fair_home_spread",
                "market_home_spread",
                "model_side",
            ]
        ],
        on=["away_team", "home_team"],
        how="inner",
        validate="many_to_one",
    )

    shopping = shopping.loc[
        shopping["model_side"] != "PASS"
    ].copy()

    shopping["line_deviation_from_consensus"] = (
        shopping["home_spread"]
        - shopping["market_home_spread"]
    ).abs()

    shopping = shopping.loc[
        shopping["line_deviation_from_consensus"] < 1.5
    ].copy()

    shopping["offered_spread"] = np.where(
        shopping["model_side"]
        == shopping["home_team"],
        shopping["home_spread"],
        shopping["away_spread"],
    )

    shopping["offered_price_american"] = np.where(
        shopping["model_side"]
        == shopping["home_team"],
        shopping["home_price_american"],
        shopping["away_price_american"],
    )

    shopping["quote_point_edge"] = np.where(
        shopping["model_side"]
        == shopping["home_team"],
        shopping["home_spread"]
        - shopping["fair_home_spread"],
        shopping["fair_home_spread"]
        - shopping["home_spread"],
    )

    shopping["book_priority"] = np.select(
        [
            shopping["sportsbook"].isin(PRIMARY_BOOKS),
            shopping["sportsbook"].isin(SECONDARY_BOOKS),
        ],
        [1, 2],
        default=3,
    )

    shopping = shopping.sort_values(
        [
            "away_team",
            "home_team",
            "quote_point_edge",
            "offered_price_american",
            "book_priority",
        ],
        ascending=[True, True, False, False, True],
    )

    best = (
        shopping.groupby(
            ["away_team", "home_team"],
            as_index=False,
        )
        .first()
    )

    best["best_available_bet"] = best.apply(
        lambda row: (
            f"{row['model_side']} "
            f"{format_spread(row['offered_spread'])} "
            f"{format_american(row['offered_price_american'])}"
        ),
        axis=1,
    )

    best["best_quote_status"] = np.select(
        [
            (
                best["line_deviation_from_consensus"] >= 1
            )
            & (
                best["quote_age_minutes"].isna()
                | (best["quote_age_minutes"] > 15)
            ),

            best["line_deviation_from_consensus"] >= 1,

            best["quote_age_minutes"].isna()
            | (best["quote_age_minutes"] > 15),
        ],
        [
            "MANUAL_VERIFY_LINE_AND_TIME",
            "MANUAL_VERIFY_LINE",
            "MANUAL_VERIFY_TIMESTAMP",
        ],
        default="LIVE_FEED",
    )

    best_columns = [
        "away_team",
        "home_team",
        "best_available_bet",
        "book_display",
        "quote_point_edge",
        "offered_price_american",
        "line_deviation_from_consensus",
        "quote_age_minutes",
        "book_topped_up_clean",
        "best_quote_status",
    ]

    board = board.merge(
        best[best_columns],
        on=["away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )

    board = board.rename(
        columns={
            "book_display": "best_book",
            "quote_point_edge":
                "best_book_edge_points",
            "offered_price_american":
                "best_price_american",
            "quote_age_minutes":
                "best_quote_age_minutes",
            "book_topped_up_clean":
                "best_quote_topped_up",
        }
    )

    shopping_counts = (
        active.groupby(
            ["away_team", "home_team"]
        )["sportsbook"]
        .nunique()
        .rename("shopping_books_available")
        .reset_index()
    )

    board = board.merge(
        shopping_counts,
        on=["away_team", "home_team"],
        how="left",
        validate="one_to_one",
    )

    board["recommendation_eligibility"] = np.select(
        [
            board["projection_display_status"]
            .eq("EARLY_PROVISIONAL"),

            board["best_quote_status"]
            .astype(str)
            .str.startswith("MANUAL_VERIFY"),
        ],
        [
            "INELIGIBLE_PROVISIONAL",
            "VERIFY_QUOTE",
        ],
        default="MONITOR_ONLY",
    )

    board["dashboard_signal"] = np.select(
        [
            board["projection_display_status"]
            .eq("EARLY_PROVISIONAL"),

            board["best_quote_status"]
            .astype(str)
            .str.startswith("MANUAL_VERIFY"),

            board["edge_points"] >= 2,

            board["edge_points"] >= 1,
        ],
        [
            "PROVISIONAL",
            "VERIFY QUOTE",
            "LARGE DISAGREEMENT",
            "MODERATE DISAGREEMENT",
        ],
        default="SMALL / NO OPINION",
    )

    return board.sort_values(
        "edge_points",
        ascending=False,
    ).reset_index(drop=True)
    def select_pregame_game_keys(
    board,
    schedule=None,
    completed_game_keys=None,
    now_utc=None,
):
    if board.empty:
        return tuple()

    keys = set(
        board["game_key"].dropna().astype(str)
    )

    keys -= {
        str(key)
        for key in (completed_game_keys or set())
    }

    if schedule is None or schedule.empty:
        return tuple(sorted(keys))

    games = schedule.copy()
    games["_game_key"] = games.apply(
        lambda row: (
            f"{int(row['season'])}_W{int(row['week'])}_"
            f"{norm_team(row['away_team'])}_"
            f"{norm_team(row['home_team'])}"
        ),
        axis=1,
    )

    games = games.loc[
        games["_game_key"].isin(keys)
    ].copy()

    if "kickoff_utc" in games.columns:
        kickoffs = pd.to_datetime(
            games["kickoff_utc"],
            errors="coerce",
            utc=True,
        )
    else:
        local_times = pd.to_datetime(
            games["gameday"].astype(str)
            + " "
            + games["gametime"].fillna("00:00").astype(str),
            errors="coerce",
        )
        kickoffs = local_times.dt.tz_localize(
            "America/New_York"
        ).dt.tz_convert("UTC")

    now = pd.Timestamp(
        now_utc
        if now_utc is not None
        else pd.Timestamp.now(tz="UTC")
    )

    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    started = set(
        games.loc[
            kickoffs.le(now),
            "_game_key",
        ]
    )

    return tuple(sorted(keys - started))
