
import time
from io import BytesIO

CORE_V2_LINE_MOVEMENT_V1 = True
CORE_V2_OPENER_MOVEMENT_V1 = True
from pathlib import Path

import pandas as pd
import streamlit as st

from parlay_live import (
    BOOK_DISPLAY,
    EXCLUDED_BOOKS,
    fetch_live_lines,
    format_american,
    format_spread,
    rebuild_board,
)
from parlay_live import select_pregame_game_keys

st.set_page_config(
    page_title="Core V2 NFL Dashboard",
    page_icon="🏈",
    layout="wide",
)


ROOT = Path(__file__).parent


def locate_file(filename):
    for candidate in [
        ROOT / filename,
        ROOT / "data" / filename,
        ROOT / "data" / "model" / filename,
    ]:
        if candidate.exists():
            return candidate

    return None


@st.cache_data
def load_csv_bytes(file_bytes):
    return pd.read_csv(BytesIO(file_bytes))


def load_csv(path_string):
    # Cache by file contents, not only by an unchanged path. GitHub automation
    # replaces CSVs in place, so this automatically invalidates stale data.
    return load_csv_bytes(
        Path(path_string).read_bytes()
    )


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def cached_live_refresh(
    refresh_bucket,
    season,
    week,
    active_game_keys,
):
    del refresh_bucket

    return fetch_live_lines(
        api_key=st.secrets["PARLAY_API_KEY"],
        season=season,
        week=week,
        active_game_keys=set(active_game_keys),
    )


def display_home_line(
    away_team,
    home_team,
    home_spread,
):
    if pd.isna(home_spread):
        return "—"

    home_spread = float(home_spread)

    if abs(home_spread) < 0.001:
        return "Pick'em"

    if home_spread < 0:
        return (
            f"{home_team} "
            f"{format_spread(home_spread)}"
        )

    return (
        f"{away_team} "
        f"{format_spread(-home_spread)}"
    )


def parse_timestamp(value):
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        return None

    return timestamp.tz_convert(
        "America/New_York"
    )


def display_timestamp(value):
    timestamp = parse_timestamp(value)

    if timestamp is None:
        return "—"

    return timestamp.strftime(
        "%b %-d, %-I:%M %p ET"
    )


def quote_icon(status):
    status = str(status)

    if "STALE" in status or "OLD" in status:
        return f"🔴 {status}"

    if "VERIFY" in status or "AGING" in status:
        return f"🟡 {status}"

    return f"🟢 {status}"


def signal_icon(signal):
    return {
        "PROVISIONAL": "🟠 Provisional",
        "VERIFY QUOTE": "🟡 Verify quote",
        "LARGE DISAGREEMENT":
            "🔵 Large disagreement",
        "MODERATE DISAGREEMENT":
            "🔹 Moderate disagreement",
        "SMALL / NO OPINION":
            "⚪ Small / no opinion",
    }.get(str(signal), str(signal))


board_path = locate_file("current_live_board.csv") or locate_file(
    "week_01_live_board.csv"
)

lines_path = locate_file("current_book_lines.csv") or locate_file(
    "week_01_book_lines.csv"
)

validation_path = locate_file(
    "model_validation.csv"
)

openers_path = locate_file("current_openers.csv") or locate_file(
    "week_01_openers.csv"
)

grades_path = locate_file("game_grades.csv")

ratings_path = locate_file(
    "team_ratings_2026.csv"
)

missing = []

if board_path is None:
    missing.append("week_01_live_board.csv")

if lines_path is None:
    missing.append("week_01_book_lines.csv")

if validation_path is None:
    missing.append("model_validation.csv")

if openers_path is None:
    missing.append("week_01_openers.csv")

if ratings_path is None:
    missing.append("team_ratings_2026.csv")

if missing:
    st.error(
        "Missing required files: "
        + ", ".join(missing)
    )
    st.stop()

base_board = load_csv(
    str(board_path)
)

static_lines = load_csv(
    str(lines_path)
)

validation = load_csv(
    str(validation_path)
)

openers = load_csv(
    str(openers_path)
)

game_grades = (
    load_csv(str(grades_path))
    if grades_path is not None
    else pd.DataFrame()
)

power_ratings = load_csv(str(ratings_path))

active_season = int(base_board["season"].iloc[0]) if "season" in base_board else 2026
active_week = int(base_board["week"].iloc[0])

completed_game_keys = set(
    game_grades.get(
        "game_key",
        pd.Series(dtype=str),
    )
    .dropna()
    .astype(str)
)

schedule = pd.read_csv(
    ROOT / "data/schedules/schedule_2026.csv",
    low_memory=False,
)

active_game_keys = select_pregame_game_keys(
    base_board,
    schedule=schedule,
    completed_game_keys=completed_game_keys,
)

base_board = base_board.loc[
    base_board["game_key"]
    .astype(str)
    .isin(active_game_keys)
].copy()


if "live_lines" not in st.session_state:
    st.session_state.live_lines = None

if "live_metadata" not in st.session_state:
    st.session_state.live_metadata = None

if "refresh_error" not in st.session_state:
    st.session_state.refresh_error = None

if "previous_lines" not in st.session_state:
    st.session_state.previous_lines = None

if "previous_metadata" not in st.session_state:
    st.session_state.previous_metadata = None

has_api_key = (
    "PARLAY_API_KEY" in st.secrets
    and bool(
        str(
            st.secrets["PARLAY_API_KEY"]
        ).strip()
    )
)

st.title("🏈 Core V2 NFL Dashboard")
st.caption(
    "Independent fair lines, market comparison, "
    "sportsbook shopping, and weekly tracking"
)

st.info(
    "Core V2 has not yet demonstrated a reliable "
    "historical edge over the spread market. "
    "Disagreements are monitoring signals, not "
    "automatic bet recommendations."
)

refresh_col, status_col = st.columns(
    [1, 3]
)

with refresh_col:
    refresh_clicked = st.button(
        "🔄 Refresh Live Spreads + Totals — 2 API credits",
        type="primary",
        disabled=not has_api_key,
        use_container_width=True,
        help=(
            "Results are cached for 15 minutes. "
            "Repeated clicks during the cache window "
            "do not make another request."
        ),
    )

with status_col:
    if st.session_state.live_metadata:
        metadata = st.session_state.live_metadata

        st.success(
            "Live lines loaded: "
            + display_timestamp(
                metadata.get(
                    "snapshot_timestamp"
                )
            )
        )
    else:
        st.caption(
            "Displaying the repository snapshot. "
            "Scheduled snapshots are saved three times daily; "
            "the refresh button is session-only."
        )

if not has_api_key:
    st.warning(
        "Live refresh is disabled until PARLAY_API_KEY "
        "is added to the Streamlit app's Secrets."
    )

if refresh_clicked:
    refresh_bucket = int(
        time.time() // 900
    )

    with st.spinner(
        "Requesting and validating live spreads and totals..."
    ):
        try:
            live_lines, metadata = (
                cached_live_refresh(
                    refresh_bucket,
                    active_season,
                    active_week,
                    active_game_keys,
                )
            )

            lines_before_refresh = (
                st.session_state.live_lines.copy()
                if st.session_state.live_lines is not None
                else static_lines.copy()
            )

            if (
                "market_snapshot_timestamp_et"
                in lines_before_refresh.columns
                and not lines_before_refresh.empty
            ):
                previous_snapshot = (
                    lines_before_refresh[
                        "market_snapshot_timestamp_et"
                    ].iloc[0]
                )
            else:
                previous_snapshot = None

            new_snapshot = metadata.get(
                "snapshot_timestamp"
            )

            previous_snapshot_parsed = pd.to_datetime(
                previous_snapshot,
                errors="coerce",
                utc=True,
            )

            new_snapshot_parsed = pd.to_datetime(
                new_snapshot,
                errors="coerce",
                utc=True,
            )

            is_new_snapshot = (
                pd.isna(previous_snapshot_parsed)
                or pd.isna(new_snapshot_parsed)
                or previous_snapshot_parsed
                != new_snapshot_parsed
            )

            if is_new_snapshot:
                st.session_state.previous_lines = (
                    lines_before_refresh
                )

                st.session_state.previous_metadata = {
                    "snapshot_timestamp":
                        previous_snapshot
                }

            st.session_state.live_lines = (
                live_lines
            )

            st.session_state.live_metadata = (
                metadata
            )

            st.session_state.refresh_error = None

        except Exception as exc:
            st.session_state.refresh_error = (
                f"{type(exc).__name__}: {exc}"
            )

if st.session_state.refresh_error:
    st.error(
        "Live refresh failed validation. "
        "The saved repository snapshot remains active.\n\n"
        + st.session_state.refresh_error
    )

current_lines = (
    st.session_state.live_lines
    if st.session_state.live_lines is not None
    else static_lines
)

try:
    board = rebuild_board(
        base_board,
        current_lines,
    )
except Exception as exc:
    st.error(
        "The current market data could not produce "
        f"a valid board: {type(exc).__name__}: {exc}"
    )
    st.stop()

# --------------------------------------------------------------------------------------------------
# Permanent movement from the market opener
# --------------------------------------------------------------------------------------------------

for column in ["consensus_open_total", "consensus_open_total_books"]:
    if column not in openers.columns:
        openers[column] = pd.NA

opener_fields = openers[
    [
        "away_team",
        "home_team",
        "consensus_open_home_spread",
        "consensus_open_books",
        "consensus_open_total",
        "consensus_open_total_books",
        "earliest_core_open_seen_utc",
        "latest_core_open_seen_utc",
        "consensus_method",
    ]
].copy()

opener_fields = opener_fields.drop_duplicates(
    subset=[
        "away_team",
        "home_team",
    ],
    keep="last",
)

board_rows_before_opener_merge = len(board)

board = board.merge(
    opener_fields,
    on=[
        "away_team",
        "home_team",
    ],
    how="left",
    validate="one_to_one",
)

assert len(board) == board_rows_before_opener_merge, (
    "Opener merge unexpectedly changed the number "
    "of dashboard games."
)

if board[
    "consensus_open_home_spread"
].isna().any():
    missing_openers = board.loc[
        board[
            "consensus_open_home_spread"
        ].isna(),
        "game_key",
    ].tolist()

    st.error(
        "Missing consensus openers for: "
        + ", ".join(missing_openers)
    )
    st.stop()

board["open_to_current_move"] = (
    board["market_home_spread"]
    -
    board["consensus_open_home_spread"]
)

board["total_move_from_open"] = (
    board["market_total"] - board["consensus_open_total"]
)

board["open_distance_from_model"] = (
    board["consensus_open_home_spread"]
    -
    board["fair_home_spread"]
).abs()

board["current_distance_from_model"] = (
    board["market_home_spread"]
    -
    board["fair_home_spread"]
).abs()


def opener_move_direction(row):
    move = row["open_to_current_move"]

    if pd.isna(move):
        return "—"

    if abs(float(move)) < 0.001:
        return "Unchanged"

    if float(move) < 0:
        return f"Toward {row['home_team']}"

    return f"Toward {row['away_team']}"


def model_alignment_label(row):
    opening_distance = row[
        "open_distance_from_model"
    ]

    current_distance = row[
        "current_distance_from_model"
    ]

    if (
        pd.isna(opening_distance)
        or pd.isna(current_distance)
    ):
        return "—"

    difference = (
        float(current_distance)
        -
        float(opening_distance)
    )

    if abs(difference) < 0.001:
        return "Same distance"

    if difference < 0:
        return "Toward Core V2"

    return "Away from Core V2"


board["opener_move_direction"] = (
    board.apply(
        opener_move_direction,
        axis=1,
    )
)

board["market_vs_model_path"] = (
    board.apply(
        model_alignment_label,
        axis=1,
    )
)


# --------------------------------------------------------------------------------------------------
# Line movement since the prior displayed snapshot
# --------------------------------------------------------------------------------------------------

movement = pd.DataFrame()

if st.session_state.previous_lines is not None:
    try:
        previous_board = rebuild_board(
            base_board,
            st.session_state.previous_lines,
        )

        movement = (
            board[
                [
                    "matchup",
                    "away_team",
                    "home_team",
                    "market_home_spread",
                ]
            ]
            .merge(
                previous_board[
                    [
                        "matchup",
                        "market_home_spread",
                    ]
                ].rename(
                    columns={
                        "market_home_spread":
                            "previous_market_home_spread"
                    }
                ),
                on="matchup",
                how="left",
                validate="one_to_one",
            )
        )

        movement["market_move"] = (
            movement["market_home_spread"]
            -
            movement[
                "previous_market_home_spread"
            ]
        )

        def movement_direction(row):
            move = row["market_move"]

            if pd.isna(move):
                return "—"

            if abs(float(move)) < 0.001:
                return "Unchanged"

            if float(move) < 0:
                return f"Toward {row['home_team']}"

            return f"Toward {row['away_team']}"

        movement["movement_direction"] = (
            movement.apply(
                movement_direction,
                axis=1,
            )
        )

    except Exception as exc:
        st.warning(
            "Previous lines could not be compared: "
            f"{type(exc).__name__}: {exc}"
        )

if st.session_state.live_metadata:
    metadata = st.session_state.live_metadata

    meta1, meta2, meta3, meta4 = st.columns(4)

    meta1.metric(
        "Live games",
        metadata.get("games", "—"),
    )

    meta2.metric(
        "Live books",
        metadata.get("books", "—"),
    )

    meta3.metric(
        "API requests used",
        metadata.get("requests_used", "—"),
    )

    meta4.metric(
        "API requests remaining",
        metadata.get(
            "requests_remaining",
            "—",
        ),
    )

market_timestamp = (
    current_lines[
        "market_snapshot_timestamp_et"
    ].iloc[0]
    if (
        "market_snapshot_timestamp_et"
        in current_lines.columns
        and not current_lines.empty
    )
    else None
)

# --------------------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------------------

st.sidebar.header("Core V2 Controls")

minimum_edge = st.sidebar.slider(
    "Minimum model disagreement",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.25,
)

signals = (
    board["dashboard_signal"]
    .dropna()
    .unique()
    .tolist()
)

selected_signals = st.sidebar.multiselect(
    "Signal",
    options=signals,
    default=signals,
)

show_provisional = st.sidebar.checkbox(
    "Show provisional projections",
    value=True,
)

show_stale_betmgm = st.sidebar.checkbox(
    "Show stale BetMGM API quotes",
    value=False,
)

st.sidebar.divider()
st.sidebar.write(
    f"Production version: **Core V2 {active_season} Week {active_week}**"
)
st.sidebar.write(
    "PVAR personnel layer: **Shadow validation**"
)
st.sidebar.write(
    "Market used inside model: **No**"
)
st.sidebar.write(
    "BetMGM API feed: **Excluded**"
)
st.sidebar.write(
    "Betting status: **Monitoring / CLV tracking**"
)
st.sidebar.caption(
    "Market snapshot: "
    + display_timestamp(market_timestamp)
)

# --------------------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Market Board",
        "Game Detail",
        "Sportsbook Lines",
        "Line Movement",
        "Power Ratings",
        "Model Performance",
    ]
)

with tab1:
    filtered = board.loc[
        board["edge_points"] >= minimum_edge
    ].copy()

    if selected_signals:
        filtered = filtered.loc[
            filtered["dashboard_signal"].isin(
                selected_signals
            )
        ]
    else:
        filtered = filtered.iloc[0:0]

    if not show_provisional:
        filtered = filtered.loc[
            filtered[
                "projection_display_status"
            ] != "EARLY_PROVISIONAL"
        ]

    if not movement.empty:
        filtered = filtered.merge(
            movement[
                [
  
                  "matchup",
                    "previous_market_home_spread",
                    "market_move",
                    "movement_direction",
                ]
            ],
            on="matchup",
            how="left",
            validate="one_to_one",
        )
    else:
        filtered["previous_market_home_spread"] = pd.NA
        filtered["market_move"] = pd.NA
        filtered["movement_direction"] = "—"

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Games", len(board))
    m2.metric(
        "Disagreements ≥1",
        int(
            (board["edge_points"] >= 1).sum()
        ),
    )
    m3.metric(
        "Quotes to verify",
        int(
            board["best_quote_status"]
            .astype(str)
            .str.startswith("MANUAL_VERIFY")
            .sum()
        ),
    )
    m4.metric(
        "Provisional",
        int(
            board[
                "projection_display_status"
            ]
            .eq("EARLY_PROVISIONAL")
            .sum()
        ),
    )

    display = pd.DataFrame(
        {
            "Matchup": filtered["matchup"],
            "Core V2 fair": filtered.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row["fair_home_spread"],
                ),
                axis=1,
            ),
            "Opener": filtered.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row[
                        "consensus_open_home_spread"
                    ],
                ),
                axis=1,
            ),
            "Market": filtered.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row["market_home_spread"],
                ),
                axis=1,
            ),
            "Total": filtered["market_total"].map(
                lambda value: f"{float(value):.1f}" if pd.notna(value) else "—"
            ),
            "Total move": filtered["total_move_from_open"].map(
                lambda value: f"{float(value):+.1f}" if pd.notna(value) else "—"
            ),
            "From open": filtered[
                "open_to_current_move"
            ].map(
                lambda value: (
                    f"{float(value):+.1f}"
                    if pd.notna(value)
                    else "—"
                )
            ),
            "Open direction": filtered[
                "opener_move_direction"
            ],
            "Previous": filtered.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row[
                        "previous_market_home_spread"
                    ],
                ),
                axis=1,
            ),
            "Move": filtered["market_move"].map(
                lambda value: (
                    f"{float(value):+.1f}"
                    if pd.notna(value)
                    else "—"
                )
            ),
            "Direction": filtered[
                "movement_direction"
            ],
            "Lean": filtered["model_side"],
            "Gap": filtered[
                "edge_points"
            ].round(2),
            "Best quote": filtered[
                "best_available_bet"
            ],
            "Book": filtered["best_book"],
            "Age": filtered[
                "best_quote_age_minutes"
            ].map(
                lambda value: (
                    f"{value:.0f}m"
                    if pd.notna(value)
                    else "—"
                )
            ),
            "Quote check": filtered[
                "best_quote_status"
            ].map(quote_icon),
            "Signal": filtered[
                "dashboard_signal"
            ].map(signal_icon),
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Gap measures disagreement with the "
        "four-book market consensus. It is not "
        "an expected-value estimate."
    )

with tab2:
    selected_game = st.selectbox(
        "Game",
        board["matchup"].tolist(),
        key="detail_game",
    )

    row = board.loc[
        board["matchup"] == selected_game
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Core V2 Fair Line",
        display_home_line(
            row["away_team"],
            row["home_team"],
            row["fair_home_spread"],
        ),
    )

    c1.metric(
        "Market Consensus",
        display_home_line(
            row["away_team"],
            row["home_team"],
            row["market_home_spread"],
        ),
    )

    c2.metric(
        "Model Disagreement",
        f'{row["edge_points"]:.2f} pts',
    )

    c2.metric(
        "Model Lean",
        row["model_side"],
    )

    c3.metric(
        "Best Quoted Line",
        row["best_available_bet"],
    )

    c3.metric(
        "Sportsbook",
        row["best_book"],
    )

    c3.metric(
        "Market Total",
        f'{float(row["market_total"]):.1f}'
        if pd.notna(row["market_total"])
        else "—",
    )

    st.divider()

    o1, o2, o3 = st.columns(3)

    o1.metric(
        "Market Opener",
        display_home_line(
            row["away_team"],
            row["home_team"],
            row["consensus_open_home_spread"],
        ),
    )

    o2.metric(
        "Movement From Open",
        f'{row["open_to_current_move"]:+.1f} pts',
    )

    o3.metric(
        "Market Path",
        row["market_vs_model_path"],
    )

    st.caption(
        "Total opener: "
        + (
            f'{float(row["consensus_open_total"]):.1f}; '
            f'move {float(row["total_move_from_open"]):+.1f} points.'
            if pd.notna(row["consensus_open_total"])
            and pd.notna(row["total_move_from_open"])
            else "not yet captured."
        )
    )

    st.caption(
        "Opening consensus based on "
        f'{int(row["consensus_open_books"])} books. '
        "Comparison with Core V2 is descriptive because "
        "the current model projection may postdate the opener."
    )

    if str(
        row["best_quote_status"]
    ).startswith("MANUAL_VERIFY"):
        st.warning(

            "Verify this line directly in the "
            "sportsbook before using it."
        )

    if (
        row["projection_display_status"]
        == "EARLY_PROVISIONAL"
    ):
        st.warning(
            "This projection is provisional because "
            "personnel information is incomplete."
        )

    st.write(
        {
            "Personnel status": row.get(
                "personnel_projection_status",
                "—",
            ),
            "Personnel complete": row.get(
                "personnel_info_complete",
                "—",
            ),
            "Personnel fallback used": row.get(
                "used_personnel_fallback",
                "—",
            ),
            "Core books": row.get(
                "consensus_books_available",
                "—",
            ),
            "Shopping books": row.get(
                "shopping_books_available",
                "—",
            ),
        }
    )

with tab3:
    selected_line_game = st.selectbox(
        "Matchup",
        board["matchup"].tolist(),
        key="line_game",
    )

    game_lines = current_lines.loc[
        current_lines["matchup"]
        == selected_line_game
    ].copy()

    if not show_stale_betmgm:
        game_lines = game_lines.loc[
            ~game_lines["sportsbook"].isin(
                EXCLUDED_BOOKS
            )
        ]

    game_lines["Away line"] = (
        game_lines.apply(
            lambda row: (
                f"{row['away_team']} "
                f"{format_spread(row['away_spread'])} "
                f"{format_american(row['away_price_american'])}"
            ),
            axis=1,
        )
    )

    game_lines["Home line"] = (
        game_lines.apply(
            lambda row: (
                f"{row['home_team']} "
                f"{format_spread(row['home_spread'])} "
                f"{format_american(row['home_price_american'])}"
            ),
            axis=1,
        )
    )

    game_lines["Age"] = game_lines[
        "quote_age_minutes"
    ].map(
        lambda value: (
            f"{value:.0f}m"
            if pd.notna(value)
            else "—"
        )
    )

    game_lines["Status"] = game_lines[
        "quote_status"
    ].map(quote_icon)

    line_table = game_lines[
        [
            "book_display",
            "Away line",
            "Home line",
            "Age",
            "Status",
        ]
    ].rename(
        columns={
            "book_display": "Sportsbook"
        }
    )

    st.dataframe(
        line_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Always confirm the final line and price "
        "inside the sportsbook."
    )

with tab4:
    st.subheader("Movement From Market Opener")

    opener_moves = board.copy()

    games_moved_from_open = int(
        opener_moves["open_to_current_move"]
        .abs()
        .gt(0.001)
        .sum()
    )

    largest_opener_move = (
        opener_moves["open_to_current_move"]
        .abs()
        .max()
    )

    moved_toward_model = int(
        opener_moves["market_vs_model_path"]
        .eq("Toward Core V2")
        .sum()
    )

    om1, om2, om3 = st.columns(3)

    om1.metric(
        "Games Moved",
        games_moved_from_open,
    )

    om2.metric(
        "Largest Move",
        (
            f"{largest_opener_move:.1f} pts"
            if pd.notna(largest_opener_move)
            else "—"
        ),
    )

    om3.metric(
        "Moved Toward Core V2",
        moved_toward_model,
    )

    opener_moves["Opener"] = (
        opener_moves.apply(
            lambda row: display_home_line(
                row["away_team"],
                row["home_team"],
                row[
                    "consensus_open_home_spread"
                ],
            ),
            axis=1,
        )
    )

    opener_moves["Current"] = (
        opener_moves.apply(
            lambda row: display_home_line(
                row["away_team"],
                row["home_team"],
                row["market_home_spread"],
            ),
            axis=1,
        )
    )

    opener_moves["Move"] = (
        opener_moves[
            "open_to_current_move"
        ].map(
            lambda value: (
                f"{float(value):+.1f}"
                if pd.notna(value)
                else "—"
            )
        )
    )

    opener_moves["Absolute move"] = (
        opener_moves[
            "open_to_current_move"
        ].abs()
    )

    opener_moves = opener_moves.sort_values(
        [
            "Absolute move",
            "matchup",
        ],
        ascending=[
            False,
            True,
        ],
    )

    st.dataframe(
        opener_moves[
            [
                "matchup",
                "Opener",
                "Current",
                "Move",
                "opener_move_direction",
                "market_vs_model_path",
            ]
        ].rename(
            columns={
                "matchup": "Matchup",
                "opener_move_direction": "Direction",
                "market_vs_model_path": "Vs Core V2",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Negative movement is toward the home team; "
        "positive movement is toward the away team. "
        "'Toward Core V2' is descriptive—not evidence "
        "that the model predicted the move."
    )

    st.divider()
    st.subheader("Movement Since Previous Refresh")

    previous_timestamp = None

    if st.session_state.previous_metadata:
        previous_timestamp = (
            st.session_state.previous_metadata.get(
                "snapshot_timestamp"
            )
        )

    current_timestamp = (
        st.session_state.live_metadata.get(
            "snapshot_timestamp"
        )
        if st.session_state.live_metadata
        else market_timestamp
    )

    if movement.empty:
        st.info(
            "No comparison snapshot is available yet. "
            "Refresh the live lines to compare them "
            "with the currently displayed repository snapshot."
        )
    else:
        changed_games = int(
            movement["market_move"]
            .abs()
            .gt(0.001)
            .sum()
        )

        largest_move = (
            movement["market_move"]
            .abs()
            .max()
        )

        lm1, lm2, lm3 = st.columns(3)

        lm1.metric(
            "Games Moved",
            changed_games,
        )

        lm2.metric(
            "Largest Move",
            (
                f"{largest_move:.1f} pts"
                if pd.notna(largest_move)
                else "—"
            ),
        )

        lm3.metric(
            "Games Compared",
            len(movement),
        )

        st.caption(
            "Previous: "
            + display_timestamp(previous_timestamp)
            + "  |  Current: "
            + display_timestamp(current_timestamp)
        )

        movement_display = movement.copy()

        movement_display["Previous"] = (
            movement_display.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row[
                        "previous_market_home_spread"
                    ],
                ),
                axis=1,
            )
        )

        movement_display["Current"] = (
            movement_display.apply(
                lambda row: display_home_line(
                    row["away_team"],
                    row["home_team"],
                    row["market_home_spread"],
                ),
                axis=1,
            )
        )

        movement_display["Move"] = (
            movement_display["market_move"].map(
                lambda value: (
                    f"{float(value):+.1f}"
                    if pd.notna(value)
                    else "—"
                )
            )
        )

        movement_display["Absolute move"] = (
            movement_display["market_move"].abs()
        )

        movement_display = (
            movement_display
            .sort_values(
                [
                    "Absolute move",
                    "matchup",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        st.dataframe(
            movement_display[
                [
                    "matchup",
                    "Previous",
                    "Current",
                    "Move",
                    "movement_direction",
                ]
            ].rename(
                columns={
                    "matchup": "Matchup",
                    "movement_direction": "Direction",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Movement is the change in the home-team "
            "spread. Negative movement is toward the "
            "home team; positive movement is toward "
            "the away team."
        )


with tab5:
    st.subheader("Power Ratings")

    ratings_display = power_ratings.copy()
    numeric_columns = [
        "rank",
        "current_team_rating",
        "rating_before_performance",
        "performance_adjustment",
        "qb_adjustment",
        "manual_personnel_adjustment",
        "manual_injury_adjustment",
        "games_in_performance_update",
        "performance_update_through_week",
        "performance_complete_through_week",
        "latest_week_games_graded",
        "latest_week_games_scheduled",
    ]
    for column in numeric_columns:
        if column in ratings_display.columns:
            ratings_display[column] = pd.to_numeric(
                ratings_display[column], errors="coerce"
            )

    update_week = int(
        ratings_display.get(
            "performance_update_through_week",
            pd.Series([0]),
        ).max()
    )
    update_status = str(
        ratings_display.get(
            "performance_update_status",
            pd.Series(["PRESEASON"]),
        ).iloc[0]
    )
    games_graded = int(
        ratings_display.get("latest_week_games_graded", pd.Series([0])).max()
    )
    games_scheduled = int(
        ratings_display.get("latest_week_games_scheduled", pd.Series([0])).max()
    )
    update_label = f"Week {update_week}" if update_week else "Preseason"
    if update_status == "PARTIAL":
        update_label += f" partial ({games_graded}/{games_scheduled})"
    top_team = ratings_display.sort_values(
        "current_team_rating", ascending=False
    ).iloc[0]
    performance_change = ratings_display.get(
        "performance_adjustment",
        pd.Series(0.0, index=ratings_display.index),
    ).fillna(0.0)

    pr1, pr2, pr3, pr4 = st.columns(4)
    pr1.metric("No. 1 Team", top_team["team"])
    pr2.metric("Top Rating", f'{top_team["current_team_rating"]:+.2f}')
    pr3.metric("Updated Through", update_label)
    pr4.metric("Largest Weekly Move", f'{performance_change.abs().max():.2f} pts')

    table = pd.DataFrame({
        "Rank": ratings_display["rank"].astype("Int64"),
        "Team": ratings_display.get("team_name", ratings_display["team"]),
        "Rating": ratings_display["current_team_rating"].round(2),
        "Performance Δ": performance_change.round(2),
        "Before performance": ratings_display.get(
            "rating_before_performance",
            ratings_display["current_team_rating"],
        ).round(2),
        "QB Δ": ratings_display.get(
            "qb_adjustment", pd.Series(0.0, index=ratings_display.index)
        ).round(2),
        "Manual personnel Δ": ratings_display.get(
            "manual_personnel_adjustment",
            pd.Series(0.0, index=ratings_display.index),
        ).round(2),
        "Manual injury Δ": ratings_display.get(
            "manual_injury_adjustment",
            pd.Series(0.0, index=ratings_display.index),
        ).round(2),
    }).sort_values(["Rank", "Team"])

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "Performance updates are zero-sum and opponent-adjusted: each team's "
        "final-margin residual versus the released Core V2 projection is "
        "weighted 10%, split between the teams, and capped at 0.75 points per "
        "team per game. Game-specific PVAR remains in shadow validation."
    )
    if update_status == "PARTIAL":
        st.info(
            "Monday ratings are provisional and include completed Sunday games. "
            "They will be recomputed Tuesday after Monday Night Football."
        )

with tab6:
    st.subheader("Historical Validation")

    result = validation.iloc[0]

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Games Tested",
        int(result["games"]),
    )

    p2.metric(
        "Core V2 PVAR MAE",
        f'{result["pvar_model_mae"]:.2f}',
    )

    p3.metric(
        "Market MAE",
        f'{result["market_mae"]:.2f}',
    )

    p4.metric(
        "Model-Side Cover Rate",
        f'{100 * result["model_side_cover_rate"]:.1f}%',
    )

    st.warning(
        str(result["betting_status"])
    )

    st.write(
        {
            "ATS wins": int(
                result["model_side_wins"]
            ),
            "ATS losses": int(
                result["model_side_losses"]
            ),
            "ATS pushes": int(
                result["model_side_pushes"]
            ),
            "ROI at -110": (
                f'{100 * result["roi_at_minus_110"]:.1f}%'
            ),
            "Best historical blend": (
                f'{100 * result["best_market_weight"]:.0f}% market / '
                f'{100 * result["best_model_weight"]:.0f}% model'
            ),
        }
    )

    st.markdown(
        """
        Core V2 is currently an independent football
        model and disagreement monitor. During the season,
        the dashboard will track whether its identified
        sides consistently beat the closing line.
        """
    )

    st.divider()
    st.subheader("2026 Live Tracking")

    if game_grades.empty:
        st.info(
            "Live grades will appear after a final score is available. "
            "CLV appears only when an automated pre-kickoff close was captured."
        )
    else:
        live = game_grades.copy()
        for column in [
            "projection_abs_error",
            "clv_points",
            "bet_clv_points",
            "projection_vs_close_points",
            "release_bet_spread",
            "closing_bet_spread",
            "away_score",
            "home_score",
        ]:
            if column in live.columns:
                live[column] = pd.to_numeric(live[column], errors="coerce")

        graded_picks = live.loc[
            live["ats_result_at_release"].isin(["Win", "Loss", "Push"])
        ].copy()
        wins = int(graded_picks["ats_result_at_release"].eq("Win").sum())
        losses = int(graded_picks["ats_result_at_release"].eq("Loss").sum())
        pushes = int(graded_picks["ats_result_at_release"].eq("Push").sum())
        decisions = wins + losses
        cover_rate = wins / decisions if decisions else float("nan")

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Games Graded", len(live))
        l2.metric("Live Projection MAE", f'{live["projection_abs_error"].mean():.2f}')
        l3.metric(
            "Release ATS",
            f"{wins}-{losses}-{pushes}",
            f"{100 * cover_rate:.1f}%" if pd.notna(cover_rate) else "—",
        )
        l4.metric(
            "Avg Projection vs Close",
            (
                f'{graded_picks["projection_vs_close_points"].mean():+.2f} pts'
                if (
                    "projection_vs_close_points" in graded_picks
                    and graded_picks["projection_vs_close_points"].notna().any()
                )
                else "—"
            ),
        )

        tracking_table = pd.DataFrame(
            {
                "Week": live["week"].astype(int),
                "Matchup": live["away_team"] + " @ " + live["home_team"],
                "Final": live.apply(
                    lambda row: (
                        f"{row['away_team']} {int(row['away_score'])} – "
                        f"{row['home_team']} {int(row['home_score'])}"
                        if pd.notna(row["away_score"]) and pd.notna(row["home_score"])
                        else "—"
                    ),
                    axis=1,
                ),
                "Model side": live["model_selected_team"],
                "Release line": live["release_bet_spread"].map(
                    lambda value: format_spread(value) if pd.notna(value) else "PASS"
                ),
                "Closing line": live["closing_bet_spread"].map(
                    lambda value: format_spread(value) if pd.notna(value) else "—"
                ),
                "Close status": (
                    live["closing_status"].map(
                        {
                            "CAPTURED": "Captured",
                            "MISSING_NOT_BACKFILLED": "Not captured",
                        }
                    ).fillna("—")
                    if "closing_status" in live.columns
                    else "—"
                ),
                "Projection vs close": live.get(
                    "projection_vs_close_points",
                    pd.Series(float("nan"), index=live.index),
                ).map(
                    lambda value: f"{value:+.1f}" if pd.notna(value) else "—"
                ),
                "Bet CLV": live.get(
                    "bet_clv_points",
                    live["clv_points"],
                ).map(
                    lambda value: f"{value:+.1f}" if pd.notna(value) else "—"
                ),
                "ATS": live["ats_result_at_release"],
                "Projection error": live["projection_abs_error"].map(
                    lambda value: f"{value:.1f}" if pd.notna(value) else "—"
                ),
            }
        )
        st.dataframe(
            tracking_table.sort_values(["Week", "Matchup"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Projection vs close compares the released Core V2 fair line with "
            "the automated pre-kickoff consensus in the model side's direction. "
            "Bet CLV separately compares the available release bet line with the close. "
            "Positive is favorable for either metric. "
            "A missed close is left blank and is never reconstructed from an earlier quote."
        )
