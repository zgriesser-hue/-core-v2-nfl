
import time

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
    ]:
        if candidate.exists():
            return candidate

    return None


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def cached_live_refresh(
    refresh_bucket,
    active_game_keys,
):
    del refresh_bucket

    return fetch_live_lines(
        api_key=st.secrets["PARLAY_API_KEY"],
        season=2026,
        week=1,
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


board_path = locate_file(
    "week_01_live_board.csv"
)

lines_path = locate_file(
    "week_01_book_lines.csv"
)

validation_path = locate_file(
    "model_validation.csv"
)

openers_path = locate_file(
    "week_01_openers.csv"
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

active_game_keys = tuple(
    sorted(
        f"2026_W1_{row.away_team}_{row.home_team}"
        for row in base_board.itertuples()
    )
)

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
        "🔄 Refresh Live Lines — 1 API request",
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
            "Use the refresh button for a live pull."
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
        "Requesting and validating live spreads..."
    ):
        try:
            live_lines, metadata = (
                cached_live_refresh(
                    refresh_bucket,
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

opener_fields = openers[
    [
        "game_key",
        "consensus_open_home_spread",
        "consensus_open_books",
        "earliest_core_open_seen_utc",
        "latest_core_open_seen_utc",
        "consensus_method",
    ]
].copy()

board = board.merge(
    opener_fields,
    on="game_key",
    how="left",
    validate="one_to_one",
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
    "Production version: **Core V2 Week 1**"
)
st.sidebar.write(
    "PVAR personnel layer: **Active**"
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

    st.info(
        "The in-season power-rating table will "
        "connect here after the first ratings update."
    )

    st.write(
        "Current fair lines include team ratings, "
        "quarterbacks, home-field advantage, and "
        "the active PVAR personnel layer."
    )

with tab6:
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
        model and disagreement monitor. During 2026,
        the dashboard will track whether its identified
        sides consistently beat the closing line.
        """
    )
