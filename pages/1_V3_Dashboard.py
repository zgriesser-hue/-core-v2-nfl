"""
app_v3.py - Streamlit page for the V3 model.

DEPLOYMENT (Option B - single app, two pages): this file goes in a
`pages/` folder next to app.py, e.g.
`streamlit_export/pages/1_V3_Dashboard.py`. Streamlit's built-in
multi-page support then adds it automatically as a second sidebar
entry in the SAME app.py deployment - one URL, one Streamlit Cloud
app, no changes to app.py itself. The `1_` prefix controls its
position in the sidebar (app.py's own page is always first); rename
the file to change that, no code change needed.

TWO THINGS THAT ONLY MATTER BECAUSE THIS RUNS ALONGSIDE app.py, NOT
STANDALONE:
1. ROOT below is `Path(__file__).parent.parent`, not `.parent` - this
   file lives one level deeper (in pages/) than app.py does, but
   needs to resolve the SAME repo-root paths (board CSVs, data/, etc.)
   app.py resolves from its own location. Get this wrong and every
   locate_file call silently looks in pages/ instead of the repo root.
2. st.session_state is shared across ALL pages in a multi-page app.
   Every key this file uses is prefixed `v3_` (`v3_live_lines`,
   `v3_live_metadata`, `v3_refresh_error`) specifically so it can
   never collide with app.py's own `live_lines`/`live_metadata`/
   `refresh_error` keys - without the prefix, refreshing one page's
   odds would silently overwrite the other's cached state depending
   on click order, a bug that only appears once both pages are
   actually running together.

Mirrors your existing Core V2 app.py pattern otherwise: the model
runs offline (board.py, run after each week's games complete) and
writes a CSV; this page displays it and compares it to a live pull
via your existing parlay_live.py. It never runs model code itself.

Live odds refresh reuses parlay_live.fetch_live_lines / rebuild_board
and pregame_filter.select_pregame_game_keys UNCHANGED - all three
were verified to contain zero V2-model-specific logic (they operate
on away_team/home_team/fair_home_spread/game_key, not on how
fair_home_spread was computed). The only V3-specific work is a small
adapter (`board_for_rebuild`) that renames this board's
`fair_home_spread_v3` to the generic `fair_home_spread` column
rebuild_board expects, and adds a placeholder
`projection_display_status` column (V3 doesn't have this V2 concept
yet) so rebuild_board's dashboard_signal logic doesn't KeyError.

STAGE 5: "My Bets" tab, reusing core_v2.bet_tracking UNCHANGED (it's
genuinely model-agnostic - just needs fair_home_spread in bookmaker
convention, game_key/matchup, and odds). Points at its own bet
ledger (data/tracking/my_bets_v3.csv) and Stage 4's grades file
(data/tracking/v3_game_grades.csv or the dashboard copy
v3_game_grades.csv next to the board), never V2's my_bets.csv /
game_grades.csv. Every entry uses fair_display/market_display (see
below), NOT fair_home_spread_v3/market_home_spread directly - those
two are margin convention; bet_tracking, like the rest of Core V2's
stack, needs bookmaker convention.

IMPORTANT - sign convention (read this before touching bet entry
logic): fair_home_spread_v3 is margin convention (positive = home
favored, matching pred_v30/pred_v36/nflverse's spread_line). Every
Core V2 function this file calls into (rebuild_board,
build_release_records, bet_tracking) expects standard BOOKMAKER
convention (negative = home favored) instead - exact opposites for
the same fact. This already caused one real, caught-and-fixed bug in
board.py itself (see board.py's own comments) and in this file's
Stage 2 rebuild - fair_display/market_display below are ALWAYS
bookmaker convention and are what every bet_tracking/display call
must use, never the raw fair_home_spread_v3/market_home_spread
columns.

IMPORTANT - shared resources with Core V2:
- The static "repo snapshot" lines file (current_book_lines.csv /
  week_01_book_lines.csv) is the SAME file app.py reads - Stage 1
  confirmed the capture automation writes it to a model-agnostic
  path, so there's no separate V3 copy to keep in sync.
- The live refresh button uses the SAME PARLAY_API_KEY secret and
  therefore the SAME API credit pool as Core V2's own refresh button.
  Clicking both pages' refresh buttons in the same 15-minute window
  spends credits twice for (likely) the same data - the 15-minute
  cache is keyed per page/function, not shared between the two pages.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from parlay_live import fetch_live_lines, format_american, format_spread, rebuild_board
from pregame_filter import select_pregame_game_keys
from core_v2 import bet_tracking

st.set_page_config(
    page_title="V3 NFL Dashboard",
    page_icon="\U0001f3c8",
    layout="wide",

)

ROOT = Path(__file__).parent.parent  # this file lives in pages/, one level below app.py


def locate_file(filename):
    for candidate in [ROOT / filename, ROOT / "data" / filename, ROOT / "data" / "model" / filename]:
        if candidate.exists():
            return candidate
    return None


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_data(ttl=900, show_spinner=False)
def cached_live_refresh(refresh_bucket, season, week, active_game_keys):
    del refresh_bucket  # only used to key the 15-minute cache bucket
    return fetch_live_lines(
        api_key=st.secrets["PARLAY_API_KEY"],
        season=season,
        week=week,
        active_game_keys=set(active_game_keys),
    )


def parse_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.tz_convert("America/New_York")


def display_timestamp(value):
    timestamp = parse_timestamp(value)
    if timestamp is None:
        return "\u2014"
    return timestamp.strftime("%b %-d, %-I:%M %p ET")


def display_home_line(away_team, home_team, home_spread):
    if pd.isna(home_spread):
        return "\u2014"
    home_spread = float(home_spread)
    if abs(home_spread) < 0.001:
        return "Pick'em"
    if home_spread < 0:
        return f"{home_team} {format_spread(home_spread)}"
    return f"{away_team} {format_spread(-home_spread)}"


def board_for_rebuild(v3_board: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt board.py's V3 output into the generic shape rebuild_board
    expects.

    SIGN FLIP (the important part): fair_home_spread_v3 is margin
    convention (positive = home favored, matching pred_v30/pred_v36
    and nflverse's spread_line). rebuild_board and display_home_line
    both expect standard BOOKMAKER convention (negative = home
    favored, e.g. display_home_line shows "-3.5" for a favorite) -
    parsed straight from raw sportsbook odds in
    parlay_live.py/core_v2/market.py. These are exact opposites for
    the same real-world fact. fair_home_spread here is the correctly-
    flipped value used for BOTH rebuild_board's math and every
    display_home_line call below - fair_home_spread_v3 itself is left
    untouched as V3's own margin-convention number, in case anything
    downstream needs that instead.

    Also adds a projection_display_status placeholder V3 doesn't have
    yet - "STANDARD" never matches rebuild_board's EARLY_PROVISIONAL
    check, so it's a safe no-op rather than an invented status.
    """
    out = v3_board.copy()
    out["fair_home_spread"] = -out["fair_home_spread_v3"]
    if "projection_display_status" not in out.columns:
        out["projection_display_status"] = "STANDARD"
    if "game_key" not in out.columns and {"season", "week", "away_team", "home_team"}.issubset(out.columns):
        out["game_key"] = out.apply(
            lambda r: f"{int(r['season'])}_W{int(r['week'])}_{r['away_team']}_{r['home_team']}", axis=1
        )
    return out


st.title("\U0001f3c8 V3 NFL Dashboard")
st.caption(
    "Live 2026 V3 fair lines: frozen Andy/Drew preseason offense/defense priors "
    "+ EPA state updates + HFA/neutral-site. Market lines shown here are comparison-only."
)

st.info(
    "Personnel (PVAR) is not wired into this board yet - every "
    "personnel_adjustment here is 0. See data_assembly.py's docstring."
)

board_filename = st.sidebar.text_input(
    "Board file", value="week_v3_board.csv",
    help="The CSV board.py wrote for the week you want to view.",
)

board_path = locate_file(board_filename)

if board_path is None:
    st.error(
        f"Could not find {board_filename}. Run board.py for this week "
        "first, and make sure the output CSV is in this folder (or a "
        "data/ subfolder) next to app_v3.py."
    )
    st.stop()

raw_board = load_csv(str(board_path))

required_cols = {
    "matchup", "home_team", "away_team", "fair_home_spread_v3",
    "market_home_spread", "edge_points", "model_side", "season", "week",
}
missing_cols = required_cols - set(raw_board.columns)
if missing_cols:
    st.error(f"Board file is missing expected columns: {sorted(missing_cols)}")
    st.stop()

active_season = int(raw_board["season"].iloc[0])
active_week = int(raw_board["week"].iloc[0])

# Reuses the SAME static lines file app.py reads - see module docstring.
static_lines_path = locate_file("current_book_lines.csv") or locate_file("week_01_book_lines.csv")
static_lines = load_csv(str(static_lines_path)) if static_lines_path else pd.DataFrame()

schedule_path = locate_file("data/schedules/schedule_2026.csv") or (ROOT / "data" / "schedules" / "schedule_2026.csv")
schedule = pd.read_csv(schedule_path, low_memory=False) if Path(schedule_path).exists() else pd.DataFrame()

base_board = board_for_rebuild(raw_board)

# V3's own grading history doesn't exist yet (a later stage), so no
# games are excluded as "already completed" here - every game on the
# board is treated as still-pregame for the live-refresh filter.
active_game_keys = select_pregame_game_keys(
    base_board, schedule=schedule, completed_game_keys=set()
)
base_board = base_board.loc[base_board["game_key"].astype(str).isin(active_game_keys)].copy()

for key in ["v3_live_lines", "v3_live_metadata", "v3_refresh_error"]:
    if key not in st.session_state:
        st.session_state[key] = None

has_api_key = "PARLAY_API_KEY" in st.secrets and bool(str(st.secrets["PARLAY_API_KEY"]).strip())

refresh_col, status_col = st.columns([1, 3])
with refresh_col:
    refresh_clicked = st.button(
        "\U0001f504 Refresh Live Spreads + Totals \u2014 2 API credits",
        type="primary",
        disabled=not has_api_key,
        use_container_width=True,
        help="Results are cached for 15 minutes. Shares the same API credit pool as Core V2's refresh button.",
    )
with status_col:
    if st.session_state.v3_live_metadata:
        st.success(f"Live lines loaded: {st.session_state.v3_live_metadata.get('snapshot_timestamp')}")
    else:
        st.caption("Displaying the repository snapshot. The refresh button is session-only.")

if not has_api_key:
    st.warning("Live refresh is disabled until PARLAY_API_KEY is added to this app's Secrets.")

if refresh_clicked:
    refresh_bucket = int(time.time() // 900)
    with st.spinner("Requesting and validating live spreads and totals..."):
        try:
            live_lines, metadata = cached_live_refresh(
                refresh_bucket, active_season, active_week, active_game_keys
            )
            st.session_state.v3_live_lines = live_lines
            st.session_state.v3_live_metadata = metadata
            st.session_state.v3_refresh_error = None
        except Exception as exc:
            st.session_state.v3_refresh_error = f"{type(exc).__name__}: {exc}"

if st.session_state.v3_refresh_error:
    st.error(
        "Live refresh failed validation. The board's static snapshot remains active.\n\n"
        + st.session_state.v3_refresh_error
    )

current_lines = st.session_state.v3_live_lines if st.session_state.v3_live_lines is not None else static_lines

used_live_or_static_lines = not current_lines.empty

try:
    board = rebuild_board(base_board, current_lines) if used_live_or_static_lines else base_board.copy()
except Exception as exc:
    st.error(f"The current market data could not produce a valid board: {type(exc).__name__}: {exc}")
    board = base_board.copy()
    used_live_or_static_lines = False

# Display columns, always in bookmaker convention (negative = home
# favored) since that's what display_home_line expects - see
# board_for_rebuild's docstring for why fair_home_spread_v3 itself
# can't be used directly.
#
# fair_home_spread_v3 survives untouched through both rebuild_board's
# output and the fallback (rebuild_board never drops it), so this
# works identically either way.
board["fair_display"] = -board["fair_home_spread_v3"]

if used_live_or_static_lines:
    # rebuild_board just computed market_home_spread fresh from live/
    # static sportsbook quotes - already bookmaker convention.
    board["market_display"] = board["market_home_spread"]
else:
    # No live or static lines at all - falling back to board.py's own
    # baked-in market_home_spread, which is margin convention (from
    # market_home_margin) like fair_home_spread_v3, so needs the same
    # flip. model_side/edge_points in this fallback path are board.py's
    # own precomputed values and are already correct as-is (edge_points
    # is a symmetric abs() difference; model_side was fixed to use
    # margin convention directly in board.py).
    board["market_display"] = -board["market_home_spread"]

st.sidebar.header("V3 Controls")

minimum_edge = st.sidebar.slider(
    "Minimum V3 vs. market disagreement",
    min_value=0.0, max_value=15.0, value=0.0, step=0.5,
)

st.sidebar.divider()
st.sidebar.write("Model: **2026 V3 live state**")
st.sidebar.write("Rating update: z-standardized EPA bridge; offense prior 8 games / defense prior 20 games")
st.sidebar.write("HFA: 2.20 pts; true neutral sites: 0")
market_status = (
    str(raw_board["market_signal_status"].iloc[0])
    if "market_signal_status" in raw_board.columns else "unknown"
)
if market_status == "disabled_missing_week2_v3_pregame_snapshot":
    st.sidebar.write("V3.3 market signal: **OFF for Week 3** — missing preserved Week 2 V3 pregame snapshot")
else:
    st.sidebar.write(f"V3.3 market signal status: **{market_status}**")
st.sidebar.write("PVAR: **not wired**; personnel adjustment = 0")

tab1, tab2, tab3 = st.tabs(["Market Board", "Game Detail", "My Bets"])

with tab1:
    filtered = board.loc[board["edge_points"] >= minimum_edge].copy()

    m1, m2, m3 = st.columns(3)
    m1.metric("Games", len(board))
    m2.metric("Disagreements \u22651", int((board["edge_points"] >= 1).sum()))
    m3.metric("Mean |edge|", f'{board["edge_points"].mean():.2f}' if len(board) else "\u2014")

    display = pd.DataFrame(
        {
            "Matchup": filtered["matchup"] if "matchup" in filtered else filtered["away_team"] + " @ " + filtered["home_team"],
            "V3 fair": filtered.apply(
                lambda row: display_home_line(row["away_team"], row["home_team"], row["fair_display"]), axis=1
            ),
            "Market": filtered.apply(
                lambda row: display_home_line(row["away_team"], row["home_team"], row["market_display"]), axis=1
            ),
            "Lean": filtered["model_side"],
            "Edge (pts)": filtered["edge_points"].round(2),
        }
    )

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(
        "Edge measures disagreement with the market lines shown above "
        "(live if refreshed this session, otherwise the repository "
        "snapshot) - not an expected-value estimate."
    )

with tab2:
    matchup_col = "matchup" if "matchup" in board.columns else None
    options = board[matchup_col].tolist() if matchup_col else (board["away_team"] + " @ " + board["home_team"]).tolist()
    selected_game = st.selectbox("Game", options)
    if matchup_col:
        row = board.loc[board[matchup_col] == selected_game].iloc[0]
    else:
        row = board.loc[(board["away_team"] + " @ " + board["home_team"]) == selected_game].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("V3 Fair Line", display_home_line(row["away_team"], row["home_team"], row["fair_display"]))
    c1.metric("Market", display_home_line(row["away_team"], row["home_team"], row["market_display"]))
    c2.metric("Disagreement", f'{row["edge_points"]:.2f} pts')
    c2.metric("Lean", row["model_side"])
    if "pred_v30" in row:
        c3.metric("pred_v30 (base)", f'{row["pred_v30"]:.2f}')
    if "market_adjustment" in row:
        c3.metric("Market adjustment", f'{row["market_adjustment"]:.2f}')

    if "personnel_adjustment" in row and "neutral_hfa_adjustment" in row:
        st.caption(
            f"Personnel adjustment: {row['personnel_adjustment']:.2f} pts "
            "(0 until PVAR is wired in) | "
            f"Neutral-site adjustment: {row['neutral_hfa_adjustment']:.2f} pts"
        )

with tab3:
    st.subheader("My Bets (V3)")
    st.caption(
        "Record actual V3 spread bets here. New entries snapshot the V3 fair line "
        "and market consensus currently displayed above; later refreshes do not "
        "rewrite that decision-time snapshot. This ledger and its grades are "
        "separate from Core V2's My Bets tab - see this file's own docstring."
    )

    bets_path_v3 = ROOT / "data" / "tracking" / "my_bets_v3.csv"
    bets_v3 = bet_tracking.load_bets(bets_path_v3)

    v3_grades_path = locate_file("v3_game_grades.csv") or locate_file("data/tracking/v3_game_grades.csv")
    v3_game_grades = load_csv(str(v3_grades_path)) if v3_grades_path else pd.DataFrame()

    bet_game_v3 = st.selectbox(
        "Game", board["matchup"].tolist() if "matchup" in board.columns
        else (board["away_team"] + " @ " + board["home_team"]).tolist(),
        key="v3_bet_game",
    )
    bet_row_v3 = board.loc[
        (board["matchup"] if "matchup" in board.columns else board["away_team"] + " @ " + board["home_team"]).eq(bet_game_v3)
    ].iloc[0]

    with st.form("enter_v3_bet", clear_on_submit=False):
        bet_team_v3 = st.selectbox(
            "Team",
            [str(bet_row_v3["away_team"]), str(bet_row_v3["home_team"])],
            key=f"v3_bet_team_{bet_game_v3}",
        )
        b1, b2, b3 = st.columns(3)
        bet_spread_v3 = b1.number_input("Spread", value=0.0, step=0.5, format="%.1f", key="v3_bet_spread")
        bet_odds_v3 = b2.number_input("American odds", value=-110, step=1, key="v3_bet_odds")
        bet_stake_v3 = b3.number_input("Stake ($)", min_value=0.0, value=25.0, step=5.0, key="v3_bet_stake")
        sportsbook_v3 = st.text_input("Sportsbook", value="bet365", key="v3_bet_book")
        notes_v3 = st.text_input("Notes", value="", key="v3_bet_notes")
        submitted_v3 = st.form_submit_button("Save bet")

        if submitted_v3:
            entered_at_v3 = pd.Timestamp.now(tz="UTC").isoformat()
            try:
                game_id_value_v3 = str(
                    bet_row_v3.get(
                        "game_id",
                        f"{active_season}_{active_week:02d}_{bet_row_v3['away_team']}_{bet_row_v3['home_team']}",
                    )
                )
                # fair_display/market_display (computed above) are bookmaker
                # convention - the ONLY correct input to bet_tracking here.
                # fair_home_spread_v3/market_home_spread are margin convention
                # and must never be passed to bet_tracking directly.
                selected_model_spread_v3 = bet_tracking.selected_team_model_spread(
                    bet_team_v3,
                    str(bet_row_v3["away_team"]),
                    str(bet_row_v3["home_team"]),
                    float(bet_row_v3["fair_display"]),
                )
                bet_vs_model_points_v3, bet_alignment_v3 = bet_tracking.classify_bet_alignment(
                    float(bet_spread_v3),
                    selected_model_spread_v3,
                )
                bets_v3 = bet_tracking.append_bet(
                    bets_path_v3,
                    {
                        "season": int(active_season),
                        "week": int(active_week),
                        "game_id": game_id_value_v3,
                        "matchup": bet_game_v3,
                        "team": bet_team_v3,
                        "bet_spread": float(bet_spread_v3),
                        "odds_american": int(bet_odds_v3),
                        "sportsbook": sportsbook_v3.strip(),
                        "stake": float(bet_stake_v3),
                        "entered_at_utc": entered_at_v3,
                        "model_fair_home_spread_at_entry": float(bet_row_v3["fair_display"]),
                        "market_home_spread_at_entry": (
                            float(bet_row_v3["market_display"]) if pd.notna(bet_row_v3["market_display"]) else pd.NA
                        ),
                        "model_side_at_entry": str(bet_row_v3.get("model_side", "")),
                        "model_edge_points_at_entry": (
                            float(bet_row_v3["edge_points"]) if pd.notna(bet_row_v3["edge_points"]) else pd.NA
                        ),
                        "model_selected_team_spread_at_entry": selected_model_spread_v3,
                        "bet_vs_model_points": bet_vs_model_points_v3,
                        "bet_alignment": bet_alignment_v3,
                        "status": "OPEN",
                        "notes": notes_v3.strip(),
                    },
                )
                st.success(
                    f"Saved {bet_team_v3} {format_spread(float(bet_spread_v3))} "
                    f"{format_american(int(bet_odds_v3))} at {sportsbook_v3.strip()}."
                )
            except OSError:
                st.error(
                    "This deployment cannot write persistent files. Download the ledger below "
                    "and commit it to data/tracking/my_bets_v3.csv, or enter the bet from a writable deployment."
                )

    if bets_v3.empty:
        st.info("No V3 bets recorded yet.")
    else:
        display_bets_v3 = bet_tracking.grade_bets_against_closes(bets_v3, v3_game_grades)
        display_bets_v3["Bet"] = display_bets_v3.apply(
            lambda row: f"{row['team']} {format_spread(float(row['bet_spread']))} "
            f"{format_american(float(row['odds_american']))}",
            axis=1,
        )
        display_bets_v3["Stake"] = pd.to_numeric(display_bets_v3["stake"], errors="coerce").map(
            lambda value: f"USD {value:.2f}" if pd.notna(value) else "\u2014"
        )
        display_bets_v3["Entered"] = display_bets_v3["entered_at_utc"].map(display_timestamp)
        display_bets_v3["Model at entry"] = display_bets_v3["model_fair_home_spread_at_entry"].map(
            lambda value: f"{float(value):+.2f}" if pd.notna(value) else "Legacy / unavailable"
        )
        display_bets_v3["Market at entry"] = display_bets_v3["market_home_spread_at_entry"].map(
            lambda value: f"{float(value):+.1f}" if pd.notna(value) else "Legacy / unavailable"
        )
        display_bets_v3["V3 bet-side fair"] = pd.to_numeric(
            display_bets_v3["model_selected_team_spread_at_entry"], errors="coerce"
        ).map(lambda value: format_spread(value) if pd.notna(value) else "\u2014")
        display_bets_v3["Vs model"] = pd.to_numeric(
            display_bets_v3["bet_vs_model_points"], errors="coerce"
        ).map(lambda value: f"{value:+.2f} pts" if pd.notna(value) else "\u2014")
        display_bets_v3["Alignment"] = display_bets_v3["bet_alignment"].map(
            {"WITH_MODEL": "With model", "AGAINST_MODEL": "Against model", "NEUTRAL": "Neutral"}
        ).fillna("\u2014")
        display_bets_v3["Close"] = pd.to_numeric(
            display_bets_v3["closing_selected_team_spread"], errors="coerce"
        ).map(lambda value: format_spread(value) if pd.notna(value) else "Pending")
        display_bets_v3["Actual bet CLV"] = pd.to_numeric(
            display_bets_v3["actual_bet_clv_points"], errors="coerce"
        ).map(lambda value: f"{value:+.1f} pts" if pd.notna(value) else "Pending")
        display_bets_v3["Result"] = display_bets_v3["actual_bet_result"].fillna("Pending")
        display_bets_v3["P/L"] = pd.to_numeric(
            display_bets_v3["profit_loss"], errors="coerce"
        ).map(lambda value: f"USD {value:+.2f}" if pd.notna(value) else "Pending")
        st.dataframe(
            display_bets_v3[
                ["week", "matchup", "Bet", "sportsbook", "Stake", "Entered",
                 "V3 bet-side fair", "Vs model", "Alignment",
                 "Close", "Actual bet CLV", "Result", "P/L",
                 "Model at entry", "Market at entry", "status"]
            ].rename(
                columns={"week": "Week", "matchup": "Matchup", "sportsbook": "Book", "status": "Status"}
            ).sort_values(["Week", "Entered"], ascending=[False, False]),
            use_container_width=True,
            hide_index=True,
        )

        total_staked_v3 = pd.to_numeric(bets_v3["stake"], errors="coerce").sum()
        graded_actual_v3 = display_bets_v3.loc[display_bets_v3["actual_bet_result"].isin(["Win", "Loss", "Push"])].copy()
        actual_wins_v3 = int(graded_actual_v3["actual_bet_result"].eq("Win").sum())
        actual_losses_v3 = int(graded_actual_v3["actual_bet_result"].eq("Loss").sum())
        actual_pushes_v3 = int(graded_actual_v3["actual_bet_result"].eq("Push").sum())
        clv_rows_v3 = pd.to_numeric(display_bets_v3["actual_bet_clv_points"], errors="coerce").dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tracked stake", f"USD {total_staked_v3:.2f}")
        m2.metric("Actual bet record", f"{actual_wins_v3}-{actual_losses_v3}-{actual_pushes_v3}")
        m3.metric("Avg actual bet CLV", f"{clv_rows_v3.mean():+.2f} pts" if not clv_rows_v3.empty else "Pending")
        total_pnl_v3 = pd.to_numeric(graded_actual_v3["profit_loss"], errors="coerce").sum()
        m4.metric("Actual bet P/L", f"USD {total_pnl_v3:+.2f}" if not graded_actual_v3.empty else "Pending")
        st.download_button(
            "Download V3 bet ledger CSV",
            data=bets_v3.to_csv(index=False).encode("utf-8"),
            file_name="my_bets_v3.csv",
            mime="text/csv",
        )

    st.caption(
        "Actual bet CLV is separate from model-to-close tracking. It compares the exact spread you "
        "bet with the automated pre-kickoff consensus close from the bettor's perspective; positive "
        "means you beat the close. If the close was not captured, it stays blank rather than being backfilled."
    )
