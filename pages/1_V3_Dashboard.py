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
from scripts.v3_performance import grade_releases

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
def load_csv(path_string, file_mtime_ns=None):
    del file_mtime_ns  # include modification time in Streamlit cache key
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

board_choices = {"Current Week": ROOT / "week_v3_board.csv"}
for archived_path in sorted((ROOT / "data" / "research" / "v3").glob("week_*_v3_board.csv"), reverse=True):
    try:
        archived_week = int(archived_path.name.split("_")[1])
    except (IndexError, ValueError):
        continue
    board_choices[f"Week {archived_week} archive"] = archived_path

board_choice = st.sidebar.selectbox(
    "V3 board", list(board_choices), index=0, key="v3_board_source_current_week_v2",
    help="Current Week follows the published V3 board. Select an archive to review an older week.",
)
board_path = board_choices[board_choice]
if not board_path.exists():
    st.error(f"V3 board is missing: {board_path.name}")
    st.stop()

raw_board = load_csv(str(board_path), board_path.stat().st_mtime_ns)
if "personnel_status" in raw_board.columns and not raw_board.empty:
    status = str(raw_board["personnel_status"].iloc[0])
    if status == "EARLY_PROVISIONAL_DEF_PVAR_UNAVAILABLE":
        st.warning(
            "EARLY V3 RESEARCH BOARD: EPA ratings and reviewed QB availability adjustments are applied, "
            "but current-week defensive PVAR is unavailable. These fair lines omit unknown "
            "defensive injury effects; do not treat the listed edges as fully injury-adjusted."
        )
    elif status == "EARLY_PROVISIONAL_KNOWN_DEF_PVAR_ONLY":
        st.warning("EARLY V3 RESEARCH BOARD: reviewed confirmed defensive losses and QB scenarios are applied. "
                   "Other defensive availability is still unknown until complete official reports arrive.")
    elif status in {"native_v3_fixed_225_plus_reviewed_qb", "native_v3_fixed_225_plus_reviewed_availability"}:
        st.success("Native V3 fixed 2.25 defensive PVAR and reviewed QB scenarios are applied.")
    else:
        st.info(f"Personnel source: {status}. Check this board's provenance before using its edges.")

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
st.caption(f"Loaded {board_choice}: {active_season} Week {active_week} ({board_path.name})")

# The V3 snapshot is independent of Core V2's weekly publish. Make lag visible.
current_v2_path = locate_file("current_live_board.csv")
if current_v2_path is not None:
    current_v2 = load_csv(str(current_v2_path))
    if not current_v2.empty and "week" in current_v2.columns:
        current_v2_week = int(current_v2["week"].iloc[0])
        if current_v2_week > active_week:
            st.warning(
                f"V3 Week {current_v2_week} is not published. This is the archived "
                f"Week {active_week} V3 board; the current Core V2 board is Week "
                f"{current_v2_week}. Do not use these V3 lines as current projections."
            )

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
archived_board = not bool(active_game_keys)
if archived_board:
    st.info(
        f"Showing the archived Week {active_week} V3 board. Live odds refresh is disabled "
        "for completed games; market lines below are from the saved pregame board."
    )
else:
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
        disabled=not has_api_key or archived_board,
        use_container_width=True,
        help="Results are cached for 15 minutes. Shares the same API credit pool as Core V2's refresh button.",
    )
with status_col:
    if st.session_state.v3_live_metadata and not archived_board:
        st.success(f"Live lines loaded: {st.session_state.v3_live_metadata.get('snapshot_timestamp')}")
    else:
        st.caption("Displaying the saved pregame snapshot." if archived_board else "Displaying the repository snapshot. The refresh button is session-only.")

if not has_api_key:
    st.warning("Live refresh is disabled until PARLAY_API_KEY is added to this app's Secrets.")

if refresh_clicked and not archived_board:
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

current_lines = (
    pd.DataFrame() if archived_board else
    st.session_state.v3_live_lines if st.session_state.v3_live_lines is not None else static_lines
)

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
st.sidebar.write("Personnel: **" + str(raw_board["personnel_status"].iloc[0]) + "**" if "personnel_status" in raw_board else "Personnel: unknown")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Market Board", "Game Detail", "Power Ratings", "Model Performance", "My Bets", "Early Injury Watch"]
)

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
            "QB availability": filtered["qb_availability_notes"].fillna("") if "qb_availability_notes" in filtered else "",
            "Known defensive outs": filtered["def_availability_notes"].fillna("") if "def_availability_notes" in filtered else "",
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

    if "qb_availability_notes" in row and pd.notna(row["qb_availability_notes"]) and str(row["qb_availability_notes"]).strip():
        st.caption("QB availability: " + str(row["qb_availability_notes"]))

    if "def_availability_notes" in row and pd.notna(row["def_availability_notes"]) and str(row["def_availability_notes"]).strip():
        st.caption("Known defensive outs: " + str(row["def_availability_notes"]))

    if "personnel_adjustment" in row and "neutral_hfa_adjustment" in row:
        st.caption(
            f"Personnel adjustment: {row['personnel_adjustment']:.2f} pts "
            "(saved board value) | "
            f"Neutral-site adjustment: {row['neutral_hfa_adjustment']:.2f} pts"
        )

with tab3:
    st.subheader("V3 Power Ratings")
    st.caption("Pregame V3 team strength entering this week. Higher is better; centered at league average.")
    teams = sorted(set(raw_board["away_team"]).union(set(raw_board["home_team"])))
    # Reconstruct the overall pregame strength directly from the matchup
    # equations: persistent_margin = home_strength - away_strength.
    equations=[]
    values=[]
    for row in raw_board.itertuples(index=False):
        x={team:0.0 for team in teams}
        x[str(row.home_team)]=1.0
        x[str(row.away_team)]=-1.0
        equations.append([x[t] for t in teams])
        values.append(float(row.persistent_margin))
    # Add a zero-sum centering equation; it changes only the arbitrary level,
    # never any matchup difference.
    import numpy as np
    A=np.asarray(equations+[([1.0]*len(teams))],dtype=float)
    y=np.asarray(values+[0.0],dtype=float)
    ratings, *_=np.linalg.lstsq(A,y,rcond=None)
    rating_df=pd.DataFrame({"Team":teams,"V3 Power Rating":ratings}).sort_values("V3 Power Rating",ascending=False)
    rating_df["V3 Power Rating"]=rating_df["V3 Power Rating"].round(2)
    st.dataframe(rating_df,use_container_width=True,hide_index=True)
    st.caption("These are relative overall team-strength ratings reconstructed from V3's saved pregame state; HFA, personnel, market and neutral-site adjustments are not included.")

with tab4:
    st.subheader("V3 Model Performance")
    st.caption("Scores are matched to archived pregame V3 boards. V3 fair margins use positive = home favored.")
    shared_grades_path = ROOT / "data" / "tracking" / "game_grades.csv"
    archived_paths = sorted((ROOT / "data" / "research" / "v3").glob("week_*_v3_board.csv"))
    archived = [load_csv(str(path), path.stat().st_mtime_ns) for path in archived_paths]
    archived = [frame for frame in archived if {"game_key", "fair_home_spread_v3"}.issubset(frame.columns)]
    if not shared_grades_path.exists() or not archived:
        st.info("No archived V3 projections and completed game scores are available yet.")
    else:
        outcomes = load_csv(str(shared_grades_path), shared_grades_path.stat().st_mtime_ns)
        archive = pd.concat(archived, ignore_index=True)
        if archive["game_key"].duplicated().any():
            st.error("Duplicate games in V3 pregame archives; performance cannot be graded safely.")
        else:
            completed = outcomes[["game_key", "away_score", "home_score", "actual_home_margin"]].copy()
            completed = completed.loc[completed["away_score"].notna() & completed["home_score"].notna()]
            if completed["game_key"].duplicated().any():
                st.error("Duplicate graded games; performance cannot be graded safely.")
            else:
                g = archive.merge(completed, on="game_key", how="inner", validate="one_to_one")
                g["fair_home_spread_v3"] = pd.to_numeric(g["fair_home_spread_v3"], errors="coerce")
                g["actual_home_margin"] = pd.to_numeric(g["actual_home_margin"], errors="coerce")
                g = g.dropna(subset=["fair_home_spread_v3", "actual_home_margin"])
                if g.empty:
                    st.info("No archived V3 games have final scores yet.")
                else:
                    g["absolute_error"] = (g["actual_home_margin"] - g["fair_home_spread_v3"]).abs()
                    m1, m2 = st.columns(2)
                    m1.metric("Games graded", len(g))
                    m2.metric("V3 MAE", f'{g["absolute_error"].mean():.2f}')
                    g["V3 fair"] = g.apply(
                        lambda row: display_home_line(row["away_team"], row["home_team"],
                                                      -row["fair_home_spread_v3"]), axis=1
                    )
                    g["Final"] = g.apply(
                        lambda row: f'{row["away_team"]} {int(row["away_score"])} – '
                                    f'{row["home_team"]} {int(row["home_score"])}', axis=1
                    )
                    g["Abs error"] = g["absolute_error"].round(2)
                    st.dataframe(
                        g.rename(columns={"week": "Week", "matchup": "Matchup"})[
                            ["Week", "Matchup", "V3 fair", "Final", "Abs error"]
                        ].sort_values("Week", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption("This grades the saved V3 fair line against the final margin. "
                               "A missing closing line does not block score grading.")

    st.markdown("#### V3 release ATS and closing line")
    st.caption("Uses the first timestamped V3 release with a captured pregame market quote. "
               "ATS is at that quoted spread. Projection vs close is the V3 fair line's "
               "signed edge on its original side at the captured close; market CLV is "
               "the original market spread against the close. These are distinct from "
               "your actual wagers in My Bets. Older archive-only games are excluded.")
    release_path = ROOT / "data" / "tracking" / "v3_projection_releases.csv"
    close_path = ROOT / "data" / "tracking" / "closing_consensus.csv"
    if not release_path.exists() or not close_path.exists() or not shared_grades_path.exists():
        st.info("No timestamped V3 release, score, and close records are available yet.")
    else:
        releases = load_csv(str(release_path), release_path.stat().st_mtime_ns)
        closes = load_csv(str(close_path), close_path.stat().st_mtime_ns)
        outcomes = load_csv(str(shared_grades_path), shared_grades_path.stat().st_mtime_ns)
        try:
            vg = grade_releases(releases, outcomes, closes)
        except ValueError as exc:
            st.error(f"V3 release grading stopped: {exc}")
        else:
            if vg.empty:
                st.info("No V3 release has a timestamped pregame market quote yet.")
            else:
                completed_v3 = vg.loc[vg.ats_result.ne("Pending")]
                wins = int(completed_v3.ats_result.eq("Win").sum())
                losses = int(completed_v3.ats_result.eq("Loss").sum())
                pushes = int(completed_v3.ats_result.eq("Push").sum())
                valid_close_v3 = vg.loc[vg.closing_home_spread.notna()]
                a, b, c, d = st.columns(4)
                a.metric("V3 ATS", f"{wins}-{losses}-{pushes}")
                b.metric("Graded release MAE", f"{completed_v3.absolute_error.mean():.2f}" if len(completed_v3) else "—")
                c.metric("Projection vs close", f"{valid_close_v3.projection_vs_close_points.mean():+.2f}" if len(valid_close_v3) else "—")
                d.metric("Market CLV", f"{valid_close_v3.market_clv_points.mean():+.2f}" if len(valid_close_v3) else "—")
                st.caption(f"{len(completed_v3)} scored releases; {len(valid_close_v3)} valid captured closes. "
                           "Pushes are excluded from win rate. No ROI is inferred without bet odds.")
                shown = vg[["week", "away_team", "home_team", "release_timestamp_utc",
                            "model_side", "fair_home_margin_v3", "release_market_home_spread",
                            "closing_home_spread", "ats_result", "projection_vs_close_points",
                            "market_clv_points"]].copy()
                shown["matchup"] = shown.away_team + " @ " + shown.home_team
                st.dataframe(shown[["week", "matchup", "release_timestamp_utc", "model_side",
                                    "fair_home_margin_v3", "release_market_home_spread",
                                    "closing_home_spread", "ats_result", "projection_vs_close_points",
                                    "market_clv_points"]].sort_values("week", ascending=False),
                             use_container_width=True, hide_index=True)

with tab5:
    st.subheader("My Bets — V3 View")
    st.caption(
        "This is the same permanent wager ledger used by Core V2. "
        "Spread backing uses the saved V3 board for the bet's own week. Results use final scores; closing-line comparisons require a captured close."
    )
    bets_path_v3 = ROOT / "data" / "tracking" / "my_bets.csv"
    bets_v3 = bet_tracking.load_bets(bets_path_v3)
    shared_grades_path = locate_file("game_grades.csv") or locate_file("data/tracking/game_grades.csv")
    shared_grades = load_csv(str(shared_grades_path), shared_grades_path.stat().st_mtime_ns) if shared_grades_path else pd.DataFrame()
    if bets_v3.empty:
        st.info("No bets recorded yet.")
    else:
        display_bets_v3=bet_tracking.grade_bets_against_closes(bets_v3,shared_grades)
        archived_bet_boards = [
            load_csv(str(path), path.stat().st_mtime_ns)
            for path in sorted((ROOT / "data" / "research" / "v3").glob("week_*_v3_board.csv"))
        ]
        archived_bet_boards = [frame for frame in archived_bet_boards
                               if {"season", "week", "matchup", "fair_home_spread_v3",
                                   "away_team", "home_team"}.issubset(frame.columns)]
        historical_board = (pd.concat(archived_bet_boards, ignore_index=True)
                            if archived_bet_boards else raw_board)
        historical_board = historical_board.drop_duplicates(
            ["season", "week", "matchup"], keep="first"
        )
        fair_map = {
            (int(row.season), int(row.week), str(row.matchup)):
            (float(row.fair_home_spread_v3), str(row.away_team), str(row.home_team))
            for row in historical_board.itertuples(index=False)
            if pd.notna(row.fair_home_spread_v3)
        }
        def v3_backing(row):
            typ = str(row.get("bet_type", "SPREAD")).upper()
            if typ != "SPREAD":
                return "N/A — no V3 ML/total model"
            key = (int(row["season"]), int(row["week"]), str(row["matchup"]))
            if key not in fair_map:
                return "Unavailable"
            home_margin, away_team, home_team = fair_map[key]
            team = str(row.get("selection") or row.get("team"))
            try:
                selected_fair = bet_tracking.selected_team_model_spread(
                    team, away_team, home_team, -home_margin
                )
                _, alignment = bet_tracking.classify_bet_alignment(
                    float(row["bet_line"]), selected_fair
                )
                return {"WITH_MODEL": "Backed by V3", "AGAINST_MODEL": "Not backed by V3",
                        "NEUTRAL": "Neutral vs V3"}.get(alignment, "Unavailable")
            except (ValueError, TypeError):
                return "Unavailable"
        display_bets_v3["V3 backing"]=display_bets_v3.apply(v3_backing,axis=1)
        def bet_label(row):
            typ=str(row.get("bet_type","SPREAD")).upper()
            if typ=="MONEYLINE": return f"{row['selection']} ML {format_american(float(row['odds_american']))}"
            if typ=="TOTAL": return f"{str(row['selection']).title()} {float(row['bet_line']):.1f} {format_american(float(row['odds_american']))}"
            return f"{row['selection']} {format_spread(float(row['bet_line']))} {format_american(float(row['odds_american']))}"
        display_bets_v3["Bet"]=display_bets_v3.apply(bet_label,axis=1)
        display_bets_v3["Stake"]=pd.to_numeric(display_bets_v3["stake"],errors="coerce").map(lambda x:f"USD {x:.2f}")
        display_bets_v3["Result"]=display_bets_v3["actual_bet_result"].fillna("Pending")
        display_bets_v3["P/L"]=pd.to_numeric(display_bets_v3["profit_loss"],errors="coerce").map(lambda x:f"USD {x:+.2f}" if pd.notna(x) else "Pending")
        st.dataframe(
            display_bets_v3[["week","matchup","bet_type","Bet","sportsbook","Stake","V3 backing","Result","P/L"]]
            .rename(columns={"week":"Week","matchup":"Matchup","bet_type":"Type","sportsbook":"Book"})
            .sort_values("Week",ascending=False),
            use_container_width=True,hide_index=True
        )
        summary=bet_tracking.summarize_bets(display_bets_v3)
        overall=summary.loc[summary["bet_type"].eq("OVERALL")].iloc[0]
        m1,m2,m3=st.columns(3)
        m1.metric("Overall record",f"{int(overall['wins'])}-{int(overall['losses'])}-{int(overall['pushes'])}")
        m2.metric("Gain / loss",f"USD {overall['profit_loss']:+.2f}")
        backed=display_bets_v3["V3 backing"].eq("Backed by V3").sum()
        m3.metric("Spread bets backed by V3",int(backed))


with tab6:
    st.subheader("Early Injury Watch")
    st.caption("Prior-week absences are verification leads only. Blank current-week status means availability is unknown; no PVAR adjustment is applied.")
    queue_path = ROOT / "data" / "monitor" / f"injury_review_queue_{active_season}.csv"
    if queue_path.exists():
        queue = load_csv(str(queue_path), queue_path.stat().st_mtime_ns)
        queue = queue.loc[queue["season"].eq(active_season) & queue["week"].eq(active_week)]
        if not queue.empty:
            st.markdown("**Official-source review queue**")
            st.caption("Headlines and practice reports are leads. They do not change a fair line until reviewed.")
            queue = queue.copy()
            queue["Potential defensive pts"] = pd.to_numeric(
                queue["potential_def_points"], errors="coerce"
            ).round(2)
            st.dataframe(
                queue.rename(columns={"team": "Team", "player": "Player",
                                      "signal": "Signal", "headline": "Headline",
                                      "source_url": "Source", "first_seen_utc": "First seen",
                                      "review_status": "Review"})[
                    ["Review", "Team", "Player", "Signal", "Potential defensive pts",
                     "Headline", "Source", "First seen"]
                ].sort_values("Potential defensive pts", ascending=False, na_position="last"),
                use_container_width=True, hide_index=True,
                column_config={"Source": st.column_config.LinkColumn("Source")},
            )
        else:
            st.caption("No current-week official-source review leads yet.")
    else:
        st.caption("The official-source monitor has not published a review queue yet.")
    st.divider()
    watch_path = ROOT / "data" / "research" / "v3" / "week_04_early_injury_watchlist.csv"
    if watch_path.exists():
        watch = load_csv(str(watch_path))
        if not watch.empty:
            st.caption(f"Research snapshot: Week {int(watch['week'].iloc[0])}. Refresh when official reports arrive.")
            cols = ["game_id", "team", "player_name", "position_group", "prior_game_status", "current_game_status", "verification", "pvar_production"]
            shown = watch[cols].copy()
            shown["pvar_production"] = pd.to_numeric(shown["pvar_production"], errors="coerce")
            st.dataframe(shown.sort_values("pvar_production", ascending=False, na_position="last"), use_container_width=True, hide_index=True)
        else:
            st.info("The early injury watchlist is empty.")
    else:
        st.info("No early injury watchlist has been published for this week.")
