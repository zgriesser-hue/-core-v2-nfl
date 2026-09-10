
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Core V2 NFL Dashboard",
    page_icon="🏈",
    layout="wide",
)

st.title("🏈 Core V2 NFL Dashboard")
st.caption("Independent fair lines, market comparison, and weekly tracking")

DATA_DIR = Path("data")
PROJ_FILE = DATA_DIR / "week_01_projections.csv"

def load_board():
    if PROJ_FILE.exists():
        return pd.read_csv(PROJ_FILE)

    return pd.DataFrame([{
        "week": 1,
        "away_team": "NE",
        "home_team": "SEA",
        "core_v2_home_margin": 2.2985,
        "core_v2_home_spread": -2.2985,
        "market_home_spread": -3.5,
        "model_side": "NE +3.5",
        "model_market_gap": 1.2015,
        "actual_home_margin": 3.0,
        "abs_error": 0.7015,
        "winner_correct": True,
        "ats_result": "Win",
        "closing_home_spread": None,
        "clv_points": None,
    }])

df = load_board()

def signed_line(team, spread):
    if pd.isna(spread):
        return "—"
    return f"{team} {spread:+.1f}"

tab1, tab2, tab3, tab4 = st.tabs([
    "Weekly Board",
    "Game Detail",
    "Power Ratings",
    "Model Performance"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Week", int(df["week"].max()) if "week" in df else 1)
    c2.metric("Games", len(df))

    if "abs_error" in df and df["abs_error"].notna().any():
        c3.metric("MAE", f'{df["abs_error"].dropna().mean():.2f}')
    else:
        c3.metric("MAE", "—")

    if "winner_correct" in df and df["winner_correct"].notna().any():
        pct = 100 * df["winner_correct"].dropna().astype(bool).mean()
        c4.metric("Winner Accuracy", f"{pct:.0f}%")
    else:
        c4.metric("Winner Accuracy", "—")

    st.subheader("This Week")
    display_cols = [
        c for c in [
            "away_team",
            "home_team",
            "core_v2_home_spread",
            "market_home_spread",
            "model_market_gap",
            "model_side",
            "actual_home_margin",
            "abs_error",
            "ats_result",
            "closing_home_spread",
            "clv_points",
        ] if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Game Breakdown")

    games = [f'{r.away_team} @ {r.home_team}' for r in df.itertuples()]
    selected = st.selectbox("Game", games)
    idx = games.index(selected)
    row = df.iloc[idx]

    left, right = st.columns(2)

    with left:
        st.metric("Core V2 Fair Line", signed_line(row["home_team"], row["core_v2_home_spread"]))
        st.metric("Market Line", signed_line(row["home_team"], row["market_home_spread"]))
        if pd.notna(row.get("model_market_gap")):
            st.metric("Model / Market Gap", f'{row["model_market_gap"]:.2f} pts')

    with right:
        if pd.notna(row.get("actual_home_margin")):
            winner = row["home_team"] if row["actual_home_margin"] > 0 else row["away_team"]
            st.metric("Actual Margin", f'{winner} by {abs(row["actual_home_margin"]):.1f}')
        if pd.notna(row.get("abs_error")):
            st.metric("Projection Error", f'{row["abs_error"]:.2f} pts')
        if pd.notna(row.get("ats_result")):
            st.metric("ATS Result", str(row["ats_result"]))

    st.divider()
    st.caption("Next: Base Rating → QB → HFA → Personnel → Situational → Final waterfall.")

with tab3:
    st.subheader("Power Ratings")
    st.info("This tab will connect to the frozen Andy/Drew priors and later the in-season Core V2 ratings.")

with tab4:
    st.subheader("Model Performance")
    graded = df[df["abs_error"].notna()].copy() if "abs_error" in df else pd.DataFrame()

    if graded.empty:
        st.info("No graded games yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Games Graded", len(graded))
        m2.metric("Margin MAE", f'{graded["abs_error"].mean():.2f}')
        if "winner_correct" in graded:
            acc = 100 * graded["winner_correct"].astype(bool).mean()
            m3.metric("Winner Accuracy", f"{acc:.1f}%")

        chart_df = graded.copy()
        chart_df["game"] = chart_df["away_team"] + " @ " + chart_df["home_team"]
        st.bar_chart(chart_df.set_index("game")["abs_error"])

st.sidebar.header("Core V2")
st.sidebar.write("Production version: **Week 1 launch**")
st.sidebar.write("PVAR: **Not active**")
st.sidebar.write("Market in model: **No**")
st.sidebar.divider()
st.sidebar.caption("Development stays separate. Only validated features get promoted into production.")
