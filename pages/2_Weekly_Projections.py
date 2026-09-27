"""Persistent weekly projection archive for Core V2 and V3.

Unlike the live dashboards, this page never removes games after kickoff. It is
for reviewing what each model projected for the full slate, including completed
games.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weekly Model Projections", page_icon="🏈", layout="wide")
ROOT=Path(__file__).parent.parent

def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def fmt_line(team, value):
    if pd.isna(value): return "—"
    v=float(value)
    if abs(v)<0.001: return "PK"
    return f"{team} {v:+.1f}"

st.title("Weekly Model Projections")
st.caption("Full-week archive. Games remain here after kickoff and after they are completed.")

release_path=ROOT/"data"/"tracking"/"projection_releases.csv"
v2_releases=read_csv(release_path)
v2_weeks=sorted(pd.to_numeric(v2_releases.get("week",pd.Series(dtype=float)),errors="coerce").dropna().astype(int).unique().tolist())

v3_paths=list((ROOT/"data"/"research"/"v3").glob("week_*_v3_board.csv"))
v3_weeks=[]
for p in v3_paths:
    try:
        d=pd.read_csv(p,nrows=1)
        if not d.empty and "week" in d: v3_weeks.append(int(d.iloc[0]["week"]))
    except Exception:
        pass

available=sorted(set(v2_weeks+v3_weeks),reverse=True)
if not available:
    st.info("No archived projection weeks are available yet.")
    st.stop()
week=st.selectbox("Week",available,index=0)

# V2: projection_releases contains immutable snapshots. Use the last pregame
# release for each game so later games are never hidden merely because kickoff passed.
v2=v2_releases.loc[pd.to_numeric(v2_releases["week"],errors="coerce").eq(week)].copy()
if not v2.empty:
    v2["_ts"]=pd.to_datetime(v2["release_timestamp_utc"],utc=True,errors="coerce")
    v2=v2.sort_values("_ts").groupby("game_key",as_index=False).tail(1)
    v2=v2[["game_key","away_team","home_team","fair_home_spread","market_home_spread","release_timestamp_utc"]].copy()
    v2=v2.rename(columns={"fair_home_spread":"v2_home_line","market_home_spread":"v2_market_line","release_timestamp_utc":"v2_snapshot"})

# V3 archived board. V3 uses margin convention (positive = home favored), so
# convert to bookmaker convention for side-by-side display with V2.
v3_file=ROOT/"data"/"research"/"v3"/f"week_{week:02d}_v3_board.csv"
if not v3_file.exists() and week==3:
    v3_file=ROOT/"data"/"research"/"v3"/"week_03_v3_board.csv"
v3=read_csv(v3_file)
if not v3.empty:
    v3=v3[["game_key","away_team","home_team","fair_home_spread_v3","market_home_spread"]].copy()
    v3["v3_home_line"]=-pd.to_numeric(v3["fair_home_spread_v3"],errors="coerce")
    v3["v3_market_line"]=-pd.to_numeric(v3["market_home_spread"],errors="coerce")
    v3=v3.drop(columns=["fair_home_spread_v3","market_home_spread"])

if v2.empty and v3.empty:
    st.info(f"No projection archive is available for Week {week}.")
    st.stop()
if v2.empty:
    merged=v3.copy()
elif v3.empty:
    merged=v2.copy()
else:
    merged=v2.merge(v3,on=["game_key","away_team","home_team"],how="outer")

merged["Matchup"]=merged["away_team"].astype(str)+" @ "+merged["home_team"].astype(str)
if "v2_home_line" in merged:
    merged["V2 Projection"]=merged.apply(lambda r:fmt_line(r["home_team"],r["v2_home_line"]),axis=1)
else: merged["V2 Projection"]="—"
if "v3_home_line" in merged:
    merged["V3 Projection"]=merged.apply(lambda r:fmt_line(r["home_team"],r["v3_home_line"]),axis=1)
else: merged["V3 Projection"]="—"

market_col="v2_market_line" if "v2_market_line" in merged else "v3_market_line"
merged["Market at snapshot"]=merged.apply(lambda r:fmt_line(r["home_team"],r.get(market_col)),axis=1)
st.dataframe(merged[["Matchup","V2 Projection","V3 Projection","Market at snapshot"]].sort_values("Matchup"),use_container_width=True,hide_index=True)

st.caption("V2 uses the latest saved pregame release for each game. V3 is shown from its archived weekly board. This page intentionally does not use the live pregame filter.")
