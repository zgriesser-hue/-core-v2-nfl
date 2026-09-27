from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

BET_COLUMNS = [
    "bet_id","season","week","game_id","matchup","bet_type","selection","team",
    "bet_line","bet_spread","odds_american","sportsbook","stake","entered_at_utc",
    "model_fair_home_spread_at_entry","market_home_spread_at_entry",
    "market_total_at_entry","model_side_at_entry","model_edge_points_at_entry",
    "model_selected_team_spread_at_entry","bet_vs_model_points","bet_alignment",
    "status","notes",
]

def load_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BET_COLUMNS)
    bets=pd.read_csv(path)
    for column in BET_COLUMNS:
        if column not in bets:
            bets[column]=pd.NA
    # Backward-compatible migration of the original spread-only ledger.
    # Normalize blank/NaN bet types before returning so Streamlit never sees
    # a float NaN where it expects a wager-type string.
    bet_type = bets["bet_type"].astype("string").str.strip().str.upper()
    bets["bet_type"] = bet_type.mask(bet_type.isna() | bet_type.eq(""), "SPREAD")
    missing_selection = bets["selection"].isna() | bets["selection"].astype("string").str.strip().eq("")
    bets.loc[missing_selection,"selection"]=bets.loc[missing_selection,"team"]
    spread_rows=bets["bet_type"].eq("SPREAD")
    missing_line=bets["bet_line"].isna()
    bets.loc[spread_rows & missing_line,"bet_line"]=bets.loc[spread_rows & missing_line,"bet_spread"]
    return bets[BET_COLUMNS].copy()

def make_bet_id(season:int,week:int,game_id:str,selection:str,entered_at_utc:str)->str:
    stamp=pd.Timestamp(entered_at_utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{int(season)}-{int(week):02d}-{game_id}-{selection}-{stamp}"

def append_bet(path:Path,record:dict)->pd.DataFrame:
    bets=load_bets(path)
    row={column:record.get(column,pd.NA) for column in BET_COLUMNS}
    if pd.isna(row["entered_at_utc"]):
        row["entered_at_utc"]=datetime.now(timezone.utc).isoformat(timespec="microseconds")
    if pd.isna(row["bet_type"]): row["bet_type"]="SPREAD"
    if pd.isna(row["selection"]): row["selection"]=row["team"]
    if pd.isna(row["bet_line"]) and row["bet_type"]=="SPREAD": row["bet_line"]=row["bet_spread"]
    if pd.isna(row["bet_id"]):
        row["bet_id"]=make_bet_id(int(row["season"]),int(row["week"]),str(row["game_id"]),str(row["selection"]),str(row["entered_at_utc"]))
    duplicate_fields=["season","week","game_id","bet_type","selection","bet_line","odds_american","sportsbook","stake"]
    if not bets.empty:
        duplicate=pd.Series(True,index=bets.index)
        for field in duplicate_fields:
            duplicate &= bets[field].astype(str).str.strip().str.lower().eq(str(row[field]).strip().lower())
        if duplicate.any(): return bets
    updated=pd.concat([bets,pd.DataFrame([row])],ignore_index=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    updated.to_csv(path,index=False)
    return updated

def delete_bet(path:Path,bet_id:str)->pd.DataFrame:
    bets=load_bets(path)
    updated=bets.loc[~bets["bet_id"].astype(str).eq(str(bet_id))].copy()
    path.parent.mkdir(parents=True,exist_ok=True); updated.to_csv(path,index=False)
    return updated

def american_profit(stake:float,odds:float)->float:
    stake,odds=float(stake),float(odds)
    return stake*(100.0/abs(odds)) if odds<0 else stake*(odds/100.0)

def selected_team_model_spread(team:str,away_team:str,home_team:str,fair_home_spread:float)->float:
    if str(team)==str(home_team): return float(fair_home_spread)
    if str(team)==str(away_team): return -float(fair_home_spread)
    raise ValueError(f"Bet team {team} is not in {away_team} @ {home_team}")

def classify_bet_alignment(bet_spread:float,selected_team_model_spread_value:float,neutral_tolerance:float=.25)->tuple[float,str]:
    points=float(bet_spread)-float(selected_team_model_spread_value)
    return points,("WITH_MODEL" if points>neutral_tolerance else "AGAINST_MODEL" if points<-neutral_tolerance else "NEUTRAL")

def grade_bets_against_closes(bets:pd.DataFrame,game_grades:pd.DataFrame)->pd.DataFrame:
    """Grade SPREAD, MONEYLINE and TOTAL wagers. Spread CLV is retained.

    Moneyline and total result/P&L need only final scores. Their CLV remains blank
    until dedicated closing ML/total snapshots are persisted.
    """
    out=bets.copy()
    for c in ["closing_selected_team_spread","actual_bet_clv_points","actual_bet_result","profit_loss"]:
        out[c]=pd.NA
    if out.empty or game_grades.empty: return out
    grades=game_grades.copy()
    grades["matchup"]=grades["away_team"].astype(str)+" @ "+grades["home_team"].astype(str)
    keys=["season","week","matchup"]
    merged=out.merge(grades[keys+[col for col in [
            "away_team","home_team","closing_home_spread","closing_status","away_score","home_score",
            "closing_total","closing_away_moneyline","closing_home_moneyline"
        ] if col in grades.columns]],on=keys,how="left")

    def grade_row(row):
        away_score,home_score=row.get("away_score"),row.get("home_score")
        if pd.isna(away_score) or pd.isna(home_score):
            return pd.Series([pd.NA,pd.NA,pd.NA,pd.NA])
        typ=str(row.get("bet_type","SPREAD")).upper()
        selection=str(row.get("selection",row.get("team","")))
        stake=float(row["stake"]); odds=float(row["odds_american"])
        close=pd.NA; clv=pd.NA
        if typ=="MONEYLINE":
            if float(home_score)==float(away_score): result="Push"
            else:
                winner=str(row["home_team"]) if float(home_score)>float(away_score) else str(row["away_team"])
                result="Win" if selection==winner else "Loss"
            close_col = "closing_home_moneyline" if selection==str(row["home_team"]) else "closing_away_moneyline"
            if pd.notna(row.get(close_col)):
                close=float(row[close_col])
                def implied(american):
                    american=float(american)
                    return 100.0/(american+100.0) if american>0 else abs(american)/(abs(american)+100.0)
                # Positive percentage points means the entered price implied a lower
                # break-even probability than the closing market price.
                clv=(implied(close)-implied(odds))*100.0
        elif typ=="TOTAL":
            line=pd.to_numeric(pd.Series([row.get("bet_line")]),errors="coerce").iloc[0]
            if pd.isna(line): return pd.Series([pd.NA,pd.NA,pd.NA,pd.NA])
            diff=float(home_score)+float(away_score)-float(line)
            if abs(diff)<1e-9: result="Push"
            elif selection.upper()=="OVER": result="Win" if diff>0 else "Loss"
            else: result="Win" if diff<0 else "Loss"
            if pd.notna(row.get("closing_total")):
                close=float(row["closing_total"])
                clv=(close-float(line)) if selection.upper()=="OVER" else (float(line)-close)
        else:
            if row.get("closing_status")=="CAPTURED" and pd.notna(row.get("closing_home_spread")):
                home_close=float(row["closing_home_spread"])
                close=home_close if selection==str(row["home_team"]) else -home_close
                clv=float(row["bet_line"])-float(close)
            if selection==str(row["home_team"]):
                team_score,opp_score=float(home_score),float(away_score)
            else:
                team_score,opp_score=float(away_score),float(home_score)
            ats=team_score-opp_score+float(row["bet_line"])
            result="Win" if ats>0 else "Loss" if ats<0 else "Push"
        pnl=american_profit(stake,odds) if result=="Win" else -stake if result=="Loss" else 0.0
        return pd.Series([close,clv,result,pnl])

    merged[["closing_selected_team_spread","actual_bet_clv_points","actual_bet_result","profit_loss"]]=merged.apply(grade_row,axis=1)
    return merged

def summarize_bets(graded:pd.DataFrame)->pd.DataFrame:
    """Record, stake, P/L and ROI by bet type plus OVERALL."""
    if graded.empty:
        return pd.DataFrame(columns=["bet_type","bets","wins","losses","pushes","stake","profit_loss","roi"])
    g=graded.copy()
    g["stake_num"]=pd.to_numeric(g["stake"],errors="coerce").fillna(0.0)
    g["pnl_num"]=pd.to_numeric(g["profit_loss"],errors="coerce")
    rows=[]
    for label,frame in [("OVERALL",g)]+list(g.groupby("bet_type",dropna=False)):
        graded_rows=frame[frame["actual_bet_result"].isin(["Win","Loss","Push"])]
        stake=float(graded_rows["stake_num"].sum()); pnl=float(graded_rows["pnl_num"].fillna(0).sum())
        rows.append({"bet_type":label,"bets":len(graded_rows),"wins":int((graded_rows["actual_bet_result"]=="Win").sum()),"losses":int((graded_rows["actual_bet_result"]=="Loss").sum()),"pushes":int((graded_rows["actual_bet_result"]=="Push").sum()),"stake":stake,"profit_loss":pnl,"roi":(pnl/stake if stake else float("nan"))})
    return pd.DataFrame(rows)
