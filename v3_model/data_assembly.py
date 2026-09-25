"""Weekly data assembly helpers for V3."""
from __future__ import annotations
import pandas as pd
from . import market, personnel

def load_team_game_history(path:str,through_season:int,through_week:int)->pd.DataFrame:
    df=pd.read_csv(path)
    mask=(df["season"]<through_season)|((df["season"]==through_season)&(df["week"]<through_week))
    return df[mask].reset_index(drop=True)

def build_defensive_exposure_diagnostics(audit_path:str,season:int,week:int)->tuple[pd.DataFrame,pd.DataFrame]:
    """Return player-level defensive audit plus team-level shadow diagnostics.

    Production V3.2 still consumes only def_loss_pvar. Extra columns are
    retained for shadow testing and auditability; they do not change fair lines.
    """
    audit=pd.read_csv(audit_path)
    required={"team","weighted_role_exposure"}
    missing=required-set(audit.columns)
    if missing:
        raise ValueError(f"Personnel audit missing required columns: {sorted(missing)}")
    if "matched_position" in audit.columns:
        fallback=audit.get("position_reported",pd.Series(index=audit.index,dtype=object))
        position=audit["matched_position"].where(audit["matched_position"].notna() & audit["matched_position"].astype(str).str.strip().ne(""),fallback)
    else:
        position=audit.get("position_reported",pd.Series(index=audit.index,dtype=object))
    work=audit.copy()
    work["position_group"]=position.map(personnel.normalize_position_group)
    work["is_qb"]=work.get("is_qb",pd.Series(False,index=work.index)).fillna(False).astype(bool)
    work["excluded_manual_overlap"]=work.get("excluded_manual_overlap",pd.Series(False,index=work.index)).fillna(False).astype(bool)
    work["weighted_role_exposure"]=pd.to_numeric(work["weighted_role_exposure"],errors="coerce")
    work["expected_role"]=pd.to_numeric(work.get("expected_role",pd.Series(index=work.index,dtype=float)),errors="coerce")
    work["status_weight"]=pd.to_numeric(work.get("status_weight",pd.Series(index=work.index,dtype=float)),errors="coerce")
    eligible=work["position_group"].isin(personnel.DEFENSIVE_POSITION_GROUPS)&~work["is_qb"]&~work["excluded_manual_overlap"]
    players=work.loc[eligible].copy()
    players["season"]=int(season); players["week"]=int(week)
    players["exposure_missing"]=players["weighted_role_exposure"].isna()
    players["exposure_for_sum"]=players["weighted_role_exposure"].fillna(0.0)

    rows=[]
    for team,g in players.groupby("team",sort=True):
        vals=g["exposure_for_sum"]
        row={"season":int(season),"week":int(week),"team":team,
             "def_loss_pvar":float(vals.sum()),
             "max_individual_loss":float(vals.max()) if len(vals) else 0.0,
             "def_players_reported":int(len(g)),
             "def_players_with_exposure":int((~g["exposure_missing"]).sum()),
             "def_players_missing_exposure":int(g["exposure_missing"].sum())}
        for group in sorted(personnel.DEFENSIVE_POSITION_GROUPS):
            row[f"{group.lower()}_loss"]=float(g.loc[g["position_group"]==group,"exposure_for_sum"].sum())
        rows.append(row)
    team_diag=pd.DataFrame(rows)
    cols=["season","week","team","def_loss_pvar","edge_loss","idl_loss","lb_loss","db_loss","max_individual_loss","def_players_reported","def_players_with_exposure","def_players_missing_exposure"]
    if team_diag.empty:
        team_diag=pd.DataFrame(columns=cols)
    else:
        team_diag=team_diag[cols]
    return players,team_diag

def build_defensive_exposure_table(audit_path:str,season:int,week:int)->pd.DataFrame:
    _,diag=build_defensive_exposure_diagnostics(audit_path,season,week)
    return diag[["season","week","team","def_loss_pvar"]].copy()

def build_market_signal_table(market_history_path:str)->pd.DataFrame:
    games=pd.read_csv(market_history_path)
    team_signal=market.build_team_market_signal(games)
    result=market.add_lagged_ewm_signal(team_signal,span=5)
    result["prior_games_observed"]=result.groupby(["season","team"]).cumcount()
    return result

def build_matchups(schedule_path:str,season:int,week:int,market_signal_table:pd.DataFrame|None=None,defensive_exposure_table:pd.DataFrame|None=None)->pd.DataFrame:
    sched=pd.read_csv(schedule_path)
    wk=sched[(sched["season"]==season)&(sched["week"]==week)].copy()
    if market_signal_table is not None:
        sig=market_signal_table[(market_signal_table["season"]==season)&(market_signal_table["week"]==week)][["team","market_signal_ewm5","prior_games_observed"]]
        for side in ("home","away"):
            side_sig=sig.rename(columns={"team":f"{side}_team","market_signal_ewm5":f"{side}_market_signal_ewm5","prior_games_observed":f"{side}_prior_games_observed"})
            wk=wk.merge(side_sig,on=f"{side}_team",how="left")
        for side in ("home","away"):
            found=wk[f"{side}_prior_games_observed"].notna()
            genuine_zero=found&(wk[f"{side}_prior_games_observed"]==0)
            wk[f"{side}_market_signal_provenance"]=pd.Series(pd.NA,index=wk.index,dtype="object")
            wk.loc[found&~genuine_zero,f"{side}_market_signal_provenance"]="observed"
            wk.loc[genuine_zero,f"{side}_market_signal_provenance"]="no_prior_games"
            wk.loc[~found,f"{side}_market_signal_provenance"]="missing_lookup_failed"
            wk[f"{side}_market_signal_ewm5"]=wk[f"{side}_market_signal_ewm5"].fillna(0.0)
            wk=wk.drop(columns=[f"{side}_prior_games_observed"])
    else:
        for side in ("home","away"):
            wk[f"{side}_market_signal_ewm5"]=0.0
            wk[f"{side}_market_signal_provenance"]="no_signal_table_supplied"
    if defensive_exposure_table is not None:
        exp=defensive_exposure_table[(defensive_exposure_table["season"]==season)&(defensive_exposure_table["week"]==week)][["team","def_loss_pvar"]]
        for side in ("home","away"):
            side_exp=exp.rename(columns={"team":f"{side}_team","def_loss_pvar":f"{side}_def_loss"})
            wk=wk.merge(side_exp,on=f"{side}_team",how="left")
            found=wk[f"{side}_def_loss"].notna()
            wk[f"{side}_def_loss"]=wk[f"{side}_def_loss"].fillna(0.0)
            wk[f"{side}_personnel_provenance"]=found.map({True:"real_audit_data",False:"no_defensive_exposure_reported"})
    else:
        wk["home_def_loss"]=0.0; wk["away_def_loss"]=0.0
        wk["home_personnel_provenance"]="pvar_not_wired"; wk["away_personnel_provenance"]="pvar_not_wired"
    if "neutral_site" not in wk.columns:
        wk["neutral_site"]=wk["location"].astype(str).str.strip().str.casefold().eq("neutral") if "location" in wk.columns else False
    return wk[["season","week","game_id","home_team","away_team","neutral_site","home_def_loss","away_def_loss","home_personnel_provenance","away_personnel_provenance","home_market_signal_ewm5","away_market_signal_ewm5","home_market_signal_provenance","away_market_signal_provenance"]]
