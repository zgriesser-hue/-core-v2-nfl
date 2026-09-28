"""Shadow-test richer V3 personnel specifications.

This research runner never changes production projections. It expects a
game-level panel containing the frozen V3.0 projection/actual margin plus the
weekly player-audit-derived home/away personnel features. Fitting is strictly
leave-one-season-out when multiple seasons are present.

Required columns:
season, home_margin, pred_v30,
home_def_loss_pvar, away_def_loss_pvar,
home_def_max_player_loss, away_def_max_player_loss,
home_def_edge_loss, away_def_edge_loss,
home_def_idl_loss, away_def_idl_loss,
home_def_lb_loss, away_def_lb_loss,
home_def_db_loss, away_def_db_loss
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_SPECS={
    "continuous_total":["net_total"],
    "capped_total":["net_capped"],
    "max_plus_remainder":["net_max","net_remainder"],
    "position_specific":["net_edge","net_idl","net_lb","net_db"],
    "saturating_total":["net_saturating"],
}

def _features(df:pd.DataFrame)->pd.DataFrame:
    x=pd.DataFrame(index=df.index)
    home=pd.to_numeric(df["home_def_loss_pvar"],errors="coerce").fillna(0)
    away=pd.to_numeric(df["away_def_loss_pvar"],errors="coerce").fillna(0)
    hm=pd.to_numeric(df["home_def_max_player_loss"],errors="coerce").fillna(0)
    am=pd.to_numeric(df["away_def_max_player_loss"],errors="coerce").fillna(0)
    x["net_total"]=away-home
    x["net_capped"]=away.clip(upper=1.5)-home.clip(upper=1.5)
    x["net_max"]=am-hm
    x["net_remainder"]=(away-am).clip(lower=0)-(home-hm).clip(lower=0)
    x["net_saturating"]=np.tanh(away)-np.tanh(home)
    for g in ("edge","idl","lb","db"):
        x[f"net_{g}"]=pd.to_numeric(df[f"away_def_{g}_loss"],errors="coerce").fillna(0)-pd.to_numeric(df[f"home_def_{g}_loss"],errors="coerce").fillna(0)
    return x

def _fit_no_intercept(X,y):
    coef,*_=np.linalg.lstsq(X,y,rcond=None)
    return coef

def evaluate(panel:pd.DataFrame)->pd.DataFrame:
    p=panel.copy()
    p["residual"]=pd.to_numeric(p["home_margin"],errors="coerce")-pd.to_numeric(p["pred_v30"],errors="coerce")
    p=p.loc[p["residual"].notna()].copy()
    f=_features(p)
    seasons=sorted(p["season"].dropna().unique())
    folds=seasons if len(seasons)>1 else [None]
    rows=[]
    for name,cols in FEATURE_SPECS.items():
        errors=[]; base_errors=[]; coefs=[]
        for holdout in folds:
            test=np.ones(len(p),dtype=bool) if holdout is None else p["season"].eq(holdout).to_numpy()
            train=test if holdout is None else ~test
            Xtr=f.loc[train,cols].to_numpy(float); ytr=p.loc[train,"residual"].to_numpy(float)
            Xte=f.loc[test,cols].to_numpy(float); yte=p.loc[test,"residual"].to_numpy(float)
            if len(ytr)==0 or len(yte)==0: continue
            coef=_fit_no_intercept(Xtr,ytr)
            adj=Xte@coef
            errors.extend(np.abs(yte-adj)); base_errors.extend(np.abs(yte)); coefs.append(coef)
        rows.append({"spec":name,"n":len(errors),"base_mae":np.mean(base_errors),"personnel_mae":np.mean(errors),"mae_improvement":np.mean(base_errors)-np.mean(errors),"mean_coefficients":"; ".join(f"{c:.4f}" for c in np.mean(np.vstack(coefs),axis=0)) if coefs else ""})
    return pd.DataFrame(rows).sort_values("mae_improvement",ascending=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("data/research/v3/personnel_shadow_results.csv"))
    a=ap.parse_args()
    p=pd.read_parquet(a.panel) if a.panel.suffix==".parquet" else pd.read_csv(a.panel)
    out=evaluate(p)
    a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print(out.to_string(index=False)); print(f"Saved: {a.out}")
if __name__=="__main__": main()
