from __future__ import annotations

import argparse
from io import BytesIO
import gzip

import numpy as np
import pandas as pd
import requests

from core_v2.qb_efficiency import NFLVERSE_PBP_CSV


def load_pbp(season: int) -> pd.DataFrame:
    r = requests.get(NFLVERSE_PBP_CSV.format(season=season), timeout=90)
    r.raise_for_status()
    cols = ["season","week","season_type","game_id","posteam","defteam",
            "passer_player_id","passer_player_name","qb_epa","pass_attempt","sack"]
    x = pd.read_csv(BytesIO(gzip.decompress(r.content)), usecols=cols, low_memory=False)
    for c in ["season","week","qb_epa","pass_attempt","sack"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["dropback"] = (x["pass_attempt"].eq(1) | x["sack"].eq(1)).astype(int)
    return x.loc[
        x["season"].eq(season) & x["season_type"].astype(str).eq("REG")
        & x["dropback"].eq(1)
    ].dropna(subset=["passer_player_id","posteam","defteam","qb_epa"])


def weighted_qb_means(x: pd.DataFrame) -> pd.DataFrame:
    return x.groupby("passer_player_id").agg(
        qb_epa=("qb_epa","mean"), db=("qb_epa","size")
    )


def solve(hist: pd.DataFrame, qb_prior_db: float, def_prior_db: float) -> tuple[pd.Series,pd.Series,float]:
    if hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0
    mu=float(hist["qb_epa"].mean())
    qbs=sorted(hist["passer_player_id"].astype(str).unique())
    defs=sorted(hist["defteam"].astype(str).unique())
    q=pd.Series(0.0,index=qbs); d=pd.Series(0.0,index=defs)
    qn=hist.groupby("passer_player_id").size().reindex(qbs,fill_value=0).astype(float)
    dn=hist.groupby("defteam").size().reindex(defs,fill_value=0).astype(float)
    for _ in range(20):
        z=hist.copy()
        z["dv"]=z["defteam"].map(d).fillna(0.0)
        z["qv"]=z["qb_epa"]-mu+z["dv"]
        q=z.groupby("passer_player_id")["qv"].mean().reindex(qbs,fill_value=0.0)
        q*=qn/(qn+qb_prior_db)
        z["qv2"]=z["passer_player_id"].map(q).fillna(0.0)
        z["defv"]=mu+z["qv2"]-z["qb_epa"]
        d=z.groupby("defteam")["defv"].mean().reindex(defs,fill_value=0.0)
        d*=dn/(dn+def_prior_db)
    return q,d,mu


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--seasons",default="2017,2018,2019,2022,2023,2024,2025")
    p.add_argument("--qb-prior-db",type=float,default=120.0)
    p.add_argument("--def-prior-db",type=float,default=250.0)
    p.add_argument("--blend-prior-grid",default="75,125,175,250,350")
    args=p.parse_args()
    seasons=[int(v) for v in args.seasons.split(",")]
    grid=[float(v) for v in args.blend_prior_grid.split(",")]
    data={s:load_pbp(s) for s in seasons}
    rows=[]

    # Previous-season QB EPA is a leakage-safe proxy for the preseason QB prior.
    # The winning blend strength is selected leave-one-season-out.
    for season in seasons:
        x=data[season]
        prev=data.get(season-1)
        if prev is None:
            continue
        preseason=weighted_qb_means(prev)
        for week in sorted(int(v) for v in x["week"].dropna().unique() if v>=2):
            hist=x.loc[x["week"]<week].copy()
            cur=x.loc[x["week"]==week].copy()
            q,d,mu=solve(hist,args.qb_prior_db,args.def_prior_db)
            db=hist.groupby("passer_player_id").size()
            for pid,g in cur.groupby("passer_player_id"):
                pid=str(pid)
                if pid not in q.index or pid not in preseason.index or len(g)<10:
                    continue
                opp=str(g["defteam"].mode().iloc[0])
                prior=float(preseason.loc[pid,"qb_epa"])
                actual=float(g["qb_epa"].mean())
                current_oa=mu+float(q[pid])-float(d.get(opp,0.0))
                rows.append(dict(season=season,week=week,pid=pid,db=int(db.get(pid,0)),
                                 preseason_prior=prior,current_oa=current_oa,actual=actual))
    z=pd.DataFrame(rows)
    print(f"Matched observations with prior-season QB history: {len(z)}")
    if z.empty:
        return
    base_mae=float((z["preseason_prior"]-z["actual"]).abs().mean())
    print(f"Static preseason-prior MAE: {base_mae:.5f}")
    chosen=[]
    preds=[]
    for holdout in sorted(z["season"].unique()):
        tr=z.loc[z["season"]!=holdout]; te=z.loc[z["season"]==holdout].copy()
        scored=[]
        for strength in grid:
            w=tr["db"]/(tr["db"]+strength)
            pred=(1-w)*tr["preseason_prior"]+w*tr["current_oa"]
            scored.append((float((pred-tr["actual"]).abs().mean()),strength))
        _,best=min(scored)
        w=te["db"]/(te["db"]+best)
        te["pred"]=(1-w)*te["preseason_prior"]+w*te["current_oa"]
        chosen.append((holdout,best))
        preds.append(te)
    oos=pd.concat(preds,ignore_index=True)
    blend_mae=float((oos["pred"]-oos["actual"]).abs().mean())
    blend_rmse=float(np.sqrt(((oos["pred"]-oos["actual"])**2).mean()))
    base_rmse=float(np.sqrt(((oos["preseason_prior"]-oos["actual"])**2).mean()))
    print("LOSO selected blend strengths:", chosen)
    print(f"Static prior: MAE={base_mae:.5f} RMSE={base_rmse:.5f}")
    print(f"Prior + OA QB: MAE={blend_mae:.5f} RMSE={blend_rmse:.5f}")
    print(f"MAE improvement={base_mae-blend_mae:+.5f} RMSE improvement={base_rmse-blend_rmse:+.5f}")
    print("\nBy season:")
    for s,g in oos.groupby("season"):
        a=float((g["preseason_prior"]-g["actual"]).abs().mean())
        b=float((g["pred"]-g["actual"]).abs().mean())
        print(f"{s}: n={len(g):4d} static={a:.5f} blend={b:.5f} improvement={a-b:+.5f}")


if __name__=="__main__":
    main()
