from __future__ import annotations

import argparse
from io import BytesIO
import gzip
import math

import numpy as np
import pandas as pd
import requests

from core_v2.qb_efficiency import NFLVERSE_PBP_CSV


def load_season(season: int) -> pd.DataFrame:
    r = requests.get(NFLVERSE_PBP_CSV.format(season=season), timeout=90)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    cols = [
        "season", "week", "season_type", "game_id", "posteam", "defteam",
        "passer_player_id", "passer_player_name", "qb_epa", "pass_attempt", "sack", "scramble",
    ]
    x = pd.read_csv(BytesIO(raw), usecols=cols, low_memory=False)
    for c in ["season", "week", "qb_epa", "pass_attempt", "sack", "scramble"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.loc[
        x["season"].eq(season)
        & x["season_type"].astype(str).eq("REG")
        & ((x["pass_attempt"].eq(1)) | x["sack"].eq(1) | x["scramble"].eq(1))
    ].dropna(subset=["passer_player_id", "posteam", "defteam", "qb_epa"])


def weighted_mean(v: pd.Series, w: pd.Series) -> float:
    return float(np.average(v, weights=w)) if float(w.sum()) > 0 else math.nan


def ratings_before_week(hist: pd.DataFrame, prior_db: float, defense_prior_db: float) -> tuple[pd.Series, pd.Series, float]:
    if hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0
    mu = float(hist["qb_epa"].mean())
    qbs = sorted(hist["passer_player_id"].astype(str).unique())
    defs = sorted(hist["defteam"].astype(str).unique())
    qb = pd.Series(0.0, index=qbs)
    defense = pd.Series(0.0, index=defs)
    qn = hist.groupby("passer_player_id").size().reindex(qbs, fill_value=0).astype(float)
    dn = hist.groupby("defteam").size().reindex(defs, fill_value=0).astype(float)
    for _ in range(20):
        z = hist.copy()
        z["d"] = z["defteam"].map(defense).fillna(0.0)
        z["qv"] = z["qb_epa"] - mu + z["d"]
        qb = z.groupby("passer_player_id")["qv"].mean().reindex(qbs, fill_value=0.0)
        qb *= qn / (qn + prior_db)
        z["q"] = z["passer_player_id"].map(qb).fillna(0.0)
        z["dv"] = mu + z["q"] - z["qb_epa"]
        defense = z.groupby("defteam")["dv"].mean().reindex(defs, fill_value=0.0)
        defense *= dn / (dn + defense_prior_db)
    return qb, defense, mu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2017,2018,2019,2022,2023,2024,2025")
    p.add_argument("--prior-db", type=float, default=120.0)
    p.add_argument("--defense-prior-db", type=float, default=250.0)
    p.add_argument("--min-future-db", type=int, default=10)
    args = p.parse_args()

    rows = []
    for season in [int(x) for x in args.seasons.split(",")]:
        x = load_season(season)
        for week in sorted(int(w) for w in x["week"].dropna().unique() if w >= 2):
            hist = x.loc[x["week"] < week].copy()
            cur = x.loc[x["week"] == week].copy()
            qb, defense, mu = ratings_before_week(hist, args.prior_db, args.defense_prior_db)
            raw = hist.groupby("passer_player_id")["qb_epa"].mean()
            db = hist.groupby("passer_player_id").size()
            for pid, g in cur.groupby("passer_player_id"):
                pid = str(pid)
                if pid not in qb.index or len(g) < args.min_future_db:
                    continue
                opp = str(g["defteam"].mode().iloc[0])
                actual = float(g["qb_epa"].mean())
                rows.append({
                    "season": season, "week": week, "passer_player_id": pid,
                    "prior_dropbacks": int(db.get(pid, 0)),
                    "actual_next_qb_epa": actual,
                    "raw_prior_qb_epa": float(raw.get(pid, np.nan)),
                    "oa_qb_ability": float(qb.get(pid, np.nan)),
                    "oa_matchup_forecast": mu + float(qb.get(pid, 0.0)) - float(defense.get(opp, 0.0)),
                })

    out = pd.DataFrame(rows).dropna()
    def score(pred: str) -> tuple[float, float, float]:
        err = out[pred] - out["actual_next_qb_epa"]
        return (
            float(out[[pred, "actual_next_qb_epa"]].corr().iloc[0, 1]),
            float(err.abs().mean()),
            float(np.sqrt((err ** 2).mean())),
        )
    print(f"Observations: {len(out)}")
    for col in ["raw_prior_qb_epa", "oa_qb_ability", "oa_matchup_forecast"]:
        corr, mae, rmse = score(col)
        print(f"{col:24s} corr={corr:+.5f}  MAE={mae:.5f}  RMSE={rmse:.5f}")
    print("\nBy prior-dropback bucket:")
    out["db_bucket"] = pd.cut(out["prior_dropbacks"], [-1, 49, 99, 199, 399, 10_000])
    for bucket, g in out.groupby("db_bucket", observed=True):
        if len(g) < 20:
            continue
        raw_corr = g[["raw_prior_qb_epa", "actual_next_qb_epa"]].corr().iloc[0, 1]
        oa_corr = g[["oa_matchup_forecast", "actual_next_qb_epa"]].corr().iloc[0, 1]
        print(f"{str(bucket):16s} n={len(g):4d} raw_corr={raw_corr:+.4f} oa_corr={oa_corr:+.4f}")


if __name__ == "__main__":
    main()
