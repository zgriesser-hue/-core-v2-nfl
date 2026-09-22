from __future__ import annotations

from io import BytesIO
import gzip

import numpy as np
import pandas as pd
import requests

from .projections import normalize_team

NFLVERSE_PBP_CSV = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.csv.gz"
)

QB_USECOLS = [
    "season", "week", "season_type", "game_id", "posteam", "defteam",
    "passer_player_id", "passer_player_name", "qb_epa", "cpoe",
    "play_type", "pass_attempt", "sack",
]


def fetch_qb_pbp(season: int, timeout: int = 60) -> pd.DataFrame:
    """Load fields needed for the dynamic QB experiment."""
    response = requests.get(NFLVERSE_PBP_CSV.format(season=int(season)), timeout=timeout)
    response.raise_for_status()
    raw = gzip.decompress(response.content)
    return pd.read_csv(BytesIO(raw), usecols=QB_USECOLS, low_memory=False)


def build_qb_game_panel(pbp: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    """One row per QB/game using nflfastR QB EPA on dropbacks."""
    x = pbp.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[
        x["season"].eq(int(season))
        & x["week"].le(int(through_week))
        & x["season_type"].astype(str).eq("REG")
    ].copy()
    x["team"] = x["posteam"].map(normalize_team)
    x["opponent"] = x["defteam"].map(normalize_team)
    x["qb_epa"] = pd.to_numeric(x["qb_epa"], errors="coerce")
    x["cpoe"] = pd.to_numeric(x["cpoe"], errors="coerce")
    x["pass_attempt"] = pd.to_numeric(x["pass_attempt"], errors="coerce").fillna(0)
    x["sack"] = pd.to_numeric(x["sack"], errors="coerce").fillna(0)
    x["qb_dropback"] = ((x["pass_attempt"].eq(1)) | x["sack"].eq(1)).astype(int)
    x = x.loc[x["qb_dropback"].eq(1)].dropna(
        subset=["game_id", "team", "opponent", "passer_player_id", "qb_epa"]
    )
    return (
        x.groupby(
            ["season", "week", "game_id", "team", "opponent",
             "passer_player_id", "passer_player_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            dropbacks=("qb_dropback", "size"),
            qb_epa_per_db=("qb_epa", "mean"),
            cpoe=("cpoe", "mean"),
            sacks=("sack", "sum"),
        )
    )


def solve_oa_qb_epa(
    qb_games: pd.DataFrame,
    prior_dropbacks: float = 120.0,
    defense_prior_dropbacks: float = 250.0,
    max_iterations: int = 20,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Solve QB EPA/dropback ~= mean + QB ability - opponent pass defense.

    Shrinkage is by dropbacks, not games. Positive defense means a tougher
    pass defense. This is a shadow/research signal; it is not a points rating.
    """
    h = qb_games.copy()
    if h.empty:
        return pd.DataFrame(columns=[
            "passer_player_id", "passer_player_name", "team", "oa_qb_epa_per_db",
            "raw_qb_epa_per_db", "dropbacks", "games",
        ])
    h["dropbacks"] = pd.to_numeric(h["dropbacks"], errors="coerce").fillna(0.0)
    h["qb_epa_per_db"] = pd.to_numeric(h["qb_epa_per_db"], errors="coerce")
    h = h.loc[h["dropbacks"].gt(0) & h["qb_epa_per_db"].notna()].copy()
    mu = float(np.average(h["qb_epa_per_db"], weights=h["dropbacks"]))
    qbs = sorted(h["passer_player_id"].astype(str).unique())
    defenses = sorted(h["opponent"].astype(str).unique())
    qb = pd.Series(0.0, index=qbs)
    defense = pd.Series(0.0, index=defenses)
    qb_n = h.groupby("passer_player_id")["dropbacks"].sum().reindex(qbs, fill_value=0.0)
    def_n = h.groupby("opponent")["dropbacks"].sum().reindex(defenses, fill_value=0.0)

    for _ in range(max_iterations):
        old_qb, old_def = qb.copy(), defense.copy()
        z = h.copy()
        z["def_faced"] = z["opponent"].map(defense).fillna(0.0)
        z["qb_value"] = z["qb_epa_per_db"] - mu + z["def_faced"]
        qb_num = (z["qb_value"] * z["dropbacks"]).groupby(z["passer_player_id"]).sum()
        qb = qb_num.reindex(qbs, fill_value=0.0) / qb_n.replace(0, np.nan)
        qb = qb.fillna(0.0) * (qb_n / (qb_n + float(prior_dropbacks)))
        qb -= np.average(qb, weights=np.maximum(qb_n, 1.0))

        z["qb_faced"] = z["passer_player_id"].map(qb).fillna(0.0)
        z["def_value"] = mu + z["qb_faced"] - z["qb_epa_per_db"]
        def_num = (z["def_value"] * z["dropbacks"]).groupby(z["opponent"]).sum()
        defense = def_num.reindex(defenses, fill_value=0.0) / def_n.replace(0, np.nan)
        defense = defense.fillna(0.0) * (def_n / (def_n + float(defense_prior_dropbacks)))
        defense -= np.average(defense, weights=np.maximum(def_n, 1.0))

        change = max(
            float(np.max(np.abs(qb.values - old_qb.values))),
            float(np.max(np.abs(defense.values - old_def.values))),
        )
        if change < tolerance:
            break

    meta = (
        h.sort_values(["week", "game_id"])
        .groupby("passer_player_id", as_index=False)
        .agg(
            passer_player_name=("passer_player_name", "last"),
            team=("team", "last"),
            dropbacks=("dropbacks", "sum"),
            games=("game_id", "nunique"),
        )
    )
    raw = (
        h.assign(weighted=lambda d: d["qb_epa_per_db"] * d["dropbacks"])
        .groupby("passer_player_id")
        .agg(weighted=("weighted", "sum"), db=("dropbacks", "sum"))
    )
    raw["raw_qb_epa_per_db"] = raw["weighted"] / raw["db"]
    meta["passer_player_id"] = meta["passer_player_id"].astype(str)
    meta["oa_qb_epa_per_db"] = meta["passer_player_id"].map(qb)
    meta["raw_qb_epa_per_db"] = meta["passer_player_id"].map(raw["raw_qb_epa_per_db"])
    return meta.sort_values(["dropbacks", "oa_qb_epa_per_db"], ascending=[False, False]).reset_index(drop=True)
