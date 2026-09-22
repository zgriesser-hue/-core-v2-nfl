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

DEFAULT_MATURE_EPA_WEIGHT = 0.35
DEFAULT_PRIOR_GAMES = 4.0
# Historical calibration from the nested 2017-19/2023-25 experiment:
# OA EPA matchup difference -> Core team-strength points.
DEFAULT_EPA_POINTS_PER_NET = 45.0


def fetch_nflverse_pbp(season: int, timeout: int = 60) -> pd.DataFrame:
    """Load the current nflverse season PBP without adding a parquet dependency."""
    url = NFLVERSE_PBP_CSV.format(season=int(season))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    raw = gzip.decompress(response.content)
    usecols = ["season", "week", "season_type", "game_id", "posteam", "defteam", "epa"]
    return pd.read_csv(BytesIO(raw), usecols=usecols, low_memory=False)


def build_team_game_epa(pbp: pd.DataFrame, season: int, through_week: int) -> pd.DataFrame:
    """Aggregate regular-season offensive EPA/play for completed prior weeks."""
    x = pbp.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["epa"] = pd.to_numeric(x["epa"], errors="coerce")
    x = x.loc[
        x["season"].eq(int(season))
        & x["week"].le(int(through_week))
        & x["season_type"].astype(str).eq("REG")
    ].copy()
    x["team"] = x["posteam"].map(normalize_team)
    x["opponent"] = x["defteam"].map(normalize_team)
    x = x.dropna(subset=["game_id", "team", "opponent", "epa"])
    x = x.loc[x["team"].ne("NAN") & x["opponent"].ne("NAN")]
    return (
        x.groupby(["season", "week", "game_id", "team", "opponent"], as_index=False)
        .agg(plays=("epa", "size"), epa_per_play=("epa", "mean"))
    )


def solve_oa_epa(
    team_games: pd.DataFrame,
    shrink_games: float = 4.0,
    max_iterations: int = 15,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Solve observed EPA ~= mean + offense(team) - defense(opponent)."""
    h = team_games[["team", "opponent", "epa_per_play"]].dropna().copy()
    if h.empty:
        return pd.DataFrame(columns=["team", "oa_offense", "oa_defense", "oa_net", "games"])

    mu = float(h["epa_per_play"].mean())
    teams = sorted(set(h["team"]) | set(h["opponent"]))
    offense = pd.Series(0.0, index=teams, dtype=float)
    defense = pd.Series(0.0, index=teams, dtype=float)
    off_n = h.groupby("team").size().reindex(teams, fill_value=0)
    def_n = h.groupby("opponent").size().reindex(teams, fill_value=0)
    off_shrink = off_n / (off_n + float(shrink_games))
    def_shrink = def_n / (def_n + float(shrink_games))

    for _ in range(max_iterations):
        old_off = offense.copy()
        old_def = defense.copy()

        tmp = h.copy()
        tmp["def_faced"] = tmp["opponent"].map(defense).fillna(0.0)
        tmp["off_value"] = tmp["epa_per_play"] - mu + tmp["def_faced"]
        offense = tmp.groupby("team")["off_value"].mean().reindex(teams, fill_value=0.0)
        offense = offense * off_shrink
        offense -= offense.mean()

        tmp["off_faced"] = tmp["team"].map(offense).fillna(0.0)
        tmp["def_value"] = mu + tmp["off_faced"] - tmp["epa_per_play"]
        defense = tmp.groupby("opponent")["def_value"].mean().reindex(teams, fill_value=0.0)
        defense = defense * def_shrink
        defense -= defense.mean()

        change = max(
            float(np.max(np.abs(offense.values - old_off.values))),
            float(np.max(np.abs(defense.values - old_def.values))),
        )
        if change < tolerance:
            break

    games = h.groupby("team").size().reindex(teams, fill_value=0).astype(int)
    return pd.DataFrame(
        {
            "team": teams,
            "oa_offense": offense.reindex(teams).values,
            "oa_defense": defense.reindex(teams).values,
            "oa_net": (offense + defense).reindex(teams).values,
            "games": games.values,
        }
    )


def build_shadow_ratings(
    structural_ratings: pd.DataFrame,
    oa_ratings: pd.DataFrame,
    mature_epa_weight: float = DEFAULT_MATURE_EPA_WEIGHT,
    prior_games: float = DEFAULT_PRIOR_GAMES,
    epa_points_per_net: float = DEFAULT_EPA_POINTS_PER_NET,
) -> pd.DataFrame:
    """Blend structural Core prior with opponent-adjusted current-season EPA."""
    required = {"team", "rating_before_performance"}
    missing = required - set(structural_ratings.columns)
    if missing:
        raise ValueError(f"Structural ratings missing columns: {sorted(missing)}")

    out = structural_ratings.copy()
    out["team"] = out["team"].map(normalize_team)
    oa = oa_ratings.copy()
    oa["team"] = oa["team"].map(normalize_team)
    out = out.merge(
        oa[["team", "oa_offense", "oa_defense", "oa_net", "games"]],
        on="team",
        how="left",
        validate="one_to_one",
    )
    for column in ["oa_offense", "oa_defense", "oa_net"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["games"] = pd.to_numeric(out["games"], errors="coerce").fillna(0).astype(int)

    out["epa_weight"] = (
        float(mature_epa_weight) * out["games"] / (out["games"] + float(prior_games))
    )
    out["core_weight"] = 1.0 - out["epa_weight"]
    out["epa_implied_rating"] = out["oa_net"] * float(epa_points_per_net)
    out["shadow_team_rating"] = (
        out["core_weight"] * out["rating_before_performance"]
        + out["epa_weight"] * out["epa_implied_rating"]
    )
    out["shadow_minus_structural"] = (
        out["shadow_team_rating"] - out["rating_before_performance"]
    )
    out["shadow_rank"] = out["shadow_team_rating"].rank(method="min", ascending=False).astype(int)
    out["shadow_model"] = "OA_EPA_RAMP_V1"
    out["mature_epa_weight"] = float(mature_epa_weight)
    out["prior_games_strength"] = float(prior_games)
    out["epa_points_per_net"] = float(epa_points_per_net)
    return out.sort_values("team").reset_index(drop=True)
