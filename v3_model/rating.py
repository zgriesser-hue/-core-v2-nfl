"""
V3.0 - prior-anchored latent offense/defense rating solver.

Extracted verbatim (logic unchanged) from V3.ipynb, cell "V3 RESEARCH
- REBUILD FROZEN STATES AFTER COLAB RESTART" (function
`fit_prior_anchored_od`) plus the season/week state-building loop
around it.

Model
-----
For a team-game:

    EPA_per_play(team vs opponent) ~= league_mean
                                       + offense(team)
                                       - defense(opponent)

Solved via weighted least squares across every team-game played so
far this season, PLUS two pseudo-observations per team pulling
offense/defense toward that team's previous-season prior (scaled by
`prior_alpha`), with pseudo-observation weight `prior_strength`.

Only games strictly before the target week are used - this is what
makes the resulting state usable as a *pregame* rating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriorAnchorResult:
    """Latent offense/defense ratings for every team in `priors`."""

    ratings: pd.DataFrame  # columns: team, latent_offense, latent_defense


def fit_prior_anchored_od(
    history: pd.DataFrame,
    priors: pd.DataFrame,
    prior_strength: float = 4.0,
) -> pd.DataFrame:
    """
    Solve latent offense/defense ratings from a team's history plus
    a previous-season prior.

    Parameters
    ----------
    history : DataFrame with columns [team, opponent, epa_per_play, plays]
        One row per team-game played so far (STRICTLY before the
        target week). Each game contributes one row per team
        (i.e. two rows total, one from each side's perspective).
    priors : DataFrame with columns [team, prior_offense, prior_defense]
        Previous-season-derived priors for every team expected to
        appear in `ratings`.
    prior_strength : float
        Pseudo-observation weight given to each team's prior. Higher
        values anchor the state more strongly to the prior and make
        it move more slowly as the season's games accumulate.

    Returns
    -------
    DataFrame with columns [team, latent_offense, latent_defense].
    """
    teams = sorted(priors["team"].unique())
    idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    if len(history) == 0:
        # No current-season observations -> return priors directly.
        return priors[["team", "prior_offense", "prior_defense"]].rename(
            columns={
                "prior_offense": "latent_offense",
                "prior_defense": "latent_defense",
            }
        )

    h = history.copy()

    league_mean = np.average(h["epa_per_play"], weights=h["plays"])

    rows: list[np.ndarray] = []
    y: list[float] = []
    w: list[float] = []

    # Game observations: EPA above league = offense(team) - defense(opponent)
    for r in h.itertuples(index=False):
        if r.team not in idx or r.opponent not in idx or r.plays <= 0:
            continue

        x = np.zeros(2 * n)
        x[idx[r.team]] = 1.0
        x[n + idx[r.opponent]] = -1.0

        rows.append(x)
        y.append(r.epa_per_play - league_mean)
        # Roughly one game's weight around 1 (65 plays is a typical game).
        w.append(r.plays / 65.0)

    # Prior pseudo-observations (one for offense, one for defense per team).
    for r in priors.itertuples(index=False):
        if r.team not in idx:
            continue

        x = np.zeros(2 * n)
        x[idx[r.team]] = 1.0
        rows.append(x)
        y.append(r.prior_offense)
        w.append(prior_strength)

        x = np.zeros(2 * n)
        x[n + idx[r.team]] = 1.0
        rows.append(x)
        y.append(r.prior_defense)
        w.append(prior_strength)

    X = np.vstack(rows)
    y_arr = np.asarray(y)
    w_arr = np.asarray(w)
    sw = np.sqrt(w_arr)

    coef, *_ = np.linalg.lstsq(X * sw[:, None], y_arr * sw, rcond=None)

    return pd.DataFrame(
        {
            "team": teams,
            "latent_offense": coef[:n],
            "latent_defense": coef[n:],
        }
    )


def build_previous_season_priors(
    team_game: pd.DataFrame, season: int
) -> pd.DataFrame:
    """
    Build each team's previous-season offense/defense prior, exactly
    as the notebook's "PREVIOUS-SEASON TEAM PRIORS" step does.

    `team_game` needs columns: season, team, plays, epa_per_play,
    def_epa_per_play. Defense is sign-flipped so that, like offense,
    HIGHER is always better.
    """
    prev = team_game[team_game["season"] == season - 1]

    rows = []
    for team, g in prev.groupby("team"):
        plays = g["plays"].sum()
        if plays > 0:
            off = np.average(g["epa_per_play"], weights=g["plays"])
            defense = -np.average(g["def_epa_per_play"], weights=g["plays"])
        else:
            off = 0.0
            defense = 0.0

        rows.append(
            {
                "season": season,
                "team": team,
                "prev_offense": off,
                "prev_defense": defense,
            }
        )

    return pd.DataFrame(rows)


def build_pregame_state(
    team_game: pd.DataFrame,
    season: int,
    week: int,
    previous_season_priors: pd.DataFrame,
    off_alpha: float,
    off_strength: float,
    def_alpha: float,
    def_strength: float,
) -> pd.DataFrame:
    """
    Build the pregame offense/defense state entering `week` of
    `season`, mirroring the notebook's per-season/week loop. Offense
    and defense are solved separately (each with its own prior
    weight applied to only its own side, the other pinned at 0),
    exactly as the notebook does.

    Returns a DataFrame with columns: team, season, week,
    latent_offense, latent_defense.

    DEFENSE IN DEPTH: this asserts no row in `team_game` is at or
    beyond the target week for this season, even though the caller
    is expected to have already filtered. A caller passing unfiltered
    history here is a leakage bug regardless of whether this function
    happens to filter correctly on its own - the assertion below turns
    that mistake into a loud failure instead of a silent one.
    """
    same_season = team_game[team_game["season"] == season]
    leaked = same_season[same_season["week"] >= week]
    assert leaked.empty, (
        f"build_pregame_state received {len(leaked)} row(s) at or after "
        f"the target week ({season} week {week}) - the caller must filter "
        f"team_game to strictly earlier weeks before calling this function."
    )

    sg = team_game[team_game["season"] == season]
    history = sg[sg["week"] < week].copy()

    base = previous_season_priors[previous_season_priors["season"] == season]

    off_prior = pd.DataFrame(
        {
            "team": base["team"],
            "prior_offense": off_alpha * base["prev_offense"],
            "prior_defense": 0.0,
        }
    )
    off_fit = fit_prior_anchored_od(history, off_prior, off_strength)[
        ["team", "latent_offense"]
    ]

    def_prior = pd.DataFrame(
        {
            "team": base["team"],
            "prior_offense": 0.0,
            "prior_defense": def_alpha * base["prev_defense"],
        }
    )
    def_fit = fit_prior_anchored_od(history, def_prior, def_strength)[
        ["team", "latent_defense"]
    ]

    state = off_fit.merge(def_fit, on="team", how="inner")
    state["season"] = season
    state["week"] = week
    return state
