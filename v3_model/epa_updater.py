"""
epa_updater - the validated (not-margin-based) in-season updater,
chosen per your decision to go with option 2: ship the best-tested
EPA-based split updater (Z_STANDARDIZED_ASYM, weighted nested MAE
10.5443 on the real 1,584-game historical sample) rather than
Core V2's margin-residual mechanism.

WHY THIS, AND WHAT IT DOES NOT CLAIM
--------------------------------------
Three genuinely different EPA-based candidates were tested in the
same nested-LOSO structure V3.7Q used:

    V3.0 authoritative control (mystery historical generator)  10.3909
    Q026 reimplementation (raw EPA, asymmetric defense)        10.5237
    Z_STANDARDIZED_ASYM (z-scored, this module)                 10.5443
    Z_STANDARDIZED_OA_ASYM (z-scored + opponent-adjusted)       10.5846
    FIXED_PRESEASON (no update at all)                          10.9435

None of the three EPA-based candidates beat V3.0's control - that
remains true, and this module does not pretend otherwise. What IS
true: all three clearly beat not updating at all (~0.4 MAE), and this
one (Z_STANDARDIZED_ASYM) is statistically indistinguishable from the
best-performing reimplementation (0.02 MAE apart - noise by your own
interpretation scale) while having a decisive practical advantage the
raw-EPA version doesn't: it works on ANY consistent starting scale,
including the real 2026 Andy/Drew points-scale prior, without
inventing a conversion factor. That's why this one, not the
marginally-lower-on-paper raw-EPA version, is what's wired into
production_v3.py.

HOW IT AVOIDS INVENTING A CONVERSION
--------------------------------------
The observed weekly EPA values are standardized (z-scored) against
their OWN cross-league distribution that week - a unit-free quantity,
never used at its raw EPA magnitude. The prior (whatever scale it's
in - EPA for historical previous-season priors, points for the real
2026 Andy/Drew decomposition) is separately standardized against ITS
OWN distribution. The two z-scores are blended with an asymmetric
shrinkage weight (per-observation-count, offense and defense weighted
differently - off_prior_games=8, def_prior_games=20, reused from the
V3.7Q "Q026" config rather than refit here), and the blended z-score
is scaled back using the PRIOR's OWN population standard deviation -
never an externally invented number. The function works identically
whether the prior is in EPA units or points; it never needs to know
which.

THIS IS STILL NOT A CLAIM OF VALIDATED IMPROVEMENT OVER V3.0
--------------------------------------------------------------
It's the best-available non-margin option, chosen deliberately over
"no update" and over Core V2's margin-residual mechanism (rejected on
principle, not on a test result). If a future test finds something
that beats V3.0's control, this should be replaced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFF_PRIOR_GAMES = 8.0
DEF_PRIOR_GAMES = 20.0


def _zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    mean = series.mean()
    std = series.std(ddof=0)
    z = (series - mean) / std if std > 0 else series * 0.0
    return z, mean, std


def compute_observed_epa(
    team_game_history: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """
    Season-to-date raw (not opponent-adjusted - see module docstring
    on why opponent adjustment tested worse here) offensive and
    defensive EPA per team, from games strictly before `week`.

    Returns [team, observed_offense, observed_defense, games_played].
    Defense is sign-flipped so higher is always better, matching
    every other module in this package.
    """
    sg = team_game_history[team_game_history["season"] == season]
    history = sg[sg["week"] < week]

    rows = []
    for team, g in history.groupby("team"):
        plays = g["plays"].sum()
        if plays == 0:
            continue
        rows.append(
            {
                "team": team,
                "observed_offense": np.average(g["epa_per_play"], weights=g["plays"]),
                "observed_defense": -np.average(g["def_epa_per_play"], weights=g["plays"]),
                "games_played": len(g),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["team", "observed_offense", "observed_defense", "games_played"])
    return pd.DataFrame(rows)


def update_split_state(
    prior: pd.DataFrame,
    team_game_history: pd.DataFrame,
    season: int,
    week: int,
    off_prior_games: float = OFF_PRIOR_GAMES,
    def_prior_games: float = DEF_PRIOR_GAMES,
) -> pd.DataFrame:
    """
    Blend `prior` (any consistent scale - EPA or points) with this
    season's observed EPA so far, via the z-score bridge described in
    the module docstring.

    Parameters
    ----------
    prior : DataFrame [team, offense_strength, defense_strength] -
        the team's state before this season's games (e.g. the Andy/
        Drew decomposition for 2026, or a previous-season EPA prior
        for historical testing). Whatever scale `offense_strength`/
        `defense_strength` are in, the output is in that same scale.
    team_game_history : DataFrame [season, week, team, opponent,
        plays, epa_per_play, def_epa_per_play] - this season's
        completed games. Pass already-filtered history as a matter
        of practice (see rating.build_pregame_state's leakage
        assertion for why); this function also filters internally by
        `week` as defense-in-depth.

    Returns
    -------
    DataFrame [team, offense_strength, defense_strength], same shape
    and scale as `prior`.
    """
    prior = prior.set_index("team")
    teams = list(prior.index)

    prior_off_z, prior_off_mean, prior_off_std = _zscore(prior["offense_strength"])
    prior_def_z, prior_def_mean, prior_def_std = _zscore(prior["defense_strength"])

    observed = compute_observed_epa(team_game_history, season, week).set_index("team")

    if len(observed) == 0:
        return prior.reset_index()[["team", "offense_strength", "defense_strength"]]

    obs_off_z, _, _ = _zscore(observed["observed_offense"])
    obs_def_z, _, _ = _zscore(observed["observed_defense"])

    rows = []
    for team in teams:
        if team not in observed.index:
            off_strength = prior.loc[team, "offense_strength"]
            def_strength = prior.loc[team, "defense_strength"]
        else:
            n = observed.loc[team, "games_played"]
            off_z = (off_prior_games * prior_off_z.loc[team] + n * obs_off_z.loc[team]) / (
                off_prior_games + n
            )
            def_z = (def_prior_games * prior_def_z.loc[team] + n * obs_def_z.loc[team]) / (
                def_prior_games + n
            )
            off_strength = off_z * prior_off_std + prior_off_mean
            def_strength = def_z * prior_def_std + prior_def_mean
        rows.append(
            {"team": team, "offense_strength": off_strength, "defense_strength": def_strength}
        )

    return pd.DataFrame(rows)
