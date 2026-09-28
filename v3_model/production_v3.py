"""
production_v3 - the shippable V3 pipeline.

REVISION HISTORY (why this file changed)
------------------------------------------
The first version of this module updated team ratings using Core
V2's own final-margin-residual rule. That was a real mistake, caught
before shipping: the whole reason V3 exists is to NOT use final-score
residual as the primary mechanism for estimating team quality (per
the development handoff, Section 18 vs. the project's founding
rationale). Reusing it here would have shipped exactly the mechanism
V3 was built to replace.

This version instead uses `epa_updater.update_split_state` - a
genuinely EPA-based (never touches final margin) split updater,
chosen deliberately as the best-tested non-margin option after three
different EPA-based candidates were run through the same nested-LOSO
validation V3.7Q used. None of the three beat V3.0's frozen historical
control; this is the option 2 was chosen over further searching (see
epa_updater.py's module docstring for the comparison numbers and the
full reasoning). It is shipped as "beats no update, doesn't yet match
the historical control," not as a validated win.

STILL TRUE FROM THE ORIGINAL DESIGN
--------------------------------------
The persistent-margin formula still doesn't need any EPA<->points
conversion, because offense_strength + defense_strength always stays
internally consistent with whatever scale the state started in
(points, from the real Andy/Drew prior) - epa_updater.py's z-score
bridge is exactly what preserves that property while still letting
the split move based on real in-season performance. See that module's
docstring for how.

V3.2 (personnel) and V3.3 (market signal) still layer on top exactly
as before - both are points-scale adjustments that never depended on
this updater question either way.
"""

from __future__ import annotations

import pandas as pd

from . import config, epa_updater, market, neutral, personnel

# Your own V3Config's documented default (handoff Section 20), reused
# as-is rather than fit or invented. Confirm or refit once you have
# real 2026 results to check it against.
EXPERIMENTAL_HFA_DEFAULT = 2.20


def persistent_margin(
    home_offense: float, home_defense: float, away_offense: float, away_defense: float
) -> float:
    """
    (home_off + home_def) - (away_off + away_def). Works whether the
    state is the untouched preseason prior (where this collapses to
    overall_prior_home - overall_prior_away) or a state that's been
    through epa_updater.update_split_state, where offense and defense
    may have drifted apart based on real performance - which is
    exactly how this updater is able to move the projected margin
    without ever touching final score.
    """
    return (home_offense + home_defense) - (away_offense + away_defense)


def advance_state(
    state: pd.DataFrame,
    team_game_history: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Thin wrapper around epa_updater.update_split_state, named for
    what it does in this pipeline: take last week's
    [team, offense_strength, defense_strength] and this season's
    completed games, return the state to use for `week`.
    """
    return epa_updater.update_split_state(state, team_game_history, season, week)


def predict_week_points_scale(
    state: pd.DataFrame,
    matchups: pd.DataFrame,
    hfa: float = EXPERIMENTAL_HFA_DEFAULT,
) -> pd.DataFrame:
    """
    Predict pred_v3_lite for a week's matchups, entirely in points -
    no EPA conversion anywhere in this path.

    Parameters
    ----------
    state : DataFrame [team, offense_strength, defense_strength] -
        the current points-scale state for every team. Start of
        season: the real Andy/Drew decomposition
        (preseason_priors_2026.load_2026_preseason_priors, renaming
        its `offense_strength`/`defense_strength` columns to match).
        Later weeks: whatever `advance_state` has updated it to.
    matchups : DataFrame [season, week, game_id, home_team, away_team,
        neutral_site, home_def_loss, away_def_loss,
        home_market_signal_ewm5, away_market_signal_ewm5] - same shape
        predict_2026.predict_week uses, so the same data_assembly
        helpers work for both paths.

    Returns
    -------
    matchups plus [persistent_margin, personnel_adjustment,
    market_adjustment, neutral_hfa_adjustment, pred_v3_lite].
    """
    s = state.set_index("team")

    # When assembly supplies provenance, fail closed on incomplete personnel
    # coverage. Missing injury data must never become a healthy zero.
    valid_personnel_provenance = {"player_audit_native_v3", "covered_no_defensive_losses"}
    for side in ("home", "away"):
        provenance_col = f"{side}_personnel_provenance"
        if provenance_col in matchups.columns:
            bad = ~matchups[provenance_col].isin(valid_personnel_provenance)
            if bad.any():
                bad_values = sorted(matchups.loc[bad, provenance_col].astype(str).unique())
                raise ValueError(
                    f"Native V3 personnel coverage is incomplete for {side}: {bad_values}"
                )

    market_cal = market.MarketCalibration(
        beta=config.EXPERIMENTAL_MARKET_BETA, span=config.EXPERIMENTAL_MARKET_SIGNAL_SPAN
    )

    out_rows = []
    for row in matchups.itertuples(index=False):
        home = s.loc[row.home_team]
        away = s.loc[row.away_team]
        persistent = persistent_margin(
            home["offense_strength"], home["defense_strength"],
            away["offense_strength"], away["defense_strength"],
        )

        personnel_adj = float(
            personnel.continuous_personnel_adjustment(
                [row.home_def_loss], [row.away_def_loss]
            ).iloc[0]
        )
        signal_diff = row.home_market_signal_ewm5 - row.away_market_signal_ewm5
        market_adj = float(market_cal.adjustment([signal_diff])[0])
        neutral_adj = float(neutral.neutral_hfa_adjustment([row.neutral_site], hfa)[0])

        pred = persistent + hfa + personnel_adj + market_adj + neutral_adj

        out_rows.append(
            {
                **row._asdict(),
                "persistent_margin": persistent,
                "personnel_adjustment": personnel_adj,
                "market_adjustment": market_adj,
                "neutral_hfa_adjustment": neutral_adj,
                "pred_v3_lite": pred,
            }
        )

    return pd.DataFrame(out_rows)
