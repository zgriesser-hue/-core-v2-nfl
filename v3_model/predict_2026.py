"""
predict_2026 - the live entry point.

Wires rating.py + spread.py + personnel.py + market.py + neutral.py +
pipeline.py together into one function: given this week's completed
games and the matchups you want a spread for, return pred_v36 for each.

This is the piece that was missing before: everything else in this
package was validated logic, but nothing tied it into something you
could call from your Streamlit app with "here's what's happened,
here's next week's slate" and get numbers back.

USAGE
-----
    from v3_model.predict_2026 import predict_week

    predictions = predict_week(
        team_game_history=history_df,   # season, week, team, opponent,
                                          # plays, epa_per_play, def_epa_per_play
                                          # for every completed game this season
        previous_season_priors=priors_df,  # from rating.build_previous_season_priors
        matchups=matchups_df,            # season, week, home_team, away_team,
                                          # neutral_site, home_def_loss, away_def_loss,
                                          # home_market_signal_ewm5, away_market_signal_ewm5
    )
    # predictions has one row per matchup with pred_v30 / pred_v36 / etc.

WHAT THIS DOES NOT DO
----------------------
- Pull live data. `team_game_history`, `matchups`, injury exposure and
  market signals are still your responsibility to assemble each week
  (from your PBP pipeline, injury reports, and market-history log) -
  this function only does the math once you hand it that data.
- Decide your bet. pred_v36 is a projected home margin, not a bet
  sizing or edge-detection function - the notebook's own ATS testing
  found large model/market disagreement was NOT a reliable signal
  (see NFL_MODEL_2026_MASTER_CHECKPOINT.txt), and that finding was
  about Core v1/v2, not re-tested here for V3.6.
- Claim to be production-ready. Every coefficient comes from
  config.py's EXPERIMENTAL_* constants - see that module's docstring
  for exactly which are unresolved (the O/D updater and 2026 HFA/
  conversion, per V3.7Q) vs. fit on real authoritative data but still
  short of the project's own validation bar. This function computes
  pred_v36 correctly given those inputs; it does not certify the
  inputs are right.
"""

from __future__ import annotations

import pandas as pd

from . import config, market, neutral, personnel, rating, spread


def build_pregame_states_for_week(
    team_game_history: pd.DataFrame,
    previous_season_priors: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Pregame offense/defense state for every team entering (season, week),
    using the EXPERIMENTAL_* constants from config.py.
    """
    history = team_game_history[
        (team_game_history["season"] == season) & (team_game_history["week"] < week)
    ]
    return rating.build_pregame_state(
        team_game=history,
        season=season,
        week=week,
        previous_season_priors=previous_season_priors,
        off_alpha=config.EXPERIMENTAL_OFF_ALPHA,
        off_strength=config.EXPERIMENTAL_OFF_STRENGTH,
        def_alpha=config.EXPERIMENTAL_DEF_ALPHA,
        def_strength=config.EXPERIMENTAL_DEF_STRENGTH,
    )


def predict_week(
    team_game_history: pd.DataFrame,
    previous_season_priors: pd.DataFrame,
    matchups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict pred_v36 for every matchup in `matchups`.

    Parameters
    ----------
    team_game_history : DataFrame [season, week, team, opponent, plays,
        epa_per_play, def_epa_per_play] - every completed game this
        season (and, for build_previous_season_priors, last season too).
    previous_season_priors : DataFrame [season, team, prev_offense,
        prev_defense] - from rating.build_previous_season_priors.
    matchups : DataFrame with one row per game you want a spread for:
        [season, week, game_id, home_team, away_team, neutral_site,
         home_def_loss, away_def_loss,
         home_market_signal_ewm5, away_market_signal_ewm5]
        The two market_signal_ewm5 columns are each team's OWN lagged
        signal entering this game (build with market.add_lagged_ewm_signal
        on that team's history BEFORE this week - never include this
        game's own market line).

    Returns
    -------
    DataFrame: matchups plus [pred_v30, personnel_adjustment,
    market_adjustment, neutral_hfa_adjustment, pred_v36].
    """
    out_rows = []

    # Build states once per (season, week) present in matchups, not once per row.
    state_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for season, week in matchups[["season", "week"]].drop_duplicates().itertuples(
        index=False
    ):
        state_cache[(season, week)] = build_pregame_states_for_week(
            team_game_history, previous_season_priors, season, week
        )

    personnel_cal = personnel.PersonnelCalibration(
        threshold=config.EXPERIMENTAL_PERSONNEL_THRESHOLD,
        adjustment_points=config.EXPERIMENTAL_PERSONNEL_ADJUSTMENT_POINTS,
    )
    market_cal = market.MarketCalibration(
        beta=config.EXPERIMENTAL_MARKET_BETA, span=config.EXPERIMENTAL_MARKET_SIGNAL_SPAN
    )
    spread_cal = spread.SpreadCalibration(
        hfa=config.EXPERIMENTAL_HFA, beta=config.EXPERIMENTAL_BETA
    )

    for row in matchups.itertuples(index=False):
        state = state_cache[(row.season, row.week)].set_index("team")
        home = state.loc[row.home_team]
        away = state.loc[row.away_team]

        sdiff = spread.strength_diff(
            pd.Series([home["latent_offense"]]),
            pd.Series([home["latent_defense"]]),
            pd.Series([away["latent_offense"]]),
            pd.Series([away["latent_defense"]]),
        )
        pred_v30 = float(spread_cal.predict(sdiff)[0])

        personnel_adj = float(
            personnel_cal.adjustment(
                pd.Series([row.home_def_loss]), pd.Series([row.away_def_loss])
            )[0]
        )

        signal_diff = row.home_market_signal_ewm5 - row.away_market_signal_ewm5
        market_adj = float(market_cal.adjustment([signal_diff])[0])

        neutral_adj = float(
            neutral.neutral_hfa_adjustment([row.neutral_site], config.EXPERIMENTAL_HFA)[
                0
            ]
        )

        pred_v36 = pred_v30 + personnel_adj + market_adj + neutral_adj

        out_rows.append(
            {
                **row._asdict(),
                "pred_v30": pred_v30,
                "personnel_adjustment": personnel_adj,
                "market_adjustment": market_adj,
                "neutral_hfa_adjustment": neutral_adj,
                "pred_v36": pred_v36,
            }
        )

    return pd.DataFrame(out_rows)
