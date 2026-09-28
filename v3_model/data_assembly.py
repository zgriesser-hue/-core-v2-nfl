"""
data_assembly - turn your weekly data sources into the exact shapes
predict_2026.predict_week needs.

This is the piece that was missing: predict_2026.py does the modeling
math, but needs team_game_history / previous_season_priors / matchups
already in specific shapes. This module builds those from data you
already have (or can export) each week:

    - team_game_history  <- team_game_efficiency-style CSV, updated
                             through last week's games (same shape as
                             the historical file used for calibration)
    - a schedule/matchup file for the upcoming week (home/away teams,
      neutral_site)
    - a running market-history log (one row per team-game, the market
      home margin implied for that game, plus that game's own pred_v30
      at the time - needed to build the lagged EWM5 signal)

PERSONNEL IS NOT WIRED YET (see module docstring in predict_2026.py
and prior discussion): build_matchups below sets home_def_loss /
away_def_loss to 0.0 for every game, which makes personnel_adjustment
zero - i.e. this currently runs pred_v33 (V3.0 + market), not the full
pred_v36 (V3.0 + personnel + market). Wire in a real def_loss_pvar
source and drop the 0.0 defaults once PVAR is un-shelved.
"""

from __future__ import annotations

import pandas as pd

from . import market, personnel


def load_team_game_history(path: str, through_season: int, through_week: int) -> pd.DataFrame:
    """
    Load a team_game_efficiency-shaped CSV and keep only games strictly
    before (through_season, through_week) - i.e. games actually played
    so far. `path` should point at your continuously-updated file (the
    same one used for the historical calibration, just with this
    season's completed weeks appended).

    Required columns: season, week, team, opponent, plays,
    epa_per_play, def_epa_per_play.
    """
    df = pd.read_csv(path)
    mask = (df["season"] < through_season) | (
        (df["season"] == through_season) & (df["week"] < through_week)
    )
    return df[mask].reset_index(drop=True)


def build_market_signal_table(market_history_path: str) -> pd.DataFrame:
    """
    Build each team's lagged EWM5 market-disagreement signal from a
    running market-history log.

    `market_history_path` should point at a CSV with one row per
    completed game: season, week, game_id, home_team, away_team,
    market_home_margin, pred_v30 (the model's own pregame projection
    for that game, logged at the time - needed so the signal reflects
    market-vs-model disagreement, not just the raw market line).

    Returns one row per (season, team, week) with market_signal_ewm5
    (that team's own lagged signal, safe to use as an input to a game
    in that (season, week) since it only reflects PRIOR weeks) and
    prior_games_observed (how many of that team's earlier games this
    season already fed the signal - 0 for a team's first tracked game,
    which is the only case where a null/zero signal is expected rather
    than a sign of missing data).
    """
    games = pd.read_csv(market_history_path)
    team_signal = market.build_team_market_signal(games)
    result = market.add_lagged_ewm_signal(team_signal, span=5)
    result["prior_games_observed"] = result.groupby(["season", "team"]).cumcount()
    return result


def build_matchups(
    schedule_path: str,
    season: int,
    week: int,
    market_signal_table: pd.DataFrame | None = None,
    personnel_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the `matchups` DataFrame predict_2026.predict_week expects
    for one week's slate.

    `schedule_path` should point at a CSV with at least: season, week,
    game_id, home_team, away_team, neutral_site (bool).

    `market_signal_table` is the output of build_market_signal_table
    (optional - if not supplied, every team's market signal defaults
    to 0.0, meaning "no prior disagreement", which is the right
    fallback for a team's first tracked game).

    PROVENANCE, not just values: alongside each market signal, this
    returns home/away_market_signal_provenance ("observed" - a real
    prior-week signal fed it, or "no_prior_games" - team's first
    tracked game this season, a genuine zero, not missing data). If
    `market_signal_table` is supplied but a team/week combination
    isn't found in it at all (a join or data problem, not "no prior
    games"), provenance is "missing_lookup_failed" - that case must
    NOT be silently treated the same as a real zero.

    Personnel columns (home_def_loss/away_def_loss) are set to 0.0
    with provenance "pvar_not_wired" for every game - see module
    docstring. This is a placeholder status, not "checked and zero."
    """
    sched = pd.read_csv(schedule_path)
    wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()

    if market_signal_table is not None:
        sig = market_signal_table[
            (market_signal_table["season"] == season) & (market_signal_table["week"] == week)
        ][["team", "market_signal_ewm5", "prior_games_observed"]]

        for side in ("home", "away"):
            side_sig = sig.rename(columns={
                "team": f"{side}_team",
                "market_signal_ewm5": f"{side}_market_signal_ewm5",
                "prior_games_observed": f"{side}_prior_games_observed",
            })
            wk = wk.merge(side_sig, on=f"{side}_team", how="left")

        for side in ("home", "away"):
            found = wk[f"{side}_prior_games_observed"].notna()
            genuine_zero = found & (wk[f"{side}_prior_games_observed"] == 0)
            missing = ~found  # team/week not in the signal table at all

            wk[f"{side}_market_signal_provenance"] = pd.Series(pd.NA, index=wk.index, dtype="object")
            wk.loc[found & ~genuine_zero, f"{side}_market_signal_provenance"] = "observed"
            wk.loc[genuine_zero, f"{side}_market_signal_provenance"] = "no_prior_games"
            wk.loc[missing, f"{side}_market_signal_provenance"] = "missing_lookup_failed"

            wk[f"{side}_market_signal_ewm5"] = wk[f"{side}_market_signal_ewm5"].fillna(0.0)
            wk = wk.drop(columns=[f"{side}_prior_games_observed"])
    else:
        for side in ("home", "away"):
            wk[f"{side}_market_signal_ewm5"] = 0.0
            wk[f"{side}_market_signal_provenance"] = "no_signal_table_supplied"

    # Native V3 personnel input. The weekly player audit is aggregated to
    # defensive exposure only; QB/offense remain separate model components.
    if personnel_audit is not None:
        wk = personnel.add_personnel_to_matchups(wk, personnel_audit)
    else:
        wk["home_def_loss"] = 0.0
        wk["away_def_loss"] = 0.0
        wk["home_personnel_provenance"] = "pvar_audit_not_supplied"
        wk["away_personnel_provenance"] = "pvar_audit_not_supplied"

    if "neutral_site" not in wk.columns:
        wk["neutral_site"] = False

    return wk[
        [
            "season", "week", "game_id", "home_team", "away_team", "neutral_site",
            "home_def_loss", "away_def_loss",
            "home_personnel_provenance", "away_personnel_provenance",
            "home_market_signal_ewm5", "away_market_signal_ewm5",
            "home_market_signal_provenance", "away_market_signal_provenance",
        ]
    ]
