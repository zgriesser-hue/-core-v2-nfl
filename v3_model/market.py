"""
V3.3 - lagged market-disagreement signal.

Extracted from V3.ipynb, cells "V3.3b - CORRECT MARKET SIGN + BUILD
STRICT LAGGED MARKET DISAGREEMENT SIGNAL" and "V3.3c - LOSO LAGGED
MARKET-STRENGTH TEST".

CRITICAL invariant (preserved here): the CURRENT game's market line
is NEVER used as an input to that same game's prediction. Every
signal below is built with `.shift(1)` before any rolling/EWM
aggregation, so a team's signal entering week W only reflects
market disagreement from weeks < W.

Definition
----------
For a completed historical game:
    market_minus_v30 = market_home_margin - pred_v30
    home team's signal for that game = market_minus_v30
    away team's signal for that game = -market_minus_v30

The notebook tested several lag windows (previous game, prior-3,
prior-5, season-to-date, EWM span 3, EWM span 5) and its LOSO
selection picked EWM(span=5) in every single fold - so that
STRUCTURAL choice is frozen here as the default. The regression
coefficient (beta) mapping the signal differential to a points
adjustment still needs a full-history fit (see `fit_market_beta`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_team_market_signal(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert one row per historical game into one row per team-game
    market-disagreement signal (`market_signal_game`), before any
    lagging. `games` needs columns: season, week, game_id, home_team,
    away_team, market_home_margin, pred_v30.
    """
    g = games.copy()
    g["market_minus_v30"] = g["market_home_margin"] - g["pred_v30"]

    home = g[["season", "week", "game_id", "home_team", "market_minus_v30"]].rename(
        columns={"home_team": "team"}
    )
    home["market_signal_game"] = home["market_minus_v30"]

    away = g[["season", "week", "game_id", "away_team", "market_minus_v30"]].rename(
        columns={"away_team": "team"}
    )
    away["market_signal_game"] = -away["market_minus_v30"]

    team = pd.concat(
        [
            home[["season", "week", "game_id", "team", "market_signal_game"]],
            away[["season", "week", "game_id", "team", "market_signal_game"]],
        ],
        ignore_index=True,
    )

    return team.sort_values(["season", "team", "week", "game_id"]).reset_index(
        drop=True
    )


def add_lagged_ewm_signal(team: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """
    Add `market_signal_ewm{span}`, the frozen structural choice
    (EWM span=5). shift(1) happens BEFORE the EWM so the current
    game's own market line never leaks into its own signal.
    """
    out = team.copy()
    grp = out.groupby(["season", "team"], group_keys=False)
    col = f"market_signal_ewm{span}"
    out[col] = grp["market_signal_game"].transform(
        lambda s: s.shift(1).ewm(span=span, adjust=False).mean()
    )
    return out


@dataclass(frozen=True)
class MarketCalibration:
    beta: float
    span: int = 5

    def adjustment(self, signal_diff: pd.Series | np.ndarray) -> np.ndarray:
        return self.beta * np.asarray(signal_diff)


def fit_market_beta(
    v30_residual: pd.Series, signal_diff: pd.Series, span: int = 5
) -> MarketCalibration:
    """
    No-intercept fit of v30_residual ~ signal_diff (zero prior
    disagreement -> zero adjustment, exactly as the notebook
    requires). Run on full history for the production beta.
    """
    x = signal_diff.to_numpy()
    denom = float(np.sum(x**2))
    if denom == 0:
        raise ValueError("No non-null market signal in training data.")
    beta = float(np.sum(x * v30_residual.to_numpy()) / denom)
    return MarketCalibration(beta=beta, span=span)
