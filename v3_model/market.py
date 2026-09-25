"""
V3.3 - lagged market-disagreement signal.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def build_team_market_signal(games: pd.DataFrame) -> pd.DataFrame:
    g=games.copy(); g["market_minus_v30"]=g["market_home_margin"]-g["pred_v30"]
    home=g[["season","week","game_id","home_team","market_minus_v30"]].rename(columns={"home_team":"team"}); home["market_signal_game"]=home["market_minus_v30"]
    away=g[["season","week","game_id","away_team","market_minus_v30"]].rename(columns={"away_team":"team"}); away["market_signal_game"]=-away["market_minus_v30"]
    team=pd.concat([home[["season","week","game_id","team","market_signal_game"]],away[["season","week","game_id","team","market_signal_game"]]],ignore_index=True)
    return team.sort_values(["season","team","week","game_id"]).reset_index(drop=True)

def add_lagged_ewm_signal(team: pd.DataFrame, span:int=5)->pd.DataFrame:
    out=team.copy(); grp=out.groupby(["season","team"],group_keys=False); col=f"market_signal_ewm{span}"
    out[col]=grp["market_signal_game"].transform(lambda s:s.shift(1).ewm(span=span,adjust=False).mean())
    return out

@dataclass(frozen=True)
class MarketCalibration:
    beta: float
    span: int=5
    def adjustment(self, signal_diff): return self.beta*np.asarray(signal_diff)

def fit_market_beta(v30_residual:pd.Series,signal_diff:pd.Series,span:int=5)->MarketCalibration:
    x=signal_diff.to_numpy(); denom=float(np.sum(x**2))
    if denom==0: raise ValueError("No non-null market signal in training data.")
    beta=float(np.sum(x*v30_residual.to_numpy())/denom)
    return MarketCalibration(beta=beta,span=span)
