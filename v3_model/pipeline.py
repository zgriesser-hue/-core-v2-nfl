"""
V3.6 - Pure V3 integration.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def predict_v36(pred_v30,personnel_adjustment,market_adjustment,neutral_hfa_adjustment)->np.ndarray:
    pred_v30=np.asarray(pred_v30,dtype=float); personnel_adjustment=np.asarray(personnel_adjustment,dtype=float); market_adjustment=np.asarray(market_adjustment,dtype=float); neutral_hfa_adjustment=np.asarray(neutral_hfa_adjustment,dtype=float)
    if not (pred_v30.shape==personnel_adjustment.shape==market_adjustment.shape==neutral_hfa_adjustment.shape):
        raise ValueError("All four inputs to predict_v36 must have the same shape")
    return pred_v30+personnel_adjustment+market_adjustment+neutral_hfa_adjustment
