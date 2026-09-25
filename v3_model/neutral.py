"""
V3.4 - true-neutral-site HFA removal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def neutral_hfa_adjustment(is_neutral_site: pd.Series | np.ndarray, hfa: float) -> np.ndarray:
    is_neutral=np.asarray(is_neutral_site).astype(bool)
    return np.where(is_neutral,-hfa,0.0)
