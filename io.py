from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "", "nan", "none"}


def clean_bool(value, default: bool = False) -> bool:
    """Parse provider/CSV booleans without treating the string 'False' as true."""
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write and read back a CSV before atomically promoting it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    verified = pd.read_csv(temporary)
    if len(verified) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"CSV verification failed for {path}")
    os.replace(temporary, path)

