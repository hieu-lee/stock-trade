from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ProgressCallback = Callable[[str, float | None], None]


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean_denominator = denominator.replace(0, np.nan)
    return numerator / clean_denominator


def annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    if periods <= 0:
        return 0.0
    gross = 1.0 + total_return
    if gross <= 0:
        return -1.0
    return float(gross ** (periods_per_year / periods) - 1.0)


def report_progress(callback: ProgressCallback | None, message: str, progress: float | None = None) -> None:
    if callback is None:
        return
    normalized_progress = None if progress is None else min(max(float(progress), 0.0), 1.0)
    callback(message, normalized_progress)


def subprogress(
    callback: ProgressCallback | None,
    start: float,
    end: float,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def wrapped(message: str, progress: float | None = None) -> None:
        if progress is None:
            callback(message, None)
            return
        callback(message, start + (end - start) * min(max(float(progress), 0.0), 1.0))

    return wrapped


def _json_default(value):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Unsupported json value: {type(value)!r}")
