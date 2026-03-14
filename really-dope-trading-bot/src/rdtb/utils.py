from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ProgressCallback = Callable[[str, float | None], None]


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def report_progress(callback: ProgressCallback | None, message: str, progress: float | None = None) -> None:
    if callback is not None:
        callback(message, progress)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directories([path.parent])
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = max((len(equity) - 1) / periods_per_year, 1 / periods_per_year)
    return (end / start) ** (1 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_peak = equity.cummax()
    drawdowns = equity / rolling_peak - 1.0
    return float(drawdowns.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    series = returns.dropna()
    if series.empty:
        return 0.0
    std = float(series.std(ddof=0))
    if std == 0:
        return 0.0
    return float(series.mean() / std * np.sqrt(periods_per_year))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.columns = [str(column) for column in copied.columns]
    return copied
