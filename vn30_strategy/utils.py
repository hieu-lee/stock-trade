from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def snake_case(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def flatten_columns(columns: pd.Index) -> list[str]:
    flattened: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [snake_case(item) for item in col if item not in (None, "", " ")]
            flattened.append("_".join(part for part in parts if part))
        else:
            flattened.append(snake_case(str(col)))
    return flattened


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator
