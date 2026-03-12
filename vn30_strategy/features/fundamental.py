from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from vn30_strategy.utils import snake_case


def build_fundamental_features(
    monthly_features: pd.DataFrame,
    overviews: pd.DataFrame,
    ratios: pd.DataFrame,
) -> pd.DataFrame:
    frame = monthly_features.copy()
    overview_features = _normalize_overviews(overviews)
    ratio_features = _normalize_ratios(ratios)

    if not overview_features.empty:
        frame = frame.merge(overview_features, on="symbol", how="left")

    if not ratio_features.empty:
        frame = frame.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
        ratio_features = (
            ratio_features.dropna(subset=["report_date"])
            .assign(date_key=pd.to_datetime(ratio_features["report_date"]))
            .sort_values(["date_key", "symbol"])
            .reset_index(drop=True)
        )
        frame = pd.merge_asof(
            frame,
            ratio_features,
            left_on="date",
            right_on="date_key",
            by="symbol",
            direction="backward",
        )
        frame = frame.drop(columns=["date_key"], errors="ignore")

    sector_cols = [col for col in ["icb_name2", "icb_name3", "exchange"] if col in frame.columns]
    if sector_cols:
        frame = pd.get_dummies(frame, columns=sector_cols, dummy_na=True)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _normalize_overviews(overviews: pd.DataFrame) -> pd.DataFrame:
    if overviews.empty:
        return overviews
    cols = ["symbol"]
    cols.extend(col for col in ["issue_share", "charter_capital", "icb_name2", "icb_name3", "exchange"] if col in overviews.columns)
    frame = overviews[cols].drop_duplicates(subset=["symbol"]).copy()
    frame["issue_share"] = pd.to_numeric(frame.get("issue_share"), errors="coerce")
    frame["charter_capital"] = pd.to_numeric(frame.get("charter_capital"), errors="coerce")
    return frame


def _normalize_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    if ratios.empty:
        return ratios
    frame = ratios.copy()
    if "report_date" not in frame.columns:
        return pd.DataFrame()

    value_map = {
        "roe": ["roe"],
        "roa": ["roa"],
        "debt_equity": ["debt_equity"],
        "net_profit_margin": ["net_profit_margin"],
        "financial_leverage": ["financial_leverage"],
        "pe": ["p_e"],
        "pb": ["p_b"],
        "ps": ["p_s"],
        "pcf": ["p_cash_flow"],
        "eps": ["eps"],
        "bvps": ["bvps"],
        "dividend_yield": ["dividend_yield"],
        "market_cap": ["market_cap"],
    }

    selected = pd.DataFrame({"symbol": frame["symbol"], "report_date": pd.to_datetime(frame["report_date"])})
    for feature_name, hints in value_map.items():
        match = _match_column(frame.columns, hints)
        if match:
            selected[feature_name] = pd.to_numeric(frame[match], errors="coerce")

    selected = selected.sort_values(["symbol", "report_date"]).reset_index(drop=True)
    for col in [col for col in selected.columns if col not in {"symbol", "report_date"}]:
        selected[f"{col}_change_4q"] = selected.groupby("symbol")[col].pct_change(4, fill_method=None)
    return selected


def _match_column(columns: Iterable[str], hints: list[str]) -> str | None:
    lowered = {col: snake_case(col) for col in columns}
    for col, normalized in lowered.items():
        if all(hint in normalized for hint in hints):
            return col
    return None
