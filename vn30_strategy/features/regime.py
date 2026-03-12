from __future__ import annotations

import pandas as pd


def build_regime_dataset(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "benchmark_ret_20d": "median",
        "benchmark_ret_60d": "median",
        "benchmark_vol_20d": "median",
        "benchmark_distance_ma200": "median",
        "benchmark_drawdown_252d": "median",
        "breadth_above_ma200": "median",
        "breadth_positive_20d": "median",
        "breadth_ret_20d": "median",
        "breadth_ret_60d": "median",
        "ret_20d": "median",
        "ret_60d": "median",
        "volatility_20d": "median",
        "distance_ma200": "median",
        "avg_turnover_20d": "median",
    }
    available = {key: value for key, value in aggregations.items() if key in monthly_panel.columns}
    regime = monthly_panel.groupby("date", as_index=False).agg(available)
    return regime.sort_values("date").reset_index(drop=True)
