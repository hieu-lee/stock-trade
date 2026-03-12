from __future__ import annotations

import numpy as np
import pandas as pd


def attach_forward_targets(daily_prices: pd.DataFrame, monthly_features: pd.DataFrame, holding_days: int, target_return: float) -> pd.DataFrame:
    daily = daily_prices.sort_values(["symbol", "date"]).copy()
    price_column = "adj_close" if "adj_close" in daily.columns else "close"
    daily["entry_price"] = daily.groupby("symbol")[price_column].shift(-1)
    daily["exit_price"] = daily.groupby("symbol")[price_column].shift(-(holding_days + 1))
    daily["exit_date"] = daily.groupby("symbol")["date"].shift(-(holding_days + 1))
    daily["forward_return_20d"] = np.where(
        daily["entry_price"] > 0,
        (daily["exit_price"] / daily["entry_price"]) - 1.0,
        np.nan,
    )
    daily["target_hit"] = (daily["forward_return_20d"] >= target_return).astype(float)

    labels = daily[["symbol", "date", "entry_price", "exit_price", "exit_date", "forward_return_20d", "target_hit"]]
    panel = monthly_features.merge(labels, on=["symbol", "date"], how="left")
    panel["is_trainable"] = panel["forward_return_20d"].notna()
    return panel
