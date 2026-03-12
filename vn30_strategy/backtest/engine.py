from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vn30_strategy.config import StrategyConfig


@dataclass(slots=True)
class BacktestResult:
    trades: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics: dict


def run_monthly_backtest(
    scored_panel: pd.DataFrame,
    config: StrategyConfig,
    regime_threshold: float,
    rank_threshold: float,
    max_positions: int,
) -> BacktestResult:
    panel = scored_panel.copy()
    panel = panel.sort_values(["date", "rank_probability"], ascending=[True, False])
    panel["selected"] = False

    monthly_records: list[dict] = []
    trade_rows: list[pd.DataFrame] = []
    round_trip_cost = (config.commission_bps + config.slippage_bps) * 2.0 / 10_000.0

    for date, month_df in panel.groupby("date", sort=True):
        regime_probability = float(month_df["regime_probability"].iloc[0])
        candidates = month_df.loc[month_df["rank_probability"] >= rank_threshold].copy()
        candidates = candidates.head(max_positions)
        traded = regime_probability >= regime_threshold and not candidates.empty

        if traded:
            candidates["selected"] = True
            candidates["net_forward_return"] = candidates["forward_return_20d"] - round_trip_cost
            portfolio_return = candidates["net_forward_return"].mean()
            trade_rows.append(candidates)
        else:
            portfolio_return = 0.0

        monthly_records.append(
            {
                "date": date,
                "regime_probability": regime_probability,
                "trade_count": int(len(candidates) if traded else 0),
                "portfolio_return": float(portfolio_return),
                "target_hit": float(portfolio_return >= config.target_return),
                "is_traded": bool(traded),
            }
        )

    monthly_returns = pd.DataFrame(monthly_records).sort_values("date").reset_index(drop=True)
    monthly_returns["equity_curve"] = (1.0 + monthly_returns["portfolio_return"]).cumprod()
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    metrics = _compute_metrics(monthly_returns, config.target_return)
    return BacktestResult(trades=trades, monthly_returns=monthly_returns, metrics=metrics)


def _compute_metrics(monthly_returns: pd.DataFrame, target_return: float) -> dict:
    traded = monthly_returns.loc[monthly_returns["is_traded"]].copy()
    returns = monthly_returns["portfolio_return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0

    metrics = {
        "months_total": int(len(monthly_returns)),
        "months_traded": int(monthly_returns["is_traded"].sum()),
        "monthly_hit_rate": float(traded["target_hit"].mean()) if not traded.empty else 0.0,
        "average_monthly_return": float(traded["portfolio_return"].mean()) if not traded.empty else 0.0,
        "median_monthly_return": float(traded["portfolio_return"].median()) if not traded.empty else 0.0,
        "pass_80pct_requirement": bool((traded["target_hit"].mean() if not traded.empty else 0.0) >= 0.80),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "cagr_like": float(equity.iloc[-1] ** (12 / max(len(monthly_returns), 1)) - 1) if not equity.empty else 0.0,
        "precision_target_10pct": float(traded["target_hit"].mean()) if not traded.empty else 0.0,
        "target_return": target_return,
    }

    yearly_frame = monthly_returns.assign(year=monthly_returns["date"].dt.year)
    yearly = yearly_frame.groupby("year").agg(
        yearly_return=("portfolio_return", lambda x: float((1.0 + x).prod() - 1.0)),
        traded_months=("is_traded", "sum"),
    )
    yearly_hits = yearly_frame.loc[yearly_frame["is_traded"]].groupby("year")["target_hit"].mean()
    yearly["hit_rate"] = yearly.index.to_series().map(yearly_hits).fillna(0.0)
    metrics["yearly_breakdown"] = yearly.reset_index().to_dict("records")
    return metrics
