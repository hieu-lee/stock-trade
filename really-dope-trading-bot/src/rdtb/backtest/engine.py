from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.portfolio.actions import build_daily_actions
from rdtb.portfolio.optimizer import PolicyParameters, default_policy
from rdtb.utils import annualized_return, max_drawdown, sharpe_ratio


@dataclass(slots=True)
class BacktestResult:
    metrics: dict[str, float | int | dict[str, float] | bool]
    equity_curve: pd.DataFrame
    actions: pd.DataFrame
    closed_trades: pd.DataFrame


def run_backtest(
    scored_panel: pd.DataFrame,
    config: TradingBotConfig,
    policy: PolicyParameters | None = None,
    initial_cash: float | None = None,
) -> BacktestResult:
    panel = scored_panel.copy()
    if panel.empty:
        raise ValueError("A scored panel is required for backtesting.")
    policy = policy or default_policy(config)
    initial_cash = float(initial_cash if initial_cash is not None else config.initial_cash)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = sorted(panel["date"].unique())
    if len(dates) < 2:
        raise ValueError("Backtesting requires at least two market dates.")

    available_cash = initial_cash
    settled_holdings = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date"])
    pending_buys = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date", "settle_date"])
    pending_cash = pd.DataFrame(columns=["symbol", "amount", "settle_date", "execution_date"])
    actions_log: list[dict] = []
    closed_trades: list[dict] = []
    equity_log = [
        {
            "date": pd.Timestamp(dates[0]),
            "equity": initial_cash,
            "cash": available_cash,
            "pending_cash": 0.0,
            "gross_exposure": 0.0,
            "position_count": 0,
        }
    ]
    calendar_dates = [pd.Timestamp(date) for date in dates]

    for current_date, next_date in zip(dates[:-1], dates[1:]):
        settled_holdings, pending_buys, pending_cash, available_cash = _settle_pending_assets(
            current_date=pd.Timestamp(current_date),
            settled_holdings=settled_holdings,
            pending_buys=pending_buys,
            pending_cash=pending_cash,
            available_cash=available_cash,
        )
        decision_holdings = _aggregate_decision_holdings(settled_holdings, pending_buys)
        current_frame = panel.loc[panel["date"] == current_date].copy()
        next_frame = panel.loc[panel["date"] == next_date].copy()
        decision_bundle = build_daily_actions(
            scored_snapshot=current_frame,
            holdings=decision_holdings,
            cash=available_cash,
            config=config,
            policy=policy,
        )
        settled_holdings, pending_buys, pending_cash, available_cash, executed_actions, closed = _execute_actions(
            actions=decision_bundle.actions,
            settled_holdings=settled_holdings,
            pending_buys=pending_buys,
            pending_cash=pending_cash,
            available_cash=available_cash,
            next_date=pd.Timestamp(next_date),
            execution_prices=next_frame.set_index("symbol")["open"].to_dict(),
            config=config,
            calendar_dates=calendar_dates,
        )
        actions_log.extend(executed_actions)
        closed_trades.extend(closed)
        mark_prices = next_frame.set_index("symbol")["close"].to_dict()
        position_values = _mark_position_value(settled_holdings, pending_buys, mark_prices)
        receivable_cash = float(pending_cash["amount"].sum()) if not pending_cash.empty else 0.0
        equity = available_cash + receivable_cash + position_values
        gross_exposure = 0.0 if equity <= 0 else position_values / equity
        equity_log.append(
            {
                "date": pd.Timestamp(next_date),
                "equity": equity,
                "cash": available_cash,
                "pending_cash": receivable_cash,
                "gross_exposure": gross_exposure,
                "position_count": int(_aggregate_decision_holdings(settled_holdings, pending_buys)["symbol"].nunique()),
            }
        )

    equity_curve = pd.DataFrame(equity_log).sort_values("date").reset_index(drop=True)
    equity_curve["return_1d"] = equity_curve["equity"].pct_change().fillna(0.0)
    actions_frame = pd.DataFrame(actions_log)
    if actions_frame.empty:
        actions_frame = pd.DataFrame(
            columns=[
                "decision_date",
                "execution_date",
                "symbol",
                "action",
                "quantity",
                "execution_price",
                "notional",
                "fees",
                "cash_after",
            ]
        )
    closed_frame = pd.DataFrame(closed_trades)
    metrics = _compute_metrics(equity_curve, actions_frame, closed_frame, config)
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, actions=actions_frame, closed_trades=closed_frame)


def _execute_actions(
    actions: pd.DataFrame,
    settled_holdings: pd.DataFrame,
    pending_buys: pd.DataFrame,
    pending_cash: pd.DataFrame,
    available_cash: float,
    next_date: pd.Timestamp,
    execution_prices: dict[str, float],
    config: TradingBotConfig,
    calendar_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, list[dict], list[dict]]:
    book = settled_holdings.copy()
    pending_buy_book = pending_buys.copy()
    pending_cash_book = pending_cash.copy()
    logs: list[dict] = []
    closed: list[dict] = []
    buy_fee_rate = config.buy_transaction_fee_bps / 10_000.0
    sell_fee_rate = config.sell_transaction_fee_bps / 10_000.0

    for row in actions.itertuples(index=False):
        symbol = str(row.symbol)
        action = str(row.action)
        if action == "HOLD":
            continue
        if symbol not in execution_prices:
            continue
        price = float(execution_prices[symbol])
        if price <= 0:
            continue
        current_idx = book.index[book["symbol"] == symbol].tolist()
        current_qty = int(book.loc[current_idx[0], "quantity"]) if current_idx else 0

        if action in {"BUY", "ADD"}:
            desired_qty = int(row.quantity)
            affordable_qty = _affordable_quantity(
                cash=available_cash,
                price=price,
                fee_rate=buy_fee_rate,
                desired_qty=desired_qty,
                lot_size=config.lot_size,
            )
            if affordable_qty <= 0:
                continue
            notional = affordable_qty * price
            fees = notional * buy_fee_rate
            available_cash -= notional + fees
            settle_date = _settlement_date(next_date, calendar_dates, config.buy_settlement_days)
            pending_buy_row = pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "quantity": affordable_qty,
                        "avg_cost": (notional + fees) / affordable_qty,
                        "buy_date": next_date.strftime("%Y-%m-%d"),
                        "settle_date": settle_date,
                    }
                ]
            )
            pending_buy_book = pending_buy_row if pending_buy_book.empty else pd.concat([pending_buy_book, pending_buy_row], ignore_index=True)
            logs.append(
                {
                    "decision_date": pd.Timestamp(row.date),
                    "execution_date": next_date,
                    "settle_date": settle_date,
                    "symbol": symbol,
                    "action": action,
                    "quantity": affordable_qty,
                    "execution_price": price,
                    "notional": notional,
                    "fees": fees,
                    "cash_after": available_cash,
                }
            )
            continue

        sell_qty = current_qty if action == "EXIT" else min(int(row.quantity), current_qty)
        if sell_qty <= 0:
            continue
        notional = sell_qty * price
        fees = notional * sell_fee_rate
        proceeds = notional - fees
        settle_date = _settlement_date(next_date, calendar_dates, config.sell_settlement_days)
        pending_cash_row = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "amount": proceeds,
                    "settle_date": settle_date,
                    "execution_date": next_date,
                }
            ]
        )
        pending_cash_book = pending_cash_row if pending_cash_book.empty else pd.concat([pending_cash_book, pending_cash_row], ignore_index=True)
        avg_cost = float(book.loc[current_idx[0], "avg_cost"]) if current_idx else price
        pnl_pct = price / avg_cost - 1.0 if avg_cost > 0 else 0.0
        remaining = current_qty - sell_qty
        if current_idx:
            if remaining <= 0:
                book = book.drop(index=current_idx[0]).reset_index(drop=True)
            else:
                book.loc[current_idx[0], "quantity"] = remaining
        closed.append(
            {
                "symbol": symbol,
                "execution_date": next_date,
                "action": action,
                "quantity": sell_qty,
                "entry_cost": avg_cost,
                "exit_price": price,
                "pnl_pct": pnl_pct,
            }
        )
        logs.append(
            {
                "decision_date": pd.Timestamp(row.date),
                "execution_date": next_date,
                "settle_date": settle_date,
                "symbol": symbol,
                "action": action,
                "quantity": sell_qty,
                "execution_price": price,
                "notional": notional,
                "fees": fees,
                "cash_after": available_cash,
            }
        )

    book = book.sort_values("symbol").reset_index(drop=True) if not book.empty else book
    pending_buy_book = pending_buy_book.sort_values(["settle_date", "symbol"]).reset_index(drop=True) if not pending_buy_book.empty else pending_buy_book
    pending_cash_book = pending_cash_book.sort_values(["settle_date", "symbol"]).reset_index(drop=True) if not pending_cash_book.empty else pending_cash_book
    return book, pending_buy_book, pending_cash_book, float(available_cash), logs, closed


def _affordable_quantity(cash: float, price: float, fee_rate: float, desired_qty: int, lot_size: int = 100) -> int:
    max_qty = int(np.floor(cash / max(price * (1.0 + fee_rate), 1e-9)))
    target = min(desired_qty, max_qty)
    if target <= 0:
        return 0
    return int(np.floor(target / lot_size) * lot_size)


def _settle_pending_assets(
    current_date: pd.Timestamp,
    settled_holdings: pd.DataFrame,
    pending_buys: pd.DataFrame,
    pending_cash: pd.DataFrame,
    available_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    book = settled_holdings.copy()
    pending_buy_book = pending_buys.copy()
    pending_cash_book = pending_cash.copy()

    if not pending_cash_book.empty:
        ready_cash = pending_cash_book.loc[pending_cash_book["settle_date"] <= current_date].copy()
        if not ready_cash.empty:
            available_cash += float(ready_cash["amount"].sum())
            pending_cash_book = pending_cash_book.loc[pending_cash_book["settle_date"] > current_date].reset_index(drop=True)

    if not pending_buy_book.empty:
        ready_buys = pending_buy_book.loc[pending_buy_book["settle_date"] <= current_date].copy()
        if not ready_buys.empty:
            for row in ready_buys.itertuples(index=False):
                current_idx = book.index[book["symbol"] == row.symbol].tolist()
                if current_idx:
                    existing = book.loc[current_idx[0]]
                    total_qty = int(existing["quantity"]) + int(row.quantity)
                    weighted_cost = ((float(existing["avg_cost"]) * int(existing["quantity"])) + (float(row.avg_cost) * int(row.quantity))) / max(total_qty, 1)
                    book.loc[current_idx[0], "quantity"] = total_qty
                    book.loc[current_idx[0], "avg_cost"] = weighted_cost
                    book.loc[current_idx[0], "buy_date"] = min(str(existing["buy_date"]), str(row.buy_date))
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "symbol": row.symbol,
                                "quantity": row.quantity,
                                "avg_cost": row.avg_cost,
                                "buy_date": row.buy_date,
                            }
                        ]
                    )
                    book = new_row if book.empty else pd.concat([book, new_row], ignore_index=True)
            pending_buy_book = pending_buy_book.loc[pending_buy_book["settle_date"] > current_date].reset_index(drop=True)

    book = book.sort_values("symbol").reset_index(drop=True) if not book.empty else book
    return book, pending_buy_book, pending_cash_book, float(available_cash)


def _aggregate_decision_holdings(settled_holdings: pd.DataFrame, pending_buys: pd.DataFrame) -> pd.DataFrame:
    settled = settled_holdings.copy()
    if not settled.empty:
        settled["sellable_quantity"] = settled["quantity"]
    else:
        settled = pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"])

    pending = pending_buys.copy()
    if not pending.empty:
        pending["sellable_quantity"] = 0
    else:
        pending = pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date", "settle_date"])

    settled_slice = settled[["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"]]
    pending_slice = pending[["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"]]
    if settled_slice.empty and pending_slice.empty:
        combined = pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"])
    elif settled_slice.empty:
        combined = pending_slice.reset_index(drop=True)
    elif pending_slice.empty:
        combined = settled_slice.reset_index(drop=True)
    else:
        combined = pd.concat([settled_slice, pending_slice], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"])
    combined["cost_notional"] = combined["quantity"] * combined["avg_cost"]
    aggregated = (
        combined.groupby("symbol", as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            sellable_quantity=("sellable_quantity", "sum"),
            cost_notional=("cost_notional", "sum"),
            buy_date=("buy_date", "min"),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    aggregated["avg_cost"] = aggregated["cost_notional"] / aggregated["quantity"].replace(0, np.nan)
    return aggregated[["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"]]


def _mark_position_value(settled_holdings: pd.DataFrame, pending_buys: pd.DataFrame, mark_prices: dict[str, float]) -> float:
    total = 0.0
    for frame in [settled_holdings, pending_buys]:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            total += float(row["quantity"]) * float(mark_prices.get(row["symbol"], row["avg_cost"]))
    return total


def _settlement_date(execution_date: pd.Timestamp, calendar_dates: list[pd.Timestamp], settlement_days: int) -> pd.Timestamp:
    if execution_date not in calendar_dates:
        calendar_dates = sorted(set(calendar_dates + [execution_date]))
    index = calendar_dates.index(execution_date)
    target_index = min(index + max(int(settlement_days), 0), len(calendar_dates) - 1)
    return pd.Timestamp(calendar_dates[target_index])


def _compute_metrics(
    equity_curve: pd.DataFrame,
    actions: pd.DataFrame,
    closed_trades: pd.DataFrame,
    config: TradingBotConfig,
) -> dict[str, float | int | dict[str, float] | bool]:
    yearly_returns: dict[str, float] = {}
    for year, frame in equity_curve.groupby(equity_curve["date"].dt.year):
        if len(frame) < 2:
            continue
        yearly_returns[str(year)] = float(frame["equity"].iloc[-1] / frame["equity"].iloc[0] - 1.0)
    win_rate = float((closed_trades["pnl_pct"] > 0).mean()) if not closed_trades.empty else 0.0
    metrics: dict[str, float | int | dict[str, float] | bool] = {
        "total_return": float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0),
        "annualized_return": float(annualized_return(equity_curve["equity"])),
        "max_drawdown": float(max_drawdown(equity_curve["equity"])),
        "sharpe": float(sharpe_ratio(equity_curve["return_1d"])),
        "trade_count": int(len(actions)),
        "win_rate": win_rate,
        "avg_gross_exposure": float(equity_curve["gross_exposure"].mean()),
        "avg_position_count": float(equity_curve["position_count"].mean()),
        "yearly_returns": yearly_returns,
        "contract_years_passed": bool(
            yearly_returns
            and all(
                year not in {str(config.paper_trade_year)}
                and yearly_returns[year] >= config.deployment_min_year_return
                for year in yearly_returns
                if int(year) in set(config.final_test_years)
            )
        ),
        "contract_drawdown_passed": float(max_drawdown(equity_curve["equity"])) >= config.deployment_max_drawdown,
    }
    return metrics
