from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig


@dataclass(slots=True)
class TransactionReplayResult:
    holdings: pd.DataFrame
    available_cash: float
    pending_cash_total: float
    pending_buy_quantity: int
    notes: list[str]
    processed_transactions: pd.DataFrame


def normalize_transactions(transactions: pd.DataFrame | None) -> pd.DataFrame:
    if transactions is None or transactions.empty:
        return pd.DataFrame(columns=["date", "symbol", "action", "quantity", "price"])
    frame = transactions.copy()
    rename_map = {
        "trade_date": "date",
        "transaction_date": "date",
        "qty": "quantity",
        "side": "action",
        "type": "action",
    }
    frame = frame.rename(columns=rename_map)
    for column in ["date", "symbol", "action", "quantity", "price"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.upper().str.strip()
    frame["action"] = frame["action"].fillna("").astype(str).str.upper().str.strip()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0).astype(int)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    valid_actions = {"BUY", "SELL"}
    frame = frame.loc[
        frame["date"].notna()
        & frame["symbol"].ne("")
        & frame["action"].isin(valid_actions)
        & (frame["quantity"] > 0)
        & (frame["price"] > 0)
    ].copy()
    return frame.sort_values(["date", "symbol", "action"]).reset_index(drop=True)


def replay_transactions(
    transactions: pd.DataFrame | None,
    starting_cash: float,
    config: TradingBotConfig,
    as_of_date: pd.Timestamp,
    trading_calendar: list[pd.Timestamp] | pd.Series,
) -> TransactionReplayResult:
    normalized = normalize_transactions(transactions)
    as_of_date = pd.Timestamp(as_of_date).normalize()
    available_cash = float(starting_cash)
    settled_holdings = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date"])
    pending_buys = pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date", "settle_date"])
    pending_cash = pd.DataFrame(columns=["symbol", "amount", "settle_date", "execution_date"])
    notes: list[str] = []
    processed_rows: list[dict] = []

    for row in normalized.itertuples(index=False):
        trade_date = pd.Timestamp(row.date).normalize()
        if trade_date > as_of_date:
            continue

        settled_holdings, pending_buys, pending_cash, available_cash = _settle_pending_assets(
            current_date=trade_date,
            settled_holdings=settled_holdings,
            pending_buys=pending_buys,
            pending_cash=pending_cash,
            available_cash=available_cash,
        )

        symbol = str(row.symbol)
        quantity = int(row.quantity)
        price = float(row.price)

        if row.action == "BUY":
            notional = quantity * price
            fees = notional * (config.buy_transaction_fee_bps / 10_000.0)
            total_cost = notional + fees
            if available_cash + 1e-9 < total_cost:
                message = f"Skipped BUY `{symbol}` on {trade_date.date()} because settled cash was insufficient."
                notes.append(message)
                processed_rows.append(
                    {
                        "date": trade_date,
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": quantity,
                        "price": price,
                        "fees": fees,
                        "settle_date": pd.NaT,
                        "status": "REJECTED",
                        "note": message,
                    }
                )
                continue
            available_cash -= total_cost
            settle_date = _settlement_date(trade_date, trading_calendar, config.buy_settlement_days)
            buy_row = pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "quantity": quantity,
                        "avg_cost": total_cost / quantity,
                        "buy_date": trade_date.strftime("%Y-%m-%d"),
                        "settle_date": settle_date,
                    }
                ]
            )
            pending_buys = buy_row if pending_buys.empty else pd.concat([pending_buys, buy_row], ignore_index=True)
            processed_rows.append(
                {
                    "date": trade_date,
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "fees": fees,
                    "settle_date": settle_date,
                    "status": "EXECUTED",
                    "note": "",
                }
            )
            continue

        sellable_qty = _sellable_quantity(settled_holdings, symbol)
        if sellable_qty < quantity:
            message = (
                f"Skipped SELL `{symbol}` on {trade_date.date()} because only {sellable_qty} settled shares were sellable."
            )
            notes.append(message)
            processed_rows.append(
                {
                    "date": trade_date,
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": quantity,
                    "price": price,
                    "fees": 0.0,
                    "settle_date": pd.NaT,
                    "status": "REJECTED",
                    "note": message,
                }
            )
            continue

        notional = quantity * price
        fees = notional * (config.sell_transaction_fee_bps / 10_000.0)
        proceeds = notional - fees
        settled_holdings = _reduce_settled_holding(settled_holdings, symbol, quantity)
        settle_date = _settlement_date(trade_date, trading_calendar, config.sell_settlement_days)
        cash_row = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "amount": proceeds,
                    "settle_date": settle_date,
                    "execution_date": trade_date,
                }
            ]
        )
        pending_cash = cash_row if pending_cash.empty else pd.concat([pending_cash, cash_row], ignore_index=True)
        processed_rows.append(
            {
                "date": trade_date,
                "symbol": symbol,
                "action": "SELL",
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "settle_date": settle_date,
                "status": "EXECUTED",
                "note": "",
            }
        )

    settled_holdings, pending_buys, pending_cash, available_cash = _settle_pending_assets(
        current_date=as_of_date,
        settled_holdings=settled_holdings,
        pending_buys=pending_buys,
        pending_cash=pending_cash,
        available_cash=available_cash,
    )
    holdings = _aggregate_decision_holdings(settled_holdings, pending_buys)
    pending_cash_total = float(pending_cash["amount"].sum()) if not pending_cash.empty else 0.0
    pending_buy_quantity = int(pending_buys["quantity"].sum()) if not pending_buys.empty else 0
    processed = pd.DataFrame(processed_rows)
    return TransactionReplayResult(
        holdings=holdings,
        available_cash=float(available_cash),
        pending_cash_total=pending_cash_total,
        pending_buy_quantity=pending_buy_quantity,
        notes=notes,
        processed_transactions=processed,
    )


def _sellable_quantity(settled_holdings: pd.DataFrame, symbol: str) -> int:
    if settled_holdings.empty:
        return 0
    rows = settled_holdings.loc[settled_holdings["symbol"] == symbol, "quantity"]
    return int(rows.sum()) if not rows.empty else 0


def _reduce_settled_holding(settled_holdings: pd.DataFrame, symbol: str, quantity: int) -> pd.DataFrame:
    book = settled_holdings.copy()
    rows = book.index[book["symbol"] == symbol].tolist()
    if not rows:
        return book
    row_idx = rows[0]
    current_qty = int(book.loc[row_idx, "quantity"])
    remaining = current_qty - quantity
    if remaining <= 0:
        book = book.drop(index=row_idx).reset_index(drop=True)
    else:
        book.loc[row_idx, "quantity"] = remaining
    return book


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
                rows = book.index[book["symbol"] == row.symbol].tolist()
                if rows:
                    row_idx = rows[0]
                    existing = book.loc[row_idx]
                    total_qty = int(existing["quantity"]) + int(row.quantity)
                    weighted_cost = ((float(existing["avg_cost"]) * int(existing["quantity"])) + (float(row.avg_cost) * int(row.quantity))) / max(total_qty, 1)
                    book.loc[row_idx, "quantity"] = total_qty
                    book.loc[row_idx, "avg_cost"] = weighted_cost
                    book.loc[row_idx, "buy_date"] = min(str(existing["buy_date"]), str(row.buy_date))
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


def _settlement_date(
    execution_date: pd.Timestamp,
    trading_calendar: list[pd.Timestamp] | pd.Series,
    settlement_days: int,
) -> pd.Timestamp:
    execution_date = pd.Timestamp(execution_date).normalize()
    calendar = sorted({pd.Timestamp(item).normalize() for item in list(trading_calendar)})
    if not calendar:
        calendar = [execution_date]
    if execution_date not in calendar:
        calendar = sorted(set(calendar + [execution_date]))
    index = calendar.index(execution_date)
    target_index = index + max(int(settlement_days), 0)
    while target_index >= len(calendar):
        next_business = pd.bdate_range(start=calendar[-1] + pd.offsets.BDay(1), periods=5)
        calendar = sorted(set(calendar + [pd.Timestamp(item).normalize() for item in next_business]))
    return pd.Timestamp(calendar[target_index]).normalize()
