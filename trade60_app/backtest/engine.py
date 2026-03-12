from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade60_app.config import Trade60Config
from trade60_app.utils import annualized_return

DEFENSIVE_TREND_THRESHOLD = -0.02
DEFENSIVE_BREADTH_THRESHOLD = 0.40
DEFENSIVE_RET20_THRESHOLD = -0.04
DEFENSIVE_POSITION_FRACTION = 0.60
DEFENSIVE_CASH_FRACTION = 0.55
DEFENSIVE_ALPHA_BOOST = 0.02


@dataclass(slots=True)
class StrategyParameters:
    entry_threshold: float
    entry_quantile: float
    exit_threshold: float
    regime_threshold: float
    max_positions: int
    max_holding_days: int
    min_holding_days: int
    stop_loss_pct: float
    take_profit_pct: float
    hold_alpha_buffer: float = 0.06
    rank_keep_fraction: float = 1.0
    defensive_trim_fraction: float = 0.35
    weak_alpha_trim_fraction: float = 0.5
    profit_trim_fraction: float = 0.5


@dataclass(slots=True)
class BacktestArtifacts:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict


def run_daily_backtest(
    scored_panel: pd.DataFrame,
    config: Trade60Config,
    params: StrategyParameters,
) -> BacktestArtifacts:
    panel = scored_panel.sort_values(["date", "symbol"]).reset_index(drop=True).copy()
    dates = sorted(pd.to_datetime(panel["date"]).unique())
    if len(dates) < 2:
        raise ValueError("At least two trading dates are required for backtesting.")

    grouped = {date: frame.set_index("symbol") for date, frame in panel.groupby("date", sort=True)}
    date_index = {date: index for index, date in enumerate(dates)}
    buy_cost = (config.commission_bps + config.slippage_bps) / 10_000.0
    sell_cost = (config.commission_bps + config.slippage_bps) / 10_000.0

    cash = float(config.initial_budget)
    positions: dict[str, dict] = {}
    closed_trades: list[dict] = []
    equity_records = [
        {
            "date": pd.Timestamp(dates[0]),
            "equity": cash,
            "cash": cash,
            "positions_value": 0.0,
            "strategy_return": 0.0,
            "benchmark_return": 0.0,
            "open_positions": 0,
        }
    ]
    previous_equity = cash

    for idx in range(1, len(dates)):
        signal_date = pd.Timestamp(dates[idx - 1])
        trade_date = pd.Timestamp(dates[idx])
        signal_frame = grouped[signal_date]
        trade_frame = grouped[trade_date]
        blocked_symbols: set[str] = set()
        bought_value = 0.0
        sold_value = 0.0
        pre_trade_equity = _compute_pre_trade_equity(cash, positions, trade_frame)
        regime_probability = float(signal_frame["regime_probability"].iloc[0])
        effective_max_positions, deploy_fraction, alpha_boost = _resolve_exposure_settings(signal_frame, params)
        is_defensive = (
            effective_max_positions < params.max_positions
            or deploy_fraction < 0.999
            or alpha_boost > 0.0
        )
        alpha_floor = _compute_alpha_floor(signal_frame, params, alpha_boost)
        keep_symbols = _select_ranked_keep_symbols(signal_frame, effective_max_positions, params.rank_keep_fraction)

        for symbol, position in list(positions.items()):
            if symbol not in trade_frame.index:
                continue
            signal_row = signal_frame.loc[symbol] if symbol in signal_frame.index else None
            mark_price = float(signal_row["close"]) if signal_row is not None else float(trade_frame.at[symbol, "close"])
            alpha_probability = float(signal_row["alpha_probability"]) if signal_row is not None else float("nan")
            regime_probability_symbol = (
                float(signal_row["regime_probability"]) if signal_row is not None else regime_probability
            )
            holding_days = idx - date_index[pd.Timestamp(position["entry_date"])]
            gross_return = mark_price / float(position["entry_price"]) - 1.0
            sell_reason = None

            if holding_days >= params.max_holding_days:
                sell_reason = "max_hold"
            elif gross_return <= -params.stop_loss_pct:
                sell_reason = "stop_loss"
            elif signal_row is not None and holding_days >= params.min_holding_days:
                if alpha_probability < params.exit_threshold:
                    sell_reason = "alpha_exit"
                elif regime_probability_symbol < params.regime_threshold:
                    sell_reason = "regime_exit"

            if sell_reason is not None:
                cash, sold_value = _execute_sell(
                    cash=cash,
                    sold_value=sold_value,
                    positions=positions,
                    closed_trades=closed_trades,
                    symbol=symbol,
                    quantity=int(position["quantity"]),
                    trade_date=trade_date,
                    exit_price=float(trade_frame.at[symbol, "open"]),
                    sell_cost=sell_cost,
                    holding_days=holding_days,
                    reason=sell_reason,
                )
                blocked_symbols.add(symbol)
                continue

            trim_fraction = 0.0
            trim_reason = None
            if gross_return >= params.take_profit_pct:
                trim_fraction, trim_reason = _choose_trim(
                    trim_fraction,
                    trim_reason,
                    params.profit_trim_fraction,
                    "profit_trim",
                )
            if (
                signal_row is not None
                and holding_days >= params.min_holding_days
                and alpha_probability < (params.exit_threshold + params.hold_alpha_buffer)
            ):
                trim_fraction, trim_reason = _choose_trim(
                    trim_fraction,
                    trim_reason,
                    params.weak_alpha_trim_fraction,
                    "weak_alpha_trim",
                )
            if is_defensive:
                trim_fraction, trim_reason = _choose_trim(
                    trim_fraction,
                    trim_reason,
                    params.defensive_trim_fraction,
                    "defensive_trim",
                )

            trim_quantity = _shares_for_fraction(int(position["quantity"]), trim_fraction)
            if trim_quantity > 0:
                cash, sold_value = _execute_sell(
                    cash=cash,
                    sold_value=sold_value,
                    positions=positions,
                    closed_trades=closed_trades,
                    symbol=symbol,
                    quantity=trim_quantity,
                    trade_date=trade_date,
                    exit_price=float(trade_frame.at[symbol, "open"]),
                    sell_cost=sell_cost,
                    holding_days=holding_days,
                    reason=trim_reason or "partial_trim",
                )
                blocked_symbols.add(symbol)

        active_symbols = [symbol for symbol, position in positions.items() if int(position["quantity"]) > 0]
        if len(active_symbols) > effective_max_positions:
            overflow = len(active_symbols) - effective_max_positions
            ranked_exits = sorted(
                active_symbols,
                key=lambda symbol: (
                    symbol in keep_symbols,
                    float(signal_frame.at[symbol, "composite_score"]) if symbol in signal_frame.index else float("-inf"),
                ),
            )
            for symbol in ranked_exits[:overflow]:
                position = positions.get(symbol)
                if position is None or symbol not in trade_frame.index:
                    continue
                holding_days = idx - date_index[pd.Timestamp(position["entry_date"])]
                cash, sold_value = _execute_sell(
                    cash=cash,
                    sold_value=sold_value,
                    positions=positions,
                    closed_trades=closed_trades,
                    symbol=symbol,
                    quantity=int(position["quantity"]),
                    trade_date=trade_date,
                    exit_price=float(trade_frame.at[symbol, "open"]),
                    sell_cost=sell_cost,
                    holding_days=holding_days,
                    reason="rebalance_exit",
                )
                blocked_symbols.add(symbol)

        if regime_probability >= params.regime_threshold and effective_max_positions > 0:
            candidates = signal_frame.loc[
                (signal_frame["alpha_probability"] >= alpha_floor)
                & ~signal_frame.index.isin(blocked_symbols)
            ].copy()
            candidates = _sort_candidates(candidates)
            current_positions = len(positions)
            open_slots = max(effective_max_positions - current_positions, 0)
            top_up_symbols = [
                symbol for symbol in candidates.index if symbol in positions and symbol in keep_symbols and symbol in trade_frame.index
            ]
            new_symbols = [
                symbol for symbol in candidates.index if symbol not in positions and symbol in trade_frame.index
            ][:open_slots]
            allocation_order = top_up_symbols + new_symbols
            desired_position_count = max(current_positions + len(new_symbols), 1)
            target_position_value = (pre_trade_equity * deploy_fraction) / desired_position_count

            for symbol in allocation_order:
                open_price = float(trade_frame.at[symbol, "open"])
                cost_per_share = open_price * (1.0 + buy_cost)
                target_quantity = int(np.floor(target_position_value / max(cost_per_share, 1e-9)))
                current_quantity = int(positions[symbol]["quantity"]) if symbol in positions else 0
                delta_quantity = target_quantity - current_quantity
                if delta_quantity < 1:
                    continue
                affordable_quantity = int(np.floor(cash / max(cost_per_share, 1e-9)))
                quantity = min(delta_quantity, affordable_quantity)
                if quantity < 1:
                    continue
                cash, bought_value = _execute_buy(
                    cash=cash,
                    bought_value=bought_value,
                    positions=positions,
                    symbol=symbol,
                    quantity=quantity,
                    trade_date=trade_date,
                    open_price=open_price,
                    buy_cost=buy_cost,
                )

        positions_value = 0.0
        for symbol, position in positions.items():
            if symbol not in trade_frame.index:
                continue
            positions_value += float(position["quantity"]) * float(trade_frame.at[symbol, "close"])
        total_equity = cash + positions_value
        benchmark_return = float(trade_frame["benchmark_ret_1d"].dropna().iloc[0]) if "benchmark_ret_1d" in trade_frame else 0.0
        strategy_return = total_equity / previous_equity - 1.0 if previous_equity else 0.0
        equity_records.append(
            {
                "date": trade_date,
                "equity": total_equity,
                "cash": cash,
                "positions_value": positions_value,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "open_positions": len(positions),
                "bought_value": bought_value,
                "sold_value": sold_value,
            }
        )
        previous_equity = total_equity

    last_date = pd.Timestamp(dates[-1])
    last_frame = grouped[last_date]
    if positions:
        final_cash = cash
        final_rows: list[dict] = []
        for symbol, position in list(positions.items()):
            if symbol not in last_frame.index:
                continue
            exit_close = float(last_frame.at[symbol, "close"])
            net_exit_price = exit_close * (1.0 - sell_cost)
            proceeds = float(position["quantity"]) * net_exit_price
            final_cash += proceeds
            holding_days = date_index[last_date] - date_index[pd.Timestamp(position["entry_date"])]
            final_rows.append(
                {
                    "symbol": symbol,
                    "entry_date": position["entry_date"],
                    "exit_date": last_date,
                    "entry_price": float(position["entry_price"]),
                    "exit_price": exit_close,
                    "quantity": int(position["quantity"]),
                    "gross_return": float(exit_close / float(position["entry_price"]) - 1.0),
                    "net_return": float(net_exit_price / float(position["entry_cost_basis"]) - 1.0),
                    "holding_days": holding_days,
                    "exit_reason": "final_liquidation",
                }
            )
        positions.clear()
        closed_trades.extend(final_rows)
        if equity_records:
            last_equity = equity_records[-1]["equity"]
            final_equity = final_cash
            equity_records[-1]["equity"] = final_equity
            equity_records[-1]["cash"] = final_equity
            equity_records[-1]["positions_value"] = 0.0
            equity_records[-1]["open_positions"] = 0
            equity_records[-1]["strategy_return"] = final_equity / max(last_equity, 1e-9) - 1.0

    equity_curve = pd.DataFrame(equity_records).sort_values("date").reset_index(drop=True)
    equity_curve["benchmark_curve"] = (1.0 + equity_curve["benchmark_return"].fillna(0.0)).cumprod() * config.initial_budget
    equity_curve["equity_peak"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["equity_peak"] - 1.0
    trades = pd.DataFrame(closed_trades).sort_values(["exit_date", "symbol"]).reset_index(drop=True) if closed_trades else pd.DataFrame()
    metrics = _compute_metrics(equity_curve, trades, config)
    return BacktestArtifacts(trades=trades, equity_curve=equity_curve, metrics=metrics)


def _compute_pre_trade_equity(cash: float, positions: dict[str, dict], trade_frame: pd.DataFrame) -> float:
    total = float(cash)
    for symbol, position in positions.items():
        if symbol not in trade_frame.index:
            continue
        total += float(position["quantity"]) * float(trade_frame.at[symbol, "open"])
    return total


def _shares_for_fraction(quantity: int, fraction: float) -> int:
    if quantity <= 0 or fraction <= 0:
        return 0
    planned = int(np.floor(quantity * min(max(fraction, 0.0), 1.0)))
    if planned < 1:
        planned = 1
    return min(planned, quantity)


def _execute_sell(
    cash: float,
    sold_value: float,
    positions: dict[str, dict],
    closed_trades: list[dict],
    symbol: str,
    quantity: int,
    trade_date: pd.Timestamp,
    exit_price: float,
    sell_cost: float,
    holding_days: int,
    reason: str,
) -> tuple[float, float]:
    position = positions.get(symbol)
    if position is None or quantity < 1:
        return cash, sold_value
    quantity = min(int(quantity), int(position["quantity"]))
    net_exit_price = exit_price * (1.0 - sell_cost)
    proceeds = quantity * net_exit_price
    cash += proceeds
    sold_value += proceeds
    closed_trades.append(
        {
            "symbol": symbol,
            "entry_date": position["entry_date"],
            "exit_date": trade_date,
            "entry_price": float(position["entry_price"]),
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_return": float(exit_price / float(position["entry_price"]) - 1.0),
            "net_return": float(net_exit_price / float(position["entry_cost_basis"]) - 1.0),
            "holding_days": holding_days,
            "exit_reason": reason,
        }
    )
    remaining_quantity = int(position["quantity"]) - quantity
    if remaining_quantity <= 0:
        positions.pop(symbol, None)
    else:
        position["quantity"] = remaining_quantity
        positions[symbol] = position
    return cash, sold_value


def _execute_buy(
    cash: float,
    bought_value: float,
    positions: dict[str, dict],
    symbol: str,
    quantity: int,
    trade_date: pd.Timestamp,
    open_price: float,
    buy_cost: float,
) -> tuple[float, float]:
    if quantity < 1:
        return cash, bought_value
    cost_per_share = open_price * (1.0 + buy_cost)
    total_cost = quantity * cost_per_share
    gross_cost = quantity * open_price
    cash -= total_cost
    bought_value += gross_cost
    if symbol in positions:
        position = positions[symbol]
        current_quantity = int(position["quantity"])
        new_quantity = current_quantity + quantity
        position["quantity"] = new_quantity
        position["entry_price"] = (
            float(position["entry_price"]) * current_quantity + open_price * quantity
        ) / max(new_quantity, 1)
        position["entry_cost_basis"] = (
            float(position["entry_cost_basis"]) * current_quantity + cost_per_share * quantity
        ) / max(new_quantity, 1)
        position["last_buy_date"] = trade_date
        positions[symbol] = position
    else:
        positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_date": trade_date,
            "last_buy_date": trade_date,
            "entry_price": open_price,
            "entry_cost_basis": cost_per_share,
        }
    return cash, bought_value


def _sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["composite_score", "alpha_probability"]
    available_sort_columns = [column for column in sort_columns if column in frame.columns]
    if not available_sort_columns:
        return frame
    return frame.sort_values(available_sort_columns, ascending=False)


def _select_ranked_keep_symbols(
    signal_frame: pd.DataFrame,
    effective_max_positions: int,
    rank_keep_fraction: float,
) -> set[str]:
    if signal_frame.empty or effective_max_positions <= 0:
        return set()
    rank_count = max(1, int(np.ceil(effective_max_positions * min(max(rank_keep_fraction, 0.25), 1.0))))
    ranked = _sort_candidates(signal_frame)
    return set(ranked.head(rank_count).index.astype(str))


def _compute_alpha_floor(signal_frame: pd.DataFrame, params: StrategyParameters, alpha_boost: float) -> float:
    if signal_frame.empty:
        return params.entry_threshold + alpha_boost
    quantile = float(signal_frame["alpha_probability"].quantile(params.entry_quantile))
    return min(params.entry_threshold, quantile) + alpha_boost


def _choose_trim(
    current_fraction: float,
    current_reason: str | None,
    candidate_fraction: float,
    candidate_reason: str,
) -> tuple[float, str | None]:
    if candidate_fraction > current_fraction:
        return candidate_fraction, candidate_reason
    return current_fraction, current_reason


def _resolve_exposure_settings(
    signal_frame: pd.DataFrame,
    params: StrategyParameters,
) -> tuple[int, float, float]:
    benchmark_distance_ma200 = float(signal_frame["benchmark_distance_ma200"].iloc[0])
    breadth_above_ma200 = float(signal_frame["breadth_above_ma200"].iloc[0])
    benchmark_ret_20d = float(signal_frame["benchmark_ret_20d"].iloc[0])

    is_defensive = (
        benchmark_distance_ma200 < DEFENSIVE_TREND_THRESHOLD
        or breadth_above_ma200 < DEFENSIVE_BREADTH_THRESHOLD
        or benchmark_ret_20d < DEFENSIVE_RET20_THRESHOLD
    )
    if is_defensive:
        return (
            max(1, int(np.ceil(params.max_positions * DEFENSIVE_POSITION_FRACTION))),
            DEFENSIVE_CASH_FRACTION,
            DEFENSIVE_ALPHA_BOOST,
        )
    return params.max_positions, 1.0, 0.0


def _compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, config: Trade60Config) -> dict:
    final_equity = float(equity_curve["equity"].iloc[-1])
    total_return = final_equity / config.initial_budget - 1.0
    benchmark_return = float(equity_curve["benchmark_curve"].iloc[-1] / config.initial_budget - 1.0)
    daily_returns = equity_curve["strategy_return"].fillna(0.0)
    cash_ratio_series = (
        equity_curve["cash"].fillna(0.0) / equity_curve["equity"].replace(0.0, np.nan)
    ).fillna(0.0)
    flat_day_ratio = float((equity_curve["open_positions"].fillna(0) == 0).mean())
    positive_benchmark_mask = equity_curve["benchmark_return"].fillna(0.0) > 0
    positive_benchmark_days = int(positive_benchmark_mask.sum())
    positive_benchmark_flat_ratio = (
        float(((equity_curve["open_positions"].fillna(0) == 0) & positive_benchmark_mask).sum() / positive_benchmark_days)
        if positive_benchmark_days > 0
        else 0.0
    )
    volatility = float(daily_returns.std(ddof=0) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(252))

    annual_returns = (
        equity_curve.assign(year=equity_curve["date"].dt.year)
        .groupby("year")["strategy_return"]
        .apply(lambda values: float((1.0 + values.fillna(0.0)).prod() - 1.0))
        .reset_index(name="strategy_return")
    )
    benchmark_yearly = (
        equity_curve.assign(year=equity_curve["date"].dt.year)
        .groupby("year")["benchmark_return"]
        .apply(lambda values: float((1.0 + values.fillna(0.0)).prod() - 1.0))
        .reset_index(name="benchmark_return")
    )
    annual_returns = annual_returns.merge(benchmark_yearly, on="year", how="left")
    annual_returns["excess_return"] = annual_returns["strategy_return"] - annual_returns["benchmark_return"]

    win_rate = float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0
    average_holding_days = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    turnover_ratio = float((equity_curve.get("bought_value", pd.Series(dtype=float)).fillna(0.0).sum() + equity_curve.get("sold_value", pd.Series(dtype=float)).fillna(0.0).sum()) / config.initial_budget)

    metrics = {
        "initial_budget": config.initial_budget,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized_return": annualized_return(total_return, periods=len(equity_curve)),
        "benchmark_return": benchmark_return,
        "benchmark_annualized_return": annualized_return(benchmark_return, periods=len(equity_curve)),
        "excess_return_vs_benchmark": total_return - benchmark_return,
        "beat_bank_target": bool(annualized_return(total_return, periods=len(equity_curve)) >= config.target_annual_return),
        "beat_benchmark": bool(total_return > benchmark_return),
        "max_drawdown": float(equity_curve["drawdown"].min()),
        "volatility": volatility,
        "sharpe": sharpe,
        "avg_cash_ratio": float(cash_ratio_series.mean()),
        "flat_day_ratio": flat_day_ratio,
        "positive_benchmark_flat_ratio": positive_benchmark_flat_ratio,
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "average_holding_days": average_holding_days,
        "average_open_positions": float(equity_curve["open_positions"].mean()),
        "turnover_ratio": turnover_ratio,
        "yearly_breakdown": annual_returns.to_dict("records"),
    }
    return metrics
