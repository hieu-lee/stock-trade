from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.portfolio.optimizer import PolicyParameters, PortfolioSnapshot, default_policy, optimize_target_weights


@dataclass(slots=True)
class DecisionBundle:
    actions: pd.DataFrame
    position_status: pd.DataFrame
    target_weights: pd.DataFrame
    notes: list[str]


def build_daily_actions(
    scored_snapshot: pd.DataFrame,
    holdings: pd.DataFrame,
    cash: float,
    config: TradingBotConfig,
    policy: PolicyParameters | None = None,
) -> DecisionBundle:
    if scored_snapshot.empty:
        raise ValueError("A scored daily snapshot is required to build actions.")
    market_date = pd.Timestamp(scored_snapshot["date"].iloc[0])
    normalized_holdings, notes = normalize_holdings(holdings, market_date)
    snapshot = PortfolioSnapshot(
        date=market_date,
        market_frame=scored_snapshot,
        holdings=normalized_holdings,
        cash=cash,
    )
    policy = policy or default_policy(config)
    target_weights = optimize_target_weights(snapshot=snapshot, config=config, policy=policy)
    actions, position_status = _compile_actions(
        target_weights=target_weights,
        holdings=normalized_holdings,
        cash=cash,
        market_date=market_date,
        config=config,
        policy=policy,
    )
    return DecisionBundle(actions=actions, position_status=position_status, target_weights=target_weights, notes=notes)


def normalize_holdings(holdings: pd.DataFrame, market_date: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    if holdings is None or holdings.empty:
        empty = pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date", "buy_date_defaulted"])
        return empty, []
    frame = holdings.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["quantity"] = pd.to_numeric(frame.get("quantity", 0), errors="coerce").fillna(0.0)
    frame["sellable_quantity"] = pd.to_numeric(frame.get("sellable_quantity", frame["quantity"]), errors="coerce").fillna(frame["quantity"])
    frame["sellable_quantity"] = frame["sellable_quantity"].clip(lower=0.0)
    frame["sellable_quantity"] = np.minimum(frame["sellable_quantity"], frame["quantity"])
    frame["avg_cost"] = pd.to_numeric(frame.get("avg_cost", 0), errors="coerce").fillna(0.0)
    notes: list[str] = []
    default_buy_date = (market_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    frame["buy_date"] = frame.get("buy_date", "").fillna("").astype(str).str.strip()
    frame["buy_date_defaulted"] = frame["buy_date"].eq("")
    if frame["buy_date_defaulted"].any():
        frame.loc[frame["buy_date_defaulted"], "buy_date"] = default_buy_date
        notes.append("Blank buy dates were defaulted to roughly one month ago for holding-period rules.")
    frame = frame.loc[frame["quantity"] > 0].reset_index(drop=True)
    return frame, notes


def _compile_actions(
    target_weights: pd.DataFrame,
    holdings: pd.DataFrame,
    cash: float,
    market_date: pd.Timestamp,
    config: TradingBotConfig,
    policy: PolicyParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = (
        holdings.set_index("symbol")
        if not holdings.empty
        else pd.DataFrame(columns=["quantity", "sellable_quantity", "avg_cost", "buy_date", "buy_date_defaulted"]).rename_axis("symbol")
    )
    reference_prices = target_weights.set_index("symbol")["close"].fillna(0.0).astype(float).to_dict()
    if holdings.empty:
        current_equity = cash
    else:
        current_equity = cash + sum(
            float(row["quantity"]) * reference_prices.get(symbol, float(row["avg_cost"]))
            for symbol, row in current.iterrows()
        )
    equity = max(current_equity, 1.0)
    actions: list[dict] = []
    statuses: list[dict] = []

    for row in target_weights.itertuples(index=False):
        symbol = str(row.symbol)
        close_price = float(row.close)
        utility_score = float(row.utility_score) if pd.notna(row.utility_score) else 0.0
        current_weight = float(row.current_weight) if hasattr(row, "current_weight") and pd.notna(row.current_weight) else 0.0
        target_weight = float(row.target_weight) if pd.notna(row.target_weight) else 0.0
        weight_delta = target_weight - current_weight
        min_trade_weight_delta = config.min_trade_weight_delta
        target_value = equity * target_weight
        target_qty = _round_lot(target_value / max(close_price, 1e-9), lot_size=config.lot_size)
        current_qty = int(current.at[symbol, "quantity"]) if symbol in current.index else 0
        sellable_qty = int(current.at[symbol, "sellable_quantity"]) if symbol in current.index else 0
        avg_cost = float(current.at[symbol, "avg_cost"]) if symbol in current.index else 0.0
        holding_days = _holding_days(current.at[symbol, "buy_date"], market_date) if symbol in current.index else 0
        pnl_pct = (close_price / avg_cost - 1.0) if avg_cost > 0 else 0.0

        action = "HOLD"
        rationale = "Weight change is too small to trade today."
        next_qty = current_qty

        force_exit = (
            current_qty > 0
            and (
                holding_days >= config.max_holding_days
                or (holding_days >= config.min_holding_days and pnl_pct <= -config.stop_loss_pct)
                or (pnl_pct >= config.take_profit_pct and utility_score < policy.add_threshold - config.hold_buffer)
                or utility_score <= policy.exit_threshold - config.hold_buffer
            )
        )

        if force_exit:
            if (
                pnl_pct >= config.take_profit_pct
                and holding_days >= config.min_holding_days
                and utility_score > policy.exit_threshold
            ):
                trim_qty = max(_round_lot(current_qty * 0.5, lot_size=config.lot_size), 0)
                if trim_qty > 0 and utility_score < policy.add_threshold:
                    action = "TRIM"
                    next_qty = max(current_qty - trim_qty, 0)
                    rationale = "Partial profit lock after the signal cooled from its strongest state."
                else:
                    action = "HOLD"
                    next_qty = current_qty
                    rationale = "Signal remains strong enough to keep the winner instead of trimming."
            else:
                if sellable_qty > 0:
                    action = "EXIT" if sellable_qty >= current_qty else "TRIM"
                    next_qty = current_qty - sellable_qty
                    rationale = (
                        "Position failed holding or risk rules; only the settled quantity can be sold today."
                        if sellable_qty < current_qty
                        else "Position failed holding or risk rules and is being fully exited."
                    )
                else:
                    action = "HOLD"
                    next_qty = current_qty
                    rationale = "The position is locked by settlement and cannot be sold yet."
        elif (
            current_qty == 0
            and target_qty > 0
            and utility_score >= max(policy.buy_threshold - config.hold_buffer, policy.exit_threshold)
            and weight_delta > min_trade_weight_delta
        ):
            action = "BUY"
            next_qty = target_qty
            rationale = "High ranked signal with acceptable downside risk."
        elif (
            current_qty > 0
            and target_qty > current_qty
            and utility_score >= max(policy.add_threshold - config.add_buffer, policy.buy_threshold)
            and weight_delta > min_trade_weight_delta
        ):
            action = "ADD"
            next_qty = target_qty
            rationale = "The symbol remains strong enough to increase position size."
        elif current_qty > 0 and target_qty <= 0 and holding_days >= config.min_holding_days:
            if sellable_qty > 0:
                sell_qty = min(current_qty, sellable_qty)
                action = "EXIT" if sell_qty >= current_qty else "TRIM"
                next_qty = current_qty - sell_qty
                rationale = (
                    "The position no longer has a target weight; only the settled quantity can rotate out today."
                    if sell_qty < current_qty
                    else "The position no longer has a target weight and is being rotated out."
                )
            else:
                action = "HOLD"
                next_qty = current_qty
                rationale = "The position should be exited, but the shares are still unsettled."
        elif (
            current_qty > 0
            and target_qty < current_qty
            and abs(weight_delta) > min_trade_weight_delta
            and (
                utility_score <= policy.trim_threshold + config.trim_buffer
                or target_weight <= current_weight * 0.7
            )
        ):
            desired_reduction = max(current_qty - target_qty, 0)
            executable_reduction = min(desired_reduction, sellable_qty)
            if executable_reduction > 0:
                action = "TRIM"
                next_qty = current_qty - executable_reduction
                rationale = (
                    "The optimizer prefers a smaller weight, but settlement limits how much can be sold today."
                    if executable_reduction < desired_reduction
                    else "The optimizer prefers a meaningfully smaller weight for this position."
                )
            else:
                action = "HOLD"
                next_qty = current_qty
                rationale = "The position should be trimmed, but the shares are still unsettled."

        delta_qty = next_qty - current_qty
        if action == "TRIM" and next_qty <= 0:
            action = "EXIT"
            rationale = "The trim fully closes the position."
        if action == "BUY" and delta_qty <= 0:
            action = "HOLD"
            next_qty = current_qty
        if action == "ADD" and delta_qty <= 0:
            action = "HOLD"
            next_qty = current_qty
        if action == "TRIM" and delta_qty >= 0:
            action = "HOLD"
            next_qty = current_qty
        if action == "EXIT" and current_qty == 0:
            action = "HOLD"
            next_qty = current_qty

        statuses.append(
            {
                "symbol": symbol,
                "status": action,
                "current_quantity": current_qty,
                "sellable_quantity": sellable_qty,
                "next_quantity": next_qty,
                "delta_quantity": next_qty - current_qty,
                "avg_cost": avg_cost,
                "reference_price": close_price,
                "utility_score": utility_score,
                "risk_probability": float(row.risk_probability) if pd.notna(row.risk_probability) else np.nan,
                "regime_probability": float(row.regime_probability) if pd.notna(row.regime_probability) else np.nan,
                "rationale": rationale,
            }
        )
        if action != "HOLD":
            actions.append(
                {
                    "date": market_date,
                    "symbol": symbol,
                    "action": action,
                    "quantity": abs(next_qty - current_qty) if action in {"BUY", "ADD", "TRIM"} else current_qty,
                    "current_quantity": current_qty,
                    "sellable_quantity": sellable_qty,
                    "next_quantity": next_qty,
                    "reference_price": close_price,
                    "target_weight": target_weight,
                    "utility_score": utility_score,
                    "risk_probability": float(row.risk_probability) if pd.notna(row.risk_probability) else np.nan,
                    "regime_probability": float(row.regime_probability) if pd.notna(row.regime_probability) else np.nan,
                    "rationale": rationale,
                }
            )

    actions_frame = pd.DataFrame(actions)
    if actions_frame.empty:
        actions_frame = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "action",
                "quantity",
                "current_quantity",
                "sellable_quantity",
                "next_quantity",
                "reference_price",
                "target_weight",
                "utility_score",
                "risk_probability",
                "regime_probability",
                "rationale",
            ]
        )
    position_status = pd.DataFrame(statuses).sort_values(["status", "symbol"]).reset_index(drop=True)
    return actions_frame.sort_values(["action", "symbol"]).reset_index(drop=True), position_status


def _round_lot(quantity: float, lot_size: int = 100) -> int:
    if quantity <= 0:
        return 0
    return int(np.floor(quantity / lot_size) * lot_size)


def _holding_days(buy_date: str, market_date: pd.Timestamp) -> int:
    return int((market_date.normalize() - pd.Timestamp(buy_date).normalize()).days)
