from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trade60_app.config import Trade60Config, build_symbol_frame
from trade60_app.data.collector import DataBundle, Trade60DataCollector, load_cached_bundle
from trade60_app.features.daily import build_daily_feature_panel, build_live_feature_snapshot
from trade60_app.backtest.engine import (
    DEFENSIVE_ALPHA_BOOST,
    DEFENSIVE_BREADTH_THRESHOLD,
    DEFENSIVE_CASH_FRACTION,
    DEFENSIVE_POSITION_FRACTION,
    DEFENSIVE_RET20_THRESHOLD,
    DEFENSIVE_TREND_THRESHOLD,
)
from trade60_app.models.trainer import (
    CleanWalkforwardArtifacts,
    TrainingArtifacts,
    calibrate_deployment_parameters,
    fit_model_bundles,
    load_model_bundle,
    run_clean_walkforward_evaluation,
    score_panel,
    train_models,
)
from trade60_app.utils import ProgressCallback, ensure_directories, read_json, report_progress, subprogress, write_json


def download_data(
    config: Trade60Config,
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> DataBundle:
    collector = Trade60DataCollector(config, progress_callback=progress_callback)
    return collector.collect_all(refresh=refresh)


def build_feature_store(
    config: Trade60Config,
    refresh_data: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    ensure_directories([config.raw_dir, config.processed_dir])
    try:
        bundle = (
            download_data(
                config,
                refresh=refresh_data,
                progress_callback=subprogress(progress_callback, 0.05, 0.45),
            )
            if refresh_data
            else load_cached_bundle(config)
        )
    except FileNotFoundError:
        bundle = download_data(config, refresh=True, progress_callback=subprogress(progress_callback, 0.05, 0.45))
    report_progress(progress_callback, "Đang xây dựng feature panel đầy đủ từ dữ liệu lịch sử...", 0.50)
    panel = build_daily_feature_panel(bundle, config, progress_callback=subprogress(progress_callback, 0.50, 0.80))
    panel.to_parquet(config.feature_store_path, index=False)
    report_progress(progress_callback, "Đã cập nhật xong feature panel đầy đủ.", 0.80)
    return panel


def train_strategy(
    config: Trade60Config,
    refresh_data: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    ensure_directories([config.artifacts_dir, config.models_dir, config.reports_dir, config.recommendations_dir])
    report_progress(progress_callback, "Bắt đầu huấn luyện Trade60...", 0.0)
    panel = build_feature_store(config, refresh_data=refresh_data, progress_callback=subprogress(progress_callback, 0.0, 0.35))
    report_progress(progress_callback, "Đang huấn luyện mô hình cơ sở...", 0.40)
    artifacts = train_models(panel, config)
    report_progress(progress_callback, "Đang chạy clean walk-forward evaluation...", 0.58)
    clean_eval = run_clean_walkforward_evaluation(panel, config)
    report_progress(progress_callback, "Đang hiệu chỉnh tham số triển khai cũ để so sánh...", 0.76)
    legacy_deployment_params, legacy_deployment_backtest = calibrate_deployment_parameters(
        artifacts.holdout_scored_panel,
        artifacts.best_params,
        config,
    )
    report_progress(progress_callback, "Đang tạo khuyến nghị mới nhất và lưu artifacts...", 0.88)
    joblib.dump(clean_eval.production_alpha_bundle, config.models_dir / "alpha_model.joblib")
    joblib.dump(clean_eval.production_regime_bundle, config.models_dir / "regime_model.joblib")
    write_json(artifacts.best_params, config.models_dir / "best_params.json")
    write_json(clean_eval.selected_params, config.models_dir / "deployment_params.json")
    write_json(legacy_deployment_params, config.models_dir / "legacy_deployment_params.json")
    baseline_trade_plan = generate_trade_plan(
        config=config,
        budget=config.initial_budget,
        holdings=build_holdings_template(config),
        refresh_data=False,
        panel=panel,
        alpha_bundle=clean_eval.production_alpha_bundle,
        regime_bundle=clean_eval.production_regime_bundle,
        best_params=clean_eval.selected_params,
    )
    artifacts.feature_importance.to_csv(config.models_dir / "feature_importance.csv", index=False)
    artifacts.validation_scored_panel.to_parquet(config.models_dir / "validation_scored_panel.parquet", index=False)
    artifacts.holdout_scored_panel.to_parquet(config.models_dir / "holdout_scored_panel.parquet", index=False)
    clean_eval.final_scored_panel.to_parquet(config.models_dir / "clean_final_scored_panel.parquet", index=False)
    artifacts.validation_backtest.trades.to_csv(config.reports_dir / "validation_trades.csv", index=False)
    artifacts.validation_backtest.equity_curve.to_csv(config.reports_dir / "validation_equity_curve.csv", index=False)
    artifacts.holdout_backtest.trades.to_csv(config.reports_dir / "holdout_trades.csv", index=False)
    artifacts.holdout_backtest.equity_curve.to_csv(config.reports_dir / "holdout_equity_curve.csv", index=False)
    clean_eval.final_backtest.trades.to_csv(config.reports_dir / "clean_final_test_trades.csv", index=False)
    clean_eval.final_backtest.equity_curve.to_csv(config.reports_dir / "clean_final_test_equity_curve.csv", index=False)
    baseline_trade_plan["actions"].to_csv(config.recommendations_dir / "baseline_recommendations.csv", index=False)
    write_json(
        {
            "portfolio_mode": "baseline_no_holdings",
            "latest_signal_date": baseline_trade_plan["latest_signal_date"],
            "cash_after_actions": baseline_trade_plan["cash_after_actions"],
            "notes": baseline_trade_plan["notes"],
            "actions": baseline_trade_plan["actions"].to_dict("records"),
            "position_status": baseline_trade_plan["position_status"].to_dict("records"),
        },
        config.recommendations_dir / "baseline_recommendations.json",
    )
    for stale_path in [
        config.recommendations_dir / "current_recommendations.csv",
        config.recommendations_dir / "current_recommendations.json",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    summary = {
        "config": asdict(config),
        "best_params": clean_eval.selected_params,
        "research_best_params": artifacts.best_params,
        "legacy_deployment_params": legacy_deployment_params,
        "split_summary": artifacts.split_summary,
        "clean_walkforward_split": clean_eval.split_summary,
        "clean_walkforward_fold_summaries": clean_eval.fold_summaries,
        "validation_metrics": artifacts.validation_backtest.metrics,
        "research_holdout_metrics": artifacts.holdout_backtest.metrics,
        "legacy_deployment_holdout_metrics": legacy_deployment_backtest.metrics,
        "holdout_metrics": clean_eval.final_backtest.metrics,
        "latest_signal_date": baseline_trade_plan["latest_signal_date"],
        "recommendation_count": int(len(baseline_trade_plan["actions"])),
    }
    write_json(summary, config.artifacts_dir / "summary.json")
    _write_strategy_report(config, summary, baseline_trade_plan, artifacts)
    report_progress(progress_callback, "Huấn luyện và cập nhật artifacts đã hoàn tất.", 1.0)
    return summary


def load_trained_artifacts(config: Trade60Config):
    alpha_bundle = load_model_bundle(config.models_dir / "alpha_model.joblib")
    regime_bundle = load_model_bundle(config.models_dir / "regime_model.joblib")
    params_path = config.models_dir / "deployment_params.json"
    if not params_path.exists():
        params_path = config.models_dir / "best_params.json"
    best_params = read_json(params_path)
    return alpha_bundle, regime_bundle, best_params


def build_holdings_template(config: Trade60Config) -> pd.DataFrame:
    frame = build_symbol_frame(config.symbols)
    frame["quantity"] = 0
    frame["avg_cost"] = 0.0
    frame["buy_date"] = ""
    return frame[["symbol", "quantity", "avg_cost", "buy_date"]]


def generate_trade_plan(
    config: Trade60Config,
    budget: float,
    holdings: pd.DataFrame | None = None,
    refresh_data: bool = False,
    panel: pd.DataFrame | None = None,
    alpha_bundle=None,
    regime_bundle=None,
    best_params: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    trading_calendar: pd.Series
    if panel is not None:
        panel = panel.copy()
        trading_calendar = pd.to_datetime(panel["date"]).drop_duplicates().sort_values().reset_index(drop=True)
    elif refresh_data or not config.feature_store_path.exists():
        report_progress(progress_callback, "Đang cập nhật dữ liệu thị trường mới nhất...", 0.02)
        bundle = download_data(config, refresh=refresh_data, progress_callback=subprogress(progress_callback, 0.05, 0.72))
        report_progress(progress_callback, "Đang tính snapshot đặc trưng gần nhất cho khuyến nghị...", 0.75)
        latest_panel, trading_calendar = build_live_feature_snapshot(
            bundle,
            config,
            progress_callback=subprogress(progress_callback, 0.75, 0.88),
        )
        panel = latest_panel
    else:
        panel = pd.read_parquet(config.feature_store_path)
        trading_calendar = pd.to_datetime(panel["date"]).drop_duplicates().sort_values().reset_index(drop=True)
    if alpha_bundle is None or regime_bundle is None or best_params is None:
        report_progress(progress_callback, "Đang nạp model đã huấn luyện...", 0.90)
        alpha_bundle, regime_bundle, best_params = load_trained_artifacts(config)

    latest_date = pd.to_datetime(panel["date"]).max()
    latest_panel = panel.loc[pd.to_datetime(panel["date"]) == latest_date].copy()
    report_progress(progress_callback, f"Đang chấm điểm cho ngày {latest_date.date()}...", 0.94)
    scored = score_panel(latest_panel, alpha_bundle, regime_bundle).sort_values(
        ["composite_score", "alpha_probability"],
        ascending=False,
    )

    normalized_holdings, holding_notes = _normalize_holdings(holdings, config, latest_date)
    actions, position_status, cash_after_actions, notes = _build_actions(
        scored,
        normalized_holdings,
        float(budget),
        best_params,
        config,
        trading_calendar,
    )
    notes = holding_notes + notes
    if actions.empty and position_status.empty:
        actions = pd.DataFrame(
            [
                {
                    "action": "DO_NOTHING",
                    "symbol": "",
                    "quantity": 0,
                    "reference_price": 0.0,
                    "alpha_probability": 0.0,
                    "regime_probability": float(scored["regime_probability"].iloc[0]) if not scored.empty else 0.0,
                    "rationale": "No symbol cleared the risk-aware buy or sell filters for the next session.",
                }
            ]
        )
    elif actions.empty:
        notes.append("No buy/sell adjustments are needed for the next session. Current holdings remain in place.")

    return {
        "latest_signal_date": pd.Timestamp(latest_date),
        "actions": actions.reset_index(drop=True),
        "position_status": position_status.reset_index(drop=True),
        "cash_after_actions": cash_after_actions,
        "notes": notes,
        "scored_panel": scored.reset_index(drop=True),
    }


def _build_actions(
    scored: pd.DataFrame,
    holdings: pd.DataFrame,
    cash_budget: float,
    best_params: dict,
    config: Trade60Config,
    trading_calendar: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, float, list[str]]:
    params = _complete_best_params(best_params, config)
    notes: list[str] = []
    action_map: dict[str, dict] = {}
    position_status_map: dict[str, dict] = {}
    sell_symbols: set[str] = set()
    available_cash = float(cash_budget)
    latest_date = pd.Timestamp(scored["date"].max()) if not scored.empty else pd.Timestamp.today()
    latest_market = scored.set_index("symbol") if not scored.empty else pd.DataFrame()
    active_holdings = holdings.loc[holdings["quantity"] > 0].copy()
    sell_cost = (config.commission_bps + config.slippage_bps) / 10_000.0
    pre_plan_equity = available_cash
    for row in active_holdings.itertuples(index=False):
        if row.symbol in latest_market.index:
            pre_plan_equity += int(row.quantity) * float(latest_market.at[row.symbol, "close"])

    regime_probability = float(scored["regime_probability"].iloc[0]) if not scored.empty else 0.0
    effective_max_positions, deploy_fraction, alpha_boost, posture_note = _resolve_live_exposure_settings(scored, params)
    is_defensive = posture_note is not None
    alpha_floor = _compute_live_alpha_floor(scored, params, alpha_boost)
    keep_symbols = _select_live_keep_symbols(scored, effective_max_positions, float(params["rank_keep_fraction"]))
    if posture_note:
        notes.append(posture_note)

    for row in active_holdings.itertuples(index=False):
        symbol = row.symbol
        if symbol not in latest_market.index:
            position_status_map[symbol] = {
                "symbol": symbol,
                "status": "KEEP",
                "current_quantity": int(row.quantity),
                "next_quantity": int(row.quantity),
                "delta_quantity": 0,
                "avg_cost": float(row.avg_cost),
                "reference_price": float(row.avg_cost),
                "alpha_probability": 0.0,
                "regime_probability": regime_probability,
                "rationale": "The latest market snapshot did not include this symbol, so the system kept it unchanged.",
            }
            continue
        market_row = latest_market.loc[symbol]
        quantity = int(row.quantity)
        avg_cost = float(row.avg_cost)
        current_close = float(market_row["close"])
        regime_probability = float(market_row["regime_probability"])
        alpha_probability = float(market_row["alpha_probability"])
        pnl = current_close / max(avg_cost, 1e-9) - 1.0
        holding_days = _holding_days_since(row.buy_date, latest_date, trading_calendar)
        if getattr(row, "buy_date_defaulted", False):
            notes.append(f"`{symbol}` buy date was blank, so the system assumed {row.buy_date} (~1 month ago).")

        position_status_map[symbol] = {
            "symbol": symbol,
            "status": "KEEP",
            "current_quantity": quantity,
            "next_quantity": quantity,
            "delta_quantity": 0,
            "avg_cost": avg_cost,
            "reference_price": current_close,
            "alpha_probability": alpha_probability,
            "regime_probability": regime_probability,
            "rationale": "The position is strong enough to keep unchanged for now.",
        }

        sell_reason = None
        if pd.notna(holding_days) and holding_days >= int(params["max_holding_days"]):
            sell_reason = "Reached the 2-month holding limit."
        elif pnl <= -float(params["stop_loss_pct"]):
            sell_reason = "Current loss breached the stop-loss threshold."
        elif pd.notna(holding_days) and holding_days >= int(params["min_holding_days"]):
            if alpha_probability < float(params["exit_threshold"]):
                sell_reason = "The stock score dropped below the exit threshold."
            elif regime_probability < float(params["regime_threshold"]):
                sell_reason = "The market regime is no longer supportive."

        if sell_reason is not None:
            sell_symbols.add(symbol)
            available_cash += quantity * current_close * (1.0 - sell_cost)
            _upsert_action(
                action_map,
                action="SELL",
                symbol=symbol,
                quantity=quantity,
                reference_price=current_close,
                alpha_probability=alpha_probability,
                regime_probability=regime_probability,
                rationale=sell_reason,
            )
            position_status_map[symbol]["status"] = "EXIT"
            position_status_map[symbol]["next_quantity"] = 0
            position_status_map[symbol]["delta_quantity"] = -quantity
            position_status_map[symbol]["rationale"] = sell_reason
            continue

        trim_fraction = 0.0
        trim_reason = None
        if pnl >= float(params["take_profit_pct"]):
            trim_fraction, trim_reason = _choose_live_trim(
                trim_fraction,
                trim_reason,
                float(params["profit_trim_fraction"]),
                "The position was partially trimmed to lock in gains.",
            )
        if (
            pd.notna(holding_days)
            and holding_days >= int(params["min_holding_days"])
            and alpha_probability < (float(params["exit_threshold"]) + float(params["hold_alpha_buffer"]))
        ):
            trim_fraction, trim_reason = _choose_live_trim(
                trim_fraction,
                trim_reason,
                float(params["weak_alpha_trim_fraction"]),
                "The stock score softened, so the system trimmed part of the position.",
            )
        if is_defensive:
            trim_fraction, trim_reason = _choose_live_trim(
                trim_fraction,
                trim_reason,
                float(params["defensive_trim_fraction"]),
                "Defensive posture reduced the position size.",
            )

        trim_quantity = _shares_for_live_fraction(quantity, trim_fraction)
        next_quantity = quantity
        if trim_quantity > 0:
            next_quantity = quantity - trim_quantity
            sell_symbols.add(symbol)
            available_cash += trim_quantity * current_close * (1.0 - sell_cost)
            _upsert_action(
                action_map,
                action="SELL",
                symbol=symbol,
                quantity=trim_quantity,
                reference_price=current_close,
                alpha_probability=alpha_probability,
                regime_probability=regime_probability,
                rationale=trim_reason or "The position was partially trimmed.",
            )
            position_status_map[symbol]["status"] = "TRIM" if next_quantity > 0 else "EXIT"
            position_status_map[symbol]["next_quantity"] = next_quantity
            position_status_map[symbol]["delta_quantity"] = next_quantity - quantity
            position_status_map[symbol]["rationale"] = trim_reason or "The position was partially trimmed."

        if next_quantity > 0:
            position_status_map[symbol]["next_quantity"] = next_quantity
            position_status_map[symbol]["delta_quantity"] = next_quantity - quantity

    remaining_positions = {
        symbol: status for symbol, status in position_status_map.items() if int(status["next_quantity"]) > 0
    }
    if len(remaining_positions) > effective_max_positions:
        overflow = len(remaining_positions) - effective_max_positions
        ranked_exits = sorted(
            remaining_positions,
            key=lambda symbol: (
                symbol in keep_symbols,
                float(latest_market.at[symbol, "composite_score"]) if symbol in latest_market.index else float("-inf"),
            ),
        )
        for symbol in ranked_exits[:overflow]:
            current_status = position_status_map[symbol]
            remaining_quantity = int(current_status["next_quantity"])
            if remaining_quantity < 1:
                continue
            available_cash += remaining_quantity * float(current_status["reference_price"]) * (1.0 - sell_cost)
            sell_symbols.add(symbol)
            _upsert_action(
                action_map,
                action="SELL",
                symbol=symbol,
                quantity=remaining_quantity,
                reference_price=float(current_status["reference_price"]),
                alpha_probability=float(current_status["alpha_probability"]),
                regime_probability=float(current_status["regime_probability"]),
                rationale="The holding rank slipped behind stronger opportunities, so the system exited the remainder.",
            )
            current_status["status"] = "EXIT"
            current_status["next_quantity"] = 0
            current_status["delta_quantity"] = -int(current_status["current_quantity"])
            current_status["rationale"] = "The holding rank slipped behind stronger opportunities, so the system exited the remainder."

    live_positions = {
        symbol: status for symbol, status in position_status_map.items() if int(status["next_quantity"]) > 0
    }
    if regime_probability >= float(params["regime_threshold"]) and effective_max_positions > 0:
        candidates = scored.loc[
            (scored["alpha_probability"] >= alpha_floor)
            & ~scored["symbol"].isin(sell_symbols)
        ].copy()
        candidates = _sort_live_candidates(candidates)
        current_positions = len(live_positions)
        open_slots = max(effective_max_positions - current_positions, 0)
        top_up_symbols = [
            symbol for symbol in candidates["symbol"].tolist() if symbol in live_positions and symbol in keep_symbols
        ]
        new_symbols = [
            symbol for symbol in candidates["symbol"].tolist() if symbol not in live_positions
        ][:open_slots]
        desired_position_count = max(current_positions + len(new_symbols), 1)
        target_position_value = (pre_plan_equity * deploy_fraction) / desired_position_count

        for symbol in top_up_symbols + new_symbols:
            market_row = latest_market.loc[symbol]
            reference_price = float(market_row["close"])
            cost_per_share = reference_price * (1.0 + sell_cost)
            target_quantity = int(np.floor(target_position_value / max(cost_per_share, 1e-9)))
            current_quantity = int(live_positions[symbol]["next_quantity"]) if symbol in live_positions else 0
            delta_quantity = target_quantity - current_quantity
            if delta_quantity < 1:
                continue
            affordable_quantity = int(np.floor(available_cash / max(cost_per_share, 1e-9)))
            quantity = min(delta_quantity, affordable_quantity)
            if quantity < 1:
                continue
            available_cash -= quantity * cost_per_share
            _upsert_action(
                action_map,
                action="BUY",
                symbol=symbol,
                quantity=quantity,
                reference_price=reference_price,
                alpha_probability=float(market_row["alpha_probability"]),
                regime_probability=float(market_row["regime_probability"]),
                rationale=(
                    "The score remains strong enough to add to the position."
                    if symbol in live_positions
                    else _explain_symbol(market_row)
                ),
            )
            if symbol in live_positions:
                existing_status = position_status_map[symbol]
                new_quantity = int(existing_status["next_quantity"]) + quantity
                existing_status["status"] = "TOP_UP"
                existing_status["next_quantity"] = new_quantity
                existing_status["delta_quantity"] = new_quantity - int(existing_status["current_quantity"])
                existing_status["avg_cost"] = _weighted_average_price(
                    current_quantity=int(existing_status["current_quantity"]),
                    current_price=float(existing_status["avg_cost"]),
                    added_quantity=quantity,
                    added_price=reference_price,
                )
                existing_status["rationale"] = "The score remains strong enough to add to the position."
                live_positions[symbol] = existing_status
    elif regime_probability < float(params["regime_threshold"]):
        notes.append("The regime filter is defensive today, so the system will not open new positions for tomorrow.")

    actions = _empty_actions_frame() if not action_map else pd.DataFrame(action_map.values())
    if not actions.empty:
        actions = actions.sort_values(["action", "symbol"], ascending=[True, True]).reset_index(drop=True)
    position_status = _empty_position_status_frame() if not position_status_map else pd.DataFrame(position_status_map.values())
    if not position_status.empty:
        position_status = position_status.sort_values("symbol").reset_index(drop=True)
    return actions, position_status, available_cash, notes


def _normalize_holdings(
    holdings: pd.DataFrame | None,
    config: Trade60Config,
    reference_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    template = build_holdings_template(config)
    if holdings is None or holdings.empty:
        template["buy_date_defaulted"] = False
        return template, []
    normalized = holdings.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized["quantity"] = pd.to_numeric(
        normalized["quantity"] if "quantity" in normalized.columns else 0,
        errors="coerce",
    ).fillna(0).astype(int)
    normalized["avg_cost"] = pd.to_numeric(
        normalized["avg_cost"] if "avg_cost" in normalized.columns else 0.0,
        errors="coerce",
    ).fillna(0.0)
    if "buy_date" in normalized.columns:
        normalized["buy_date"] = normalized["buy_date"].fillna("").astype(str).str.strip()
    else:
        normalized["buy_date"] = ""
    normalized = normalized.loc[normalized["symbol"].isin(config.symbols)]
    if normalized.empty:
        template["buy_date_defaulted"] = False
        return template, []

    active = normalized.loc[normalized["quantity"] > 0].copy()
    if active.empty:
        template["buy_date_defaulted"] = False
        return template, []

    assumed_buy_date = (reference_date - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    aggregated_rows: list[dict] = []
    for symbol, frame in active.groupby("symbol", sort=False):
        total_quantity = int(frame["quantity"].sum())
        if total_quantity < 1:
            continue
        weighted_cost = float((frame["avg_cost"] * frame["quantity"]).sum() / max(total_quantity, 1))
        parsed_dates = pd.to_datetime(frame["buy_date"], errors="coerce")
        buy_date_defaulted = not parsed_dates.notna().any()
        chosen_buy_date = (
            assumed_buy_date
            if buy_date_defaulted
            else pd.Timestamp(parsed_dates.dropna().min()).strftime("%Y-%m-%d")
        )
        aggregated_rows.append(
            {
                "symbol": symbol,
                "quantity": total_quantity,
                "avg_cost": weighted_cost,
                "buy_date": chosen_buy_date,
                "buy_date_defaulted": buy_date_defaulted,
            }
        )

    if not aggregated_rows:
        template["buy_date_defaulted"] = False
        return template, []
    return pd.DataFrame(aggregated_rows), []


def _empty_actions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "action",
            "symbol",
            "quantity",
            "reference_price",
            "alpha_probability",
            "regime_probability",
            "rationale",
        ]
    )


def _empty_position_status_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "status",
            "current_quantity",
            "next_quantity",
            "delta_quantity",
            "avg_cost",
            "reference_price",
            "alpha_probability",
            "regime_probability",
            "rationale",
        ]
    )


def _complete_best_params(best_params: dict | None, config: Trade60Config) -> dict:
    defaults = {
        "entry_threshold": config.entry_threshold,
        "entry_quantile": 0.95,
        "exit_threshold": config.exit_threshold,
        "regime_threshold": config.regime_threshold,
        "max_positions": config.max_positions,
        "max_holding_days": config.max_holding_days,
        "min_holding_days": config.min_holding_days,
        "stop_loss_pct": config.stop_loss_pct,
        "take_profit_pct": config.take_profit_pct,
        "hold_alpha_buffer": config.hold_alpha_buffer,
        "rank_keep_fraction": config.rank_keep_fraction,
        "defensive_trim_fraction": config.defensive_trim_fraction,
        "weak_alpha_trim_fraction": config.weak_alpha_trim_fraction,
        "profit_trim_fraction": config.profit_trim_fraction,
    }
    if best_params:
        defaults.update(best_params)
    return defaults


def _upsert_action(
    action_map: dict[str, dict],
    action: str,
    symbol: str,
    quantity: int,
    reference_price: float,
    alpha_probability: float,
    regime_probability: float,
    rationale: str,
) -> None:
    if quantity < 1:
        return
    existing = action_map.get(symbol)
    if existing is None:
        action_map[symbol] = {
            "action": action,
            "symbol": symbol,
            "quantity": int(quantity),
            "reference_price": reference_price,
            "alpha_probability": alpha_probability,
            "regime_probability": regime_probability,
            "rationale": rationale,
        }
        return
    existing["quantity"] = int(existing["quantity"]) + int(quantity)
    existing["reference_price"] = reference_price
    existing["alpha_probability"] = alpha_probability
    existing["regime_probability"] = regime_probability
    existing["rationale"] = rationale
    existing["action"] = action


def _compute_live_alpha_floor(scored: pd.DataFrame, params: dict, alpha_boost: float) -> float:
    if scored.empty:
        return float(params["entry_threshold"]) + alpha_boost
    quantile = float(scored["alpha_probability"].quantile(float(params["entry_quantile"])))
    return min(float(params["entry_threshold"]), quantile) + alpha_boost


def _sort_live_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(["composite_score", "alpha_probability"], ascending=False)


def _select_live_keep_symbols(scored: pd.DataFrame, effective_max_positions: int, rank_keep_fraction: float) -> set[str]:
    if scored.empty or effective_max_positions <= 0:
        return set()
    rank_count = max(1, int(np.ceil(effective_max_positions * min(max(rank_keep_fraction, 0.25), 1.0))))
    ranked = _sort_live_candidates(scored)
    return set(ranked.head(rank_count)["symbol"].astype(str))


def _shares_for_live_fraction(quantity: int, fraction: float) -> int:
    if quantity <= 0 or fraction <= 0:
        return 0
    planned = int(np.floor(quantity * min(max(fraction, 0.0), 1.0)))
    if planned < 1:
        planned = 1
    return min(planned, quantity)


def _choose_live_trim(
    current_fraction: float,
    current_reason: str | None,
    candidate_fraction: float,
    candidate_reason: str,
) -> tuple[float, str | None]:
    if candidate_fraction > current_fraction:
        return candidate_fraction, candidate_reason
    return current_fraction, current_reason


def _weighted_average_price(
    current_quantity: int,
    current_price: float,
    added_quantity: int,
    added_price: float,
) -> float:
    total_quantity = current_quantity + added_quantity
    if total_quantity <= 0:
        return 0.0
    return ((current_quantity * current_price) + (added_quantity * added_price)) / total_quantity


def _holding_days_since(buy_date: str, latest_date: pd.Timestamp, trading_calendar: pd.Series) -> float | pd.NA:
    if not buy_date:
        return pd.NA
    parsed = pd.to_datetime(buy_date, errors="coerce")
    if pd.isna(parsed):
        return pd.NA
    trading_dates = pd.to_datetime(trading_calendar).drop_duplicates().sort_values()
    return float(((trading_dates >= parsed) & (trading_dates <= latest_date)).sum())


def _resolve_live_exposure_settings(scored: pd.DataFrame, best_params: dict) -> tuple[int, float, float, str | None]:
    if scored.empty:
        return int(best_params["max_positions"]), 1.0, 0.0, None

    benchmark_distance_ma200 = float(scored["benchmark_distance_ma200"].iloc[0])
    breadth_above_ma200 = float(scored["breadth_above_ma200"].iloc[0])
    benchmark_ret_20d = float(scored["benchmark_ret_20d"].iloc[0])

    is_defensive = (
        benchmark_distance_ma200 < DEFENSIVE_TREND_THRESHOLD
        or breadth_above_ma200 < DEFENSIVE_BREADTH_THRESHOLD
        or benchmark_ret_20d < DEFENSIVE_RET20_THRESHOLD
    )
    if not is_defensive:
        return int(best_params["max_positions"]), 1.0, 0.0, None

    reduced_positions = max(1, int(round(float(best_params["max_positions"]) * DEFENSIVE_POSITION_FRACTION)))
    return (
        reduced_positions,
        DEFENSIVE_CASH_FRACTION,
        DEFENSIVE_ALPHA_BOOST,
        "Defensive exposure mode is active, so the system will size fewer positions and keep extra cash.",
    )


def _explain_symbol(row) -> str:
    reasons: list[str] = []
    if getattr(row, "relative_strength_20d", 0) > 0:
        reasons.append("beating VNINDEX over the last month")
    if getattr(row, "distance_ma50", -1) > 0:
        reasons.append("trading above its 50-day trend")
    if getattr(row, "breadth_above_ma50", 0) > 0.55:
        reasons.append("the 60-symbol market breadth is supportive")
    if getattr(row, "volume_zscore_20d", 0) > 0:
        reasons.append("volume is stronger than its recent average")
    return ", ".join(reasons) if reasons else "best composite score in the live universe"


def _write_strategy_report(
    config: Trade60Config,
    summary: dict,
    latest_trade_plan: dict,
    artifacts: TrainingArtifacts,
) -> None:
    report_path = config.reports_dir / "strategy_report.md"
    validation = summary["validation_metrics"]
    holdout = summary["holdout_metrics"]
    research_holdout = summary.get("research_holdout_metrics", holdout)
    legacy_deployment_holdout = summary.get("legacy_deployment_holdout_metrics", holdout)
    actions = latest_trade_plan["actions"]
    action_preview = actions.head(10).to_markdown(index=False) if not actions.empty else "No baseline actions."
    report = f"""# Trade60 Strategy Report

## Objective
- Long-only daily allocation system across 60 symbols.
- Starting capital: {config.initial_budget:,.0f}
- Max holding period: {config.max_holding_days} trading days
- Constraints: no margin, no same-day round-trip on one symbol, next-session execution only.

## Validation Metrics
- Annualized return: {validation['annualized_return']:.2%}
- Total return: {validation['total_return']:.2%}
- Benchmark return: {validation['benchmark_return']:.2%}
- Max drawdown: {validation['max_drawdown']:.2%}
- Win rate: {validation['win_rate']:.2%}

## Clean Untouched Final-Test Metrics
- Annualized return: {holdout['annualized_return']:.2%}
- Total return: {holdout['total_return']:.2%}
- Benchmark return: {holdout['benchmark_return']:.2%}
- Excess return vs benchmark: {holdout['excess_return_vs_benchmark']:.2%}
- Beats {config.target_annual_return:.0%} annual hurdle: {holdout['beat_bank_target']}
- Beats benchmark: {holdout['beat_benchmark']}
- Max drawdown: {holdout['max_drawdown']:.2%}

## Research Holdout Metrics Before Deployment Calibration
- Annualized return: {research_holdout['annualized_return']:.2%}
- Total return: {research_holdout['total_return']:.2%}
- Benchmark return: {research_holdout['benchmark_return']:.2%}
- Excess return vs benchmark: {research_holdout['excess_return_vs_benchmark']:.2%}

## Legacy Deployment-Calibrated Holdout Metrics
- Annualized return: {legacy_deployment_holdout['annualized_return']:.2%}
- Total return: {legacy_deployment_holdout['total_return']:.2%}
- Benchmark return: {legacy_deployment_holdout['benchmark_return']:.2%}
- Excess return vs benchmark: {legacy_deployment_holdout['excess_return_vs_benchmark']:.2%}

## Latest Baseline Next-Day Snapshot (No Portfolio)
{action_preview}

## Notes
- Latest signal date: {latest_trade_plan['latest_signal_date']}
- Additional notes: This snapshot assumes no open holdings. {', '.join(latest_trade_plan['notes']) if latest_trade_plan['notes'] else 'None'}
- Top feature drivers:
{artifacts.feature_importance.head(10).to_markdown(index=False) if not artifacts.feature_importance.empty else 'No feature importance available.'}
"""
    report_path.write_text(report, encoding="utf-8")
