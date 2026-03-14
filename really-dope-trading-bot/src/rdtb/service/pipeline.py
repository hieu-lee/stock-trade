from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from rdtb.config import TradingBotConfig, get_default_config
from rdtb.data.collector import collect_market_data
from rdtb.features.panel import build_feature_panel
from rdtb.models.train import ModelStack, score_panel
from rdtb.portfolio.actions import build_daily_actions
from rdtb.portfolio.transactions import replay_transactions
from rdtb.research.validation import run_strict_validation, validation_to_dict
from rdtb.utils import ProgressCallback, ensure_directories, read_json, report_progress, write_json


def train_system(
    config: TradingBotConfig | None = None,
    refresh_data: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = config or get_default_config()
    _ensure_layout(config)
    report_progress(progress_callback, "Refreshing market data...", 0.0)
    bundle = collect_market_data(config=config, refresh=refresh_data, progress_callback=progress_callback, mode="full")
    report_progress(progress_callback, "Building daily feature panel...", 0.55)
    panel = build_feature_panel(
        prices=bundle.prices,
        benchmarks=bundle.benchmarks,
        config=config,
        external_markets=bundle.external_markets,
        fundamentals=bundle.fundamentals,
        company_metadata=bundle.company_metadata if config.use_company_metadata_features else None,
        flow_history=bundle.flow_history,
        event_history=bundle.event_history,
        persist_path=config.feature_panel_path,
    )
    report_progress(progress_callback, "Running strict walk-forward validation...", 0.75)
    validation = run_strict_validation(panel, config)
    _persist_production_model_stack(validation.production_model_stack, config)
    validation.final_scored_panel.to_parquet(config.scored_panel_path, index=False)
    summary = validation_to_dict(validation)
    summary["feature_panel_rows"] = int(len(panel))
    summary["latest_feature_date"] = pd.to_datetime(panel["date"]).max()
    summary["latest_market_date"] = pd.to_datetime(bundle.prices["date"]).max()
    write_json(config.training_summary_path, summary)
    write_json(
        config.deployment_manifest_path,
        {
            "deployable": validation.deployable,
            "best_policy": asdict(validation.best_policy),
            "final_metrics": validation.final_backtest.metrics,
            "latest_market_date": pd.to_datetime(bundle.prices["date"]).max(),
        },
    )
    report_progress(progress_callback, "Training pipeline completed.", 1.0)
    return summary


def generate_daily_decisions(
    cash: float,
    holdings: pd.DataFrame | None,
    config: TradingBotConfig | None = None,
    refresh_data: bool = True,
    progress_callback: ProgressCallback | None = None,
    transactions: pd.DataFrame | None = None,
    starting_cash: float | None = None,
) -> dict[str, Any]:
    config = config or get_default_config()
    _ensure_layout(config)
    if refresh_data or not config.feature_panel_path.exists():
        report_progress(progress_callback, "Refreshing market data for recommendation...", 0.0)
        bundle = collect_market_data(config=config, refresh=refresh_data, progress_callback=progress_callback, mode="decision")
        panel = build_feature_panel(
            prices=bundle.prices,
            benchmarks=bundle.benchmarks,
            config=config,
            external_markets=bundle.external_markets,
            fundamentals=bundle.fundamentals,
            company_metadata=bundle.company_metadata if config.use_company_metadata_features else None,
            flow_history=bundle.flow_history,
            event_history=bundle.event_history,
            persist_path=config.feature_panel_path,
        )
    else:
        panel = pd.read_parquet(config.feature_panel_path)
    model_stack = _load_production_model_stack(config)
    scored_panel = score_panel(panel, model_stack)
    latest_date = pd.to_datetime(scored_panel["date"]).max()
    latest_snapshot = scored_panel.loc[scored_panel["date"] == latest_date].copy().reset_index(drop=True)
    manifest = _load_manifest(config)
    transaction_summary = None
    transaction_notes: list[str] = []
    effective_holdings = holdings if holdings is not None else build_holdings_template(config)
    effective_cash = float(cash)
    if transactions is not None and not transactions.empty:
        trading_calendar = pd.to_datetime(scored_panel["date"]).drop_duplicates().sort_values().tolist()
        replay = replay_transactions(
            transactions=transactions,
            starting_cash=float(starting_cash if starting_cash is not None else cash),
            config=config,
            as_of_date=latest_date,
            trading_calendar=trading_calendar,
        )
        effective_holdings = replay.holdings
        effective_cash = replay.available_cash
        transaction_notes = replay.notes
        transaction_summary = {
            "starting_cash": float(starting_cash if starting_cash is not None else cash),
            "available_cash": replay.available_cash,
            "pending_cash_total": replay.pending_cash_total,
            "pending_buy_quantity": replay.pending_buy_quantity,
            "holdings": replay.holdings.to_dict(orient="records"),
            "processed_transactions": replay.processed_transactions.to_dict(orient="records"),
        }
    decision_bundle = build_daily_actions(
        scored_snapshot=latest_snapshot,
        holdings=effective_holdings,
        cash=effective_cash,
        config=config,
        policy=_policy_from_manifest(config, manifest),
    )
    response = {
        "date": latest_date,
        "deployable": bool(manifest.get("deployable", False)),
        "notes": transaction_notes + decision_bundle.notes,
        "actions": decision_bundle.actions.to_dict(orient="records"),
        "position_status": decision_bundle.position_status.to_dict(orient="records"),
        "top_ranked": latest_snapshot.head(15)[
            [
                "symbol",
                "close",
                "alpha_prediction",
                "risk_probability",
                "regime_probability",
                "utility_score",
                "relative_strength_20d",
            ]
        ].to_dict(orient="records"),
        "contract_summary": manifest.get("final_metrics", {}),
        "transaction_summary": transaction_summary,
    }
    write_json(config.latest_decision_path, response)
    report_progress(progress_callback, "Daily recommendation completed.", 1.0)
    return response


def build_holdings_template(config: TradingBotConfig | None = None) -> pd.DataFrame:
    config = config or get_default_config()
    return pd.DataFrame(columns=["symbol", "quantity", "sellable_quantity", "avg_cost", "buy_date"])


def _ensure_layout(config: TradingBotConfig) -> None:
    ensure_directories(
        [
            config.data_dir,
            config.raw_dir,
            config.processed_dir,
            config.manual_import_dir,
            config.price_dir,
            config.benchmark_dir,
            config.external_dir,
            config.fundamentals_dir,
            config.company_dir,
            config.flow_dir,
            config.events_dir,
            config.artifacts_dir,
            config.models_dir,
            config.reports_dir,
            config.recommendations_dir,
        ]
    )


def _persist_production_model_stack(model_stack: ModelStack, config: TradingBotConfig) -> None:
    joblib.dump(model_stack.alpha, config.alpha_model_path)
    joblib.dump(model_stack.risk, config.risk_model_path)
    joblib.dump(model_stack.regime, config.regime_model_path)


def _load_production_model_stack(config: TradingBotConfig) -> ModelStack:
    if not config.alpha_model_path.exists() or not config.risk_model_path.exists() or not config.regime_model_path.exists():
        train_system(config=config, refresh_data=False)
    return ModelStack(
        alpha=joblib.load(config.alpha_model_path),
        risk=joblib.load(config.risk_model_path),
        regime=joblib.load(config.regime_model_path),
    )


def _load_manifest(config: TradingBotConfig) -> dict[str, Any]:
    if not config.deployment_manifest_path.exists():
        train_system(config=config, refresh_data=False)
    return read_json(config.deployment_manifest_path)


def _policy_from_manifest(config: TradingBotConfig, manifest: dict[str, Any]):
    from rdtb.portfolio.optimizer import PolicyParameters, default_policy

    payload = manifest.get("best_policy")
    if not isinstance(payload, dict):
        return default_policy(config)
    return PolicyParameters(**payload)
