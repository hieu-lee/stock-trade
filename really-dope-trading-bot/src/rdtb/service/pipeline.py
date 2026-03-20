from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from rdtb.config import TradingBotConfig, get_default_config
from rdtb.data.collector import collect_market_data
from rdtb.features.panel import build_feature_panel
from rdtb.models.train import ModelStack, normalize_utility_weights, score_panel
from rdtb.portfolio.actions import build_daily_actions
from rdtb.portfolio.transactions import replay_transactions
from rdtb.research.validation import run_strict_validation, validation_to_dict
from rdtb.research.constant_search import (
    constant_search_to_dict,
    render_constant_search_markdown,
    run_constant_search,
)
from rdtb.research.validation_matrix import (
    render_validation_matrix_markdown,
    run_validation_matrix,
    validation_matrix_to_dict,
)
from rdtb.utils import ProgressCallback, ensure_directories, read_json, report_progress, write_json


def train_system(
    config: TradingBotConfig | None = None,
    refresh_data: bool = False,
    search_trials: int | None = None,
    search_timeout_seconds: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = config or get_default_config()
    _ensure_layout(config)
    report_progress(progress_callback, "Refreshing market data...", 0.0)
    bundle = collect_market_data(
        config=_research_collection_config(config),
        refresh=refresh_data,
        progress_callback=progress_callback,
        mode="full",
    )
    report_progress(progress_callback, "Running staged auto-search...", 0.25)
    search_report = run_constant_search(
        prices=bundle.prices,
        benchmarks=bundle.benchmarks,
        config=config,
        external_markets=bundle.external_markets,
        fundamentals=bundle.fundamentals,
        company_metadata=bundle.company_metadata,
        flow_history=bundle.flow_history,
        event_history=bundle.event_history,
        trials=search_trials if search_trials is not None else config.auto_search_trials,
        timeout_seconds=search_timeout_seconds if search_timeout_seconds is not None else config.auto_search_timeout_seconds,
        progress_callback=progress_callback,
        coarse_holdout_count=config.auto_search_coarse_holdout_count,
        finalist_count=config.auto_search_finalist_count,
    )
    write_json(config.constant_search_path, constant_search_to_dict(search_report))
    config.constant_search_markdown_path.write_text(render_constant_search_markdown(search_report), encoding="utf-8")

    selected_config = _config_from_overrides(config, search_report.best_config_overrides)
    selected_policy = _policy_from_payload(selected_config, search_report.best_policy)
    selected_utility_weights = normalize_utility_weights(search_report.best_utility_weights)

    report_progress(progress_callback, "Building selected daily feature panel...", 0.75)
    panel = build_feature_panel(
        prices=bundle.prices,
        benchmarks=bundle.benchmarks,
        config=selected_config,
        external_markets=bundle.external_markets,
        fundamentals=bundle.fundamentals,
        company_metadata=bundle.company_metadata if selected_config.use_company_metadata_features else None,
        flow_history=bundle.flow_history if selected_config.use_fireant_flow_features else None,
        event_history=bundle.event_history if selected_config.use_event_features else None,
        persist_path=config.feature_panel_path,
    )
    report_progress(progress_callback, "Running strict walk-forward validation...", 0.84)
    strict_validation = run_strict_validation(
        panel,
        selected_config,
        reference_policy=selected_policy,
        utility_weights=selected_utility_weights,
        tune_policy=False,
    )
    report_progress(progress_callback, "Running validation matrix deployment gate...", 0.92)
    validation_matrix = run_validation_matrix(
        panel=panel,
        prices=bundle.prices,
        benchmarks=bundle.benchmarks,
        config=selected_config,
        reference_policy=selected_policy,
        utility_weights=selected_utility_weights,
        progress_callback=progress_callback,
        policy_source="auto_search",
    )

    validation_matrix_payload = validation_matrix_to_dict(validation_matrix)
    write_json(config.validation_matrix_path, validation_matrix_payload)
    config.validation_matrix_markdown_path.write_text(render_validation_matrix_markdown(validation_matrix), encoding="utf-8")
    matrix_pass = bool(validation_matrix.summary.get("overall_pass", False))
    deployable = bool(strict_validation.deployable and matrix_pass)
    _persist_production_model_stack(strict_validation.production_model_stack, config)
    strict_validation.final_scored_panel.to_parquet(config.scored_panel_path, index=False)
    summary = validation_to_dict(strict_validation)
    summary["strict_validation_deployable"] = bool(strict_validation.deployable)
    summary["validation_matrix_policy_source"] = "auto_search"
    summary["validation_matrix_summary"] = validation_matrix.summary
    summary["auto_search_summary"] = search_report.best_summary
    summary["auto_search_final_summary"] = search_report.best_final_summary
    summary["auto_search_final_metrics"] = search_report.best_final_metrics
    summary["best_utility_weights"] = selected_utility_weights
    summary["best_config_overrides"] = search_report.best_config_overrides
    summary["target_beaten"] = bool(search_report.target_beaten)
    summary["deployable"] = deployable
    summary["feature_panel_rows"] = int(len(panel))
    summary["latest_feature_date"] = pd.to_datetime(panel["date"]).max()
    summary["latest_market_date"] = pd.to_datetime(bundle.prices["date"]).max()
    write_json(config.training_summary_path, summary)
    write_json(
        config.deployment_manifest_path,
        {
            "deployable": deployable,
            "strict_validation_deployable": strict_validation.deployable,
            "validation_matrix_pass": matrix_pass,
            "validation_matrix_policy_source": "auto_search",
            "validation_matrix_summary": validation_matrix.summary,
            "best_policy": asdict(selected_policy),
            "best_utility_weights": selected_utility_weights,
            "best_config_overrides": search_report.best_config_overrides,
            "search_summary": search_report.best_summary,
            "target_beaten": bool(search_report.target_beaten),
            "final_metrics": strict_validation.final_backtest.metrics,
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
    manifest = _load_manifest(config)
    effective_config = _config_from_manifest(config, manifest)
    utility_weights = _utility_weights_from_manifest(manifest)
    if refresh_data or not config.feature_panel_path.exists():
        report_progress(progress_callback, "Refreshing market data for recommendation...", 0.0)
        bundle = collect_market_data(
            config=_research_collection_config(effective_config),
            refresh=refresh_data,
            progress_callback=progress_callback,
            mode="decision",
        )
        panel = build_feature_panel(
            prices=bundle.prices,
            benchmarks=bundle.benchmarks,
            config=effective_config,
            external_markets=bundle.external_markets,
            fundamentals=bundle.fundamentals,
            company_metadata=bundle.company_metadata if effective_config.use_company_metadata_features else None,
            flow_history=bundle.flow_history if effective_config.use_fireant_flow_features else None,
            event_history=bundle.event_history if effective_config.use_event_features else None,
            persist_path=config.feature_panel_path,
        )
    else:
        panel = pd.read_parquet(config.feature_panel_path)
    model_stack = _load_production_model_stack(config)
    scored_panel = score_panel(panel, model_stack, utility_weights=utility_weights)
    latest_date = pd.to_datetime(scored_panel["date"]).max()
    latest_snapshot = scored_panel.loc[scored_panel["date"] == latest_date].copy().reset_index(drop=True)
    transaction_summary = None
    transaction_notes: list[str] = []
    effective_holdings = holdings if holdings is not None else build_holdings_template(effective_config)
    effective_cash = float(cash)
    if transactions is not None and not transactions.empty:
        trading_calendar = pd.to_datetime(scored_panel["date"]).drop_duplicates().sort_values().tolist()
        replay = replay_transactions(
            transactions=transactions,
            starting_cash=float(starting_cash if starting_cash is not None else cash),
            config=effective_config,
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
        config=effective_config,
        policy=_policy_from_manifest(effective_config, manifest),
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
                "regime_anchor_probability",
                "regime_participation_probability",
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


def validate_system(
    config: TradingBotConfig | None = None,
    refresh_data: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = config or get_default_config()
    _ensure_layout(config)
    manifest = read_json(config.deployment_manifest_path) if config.deployment_manifest_path.exists() else {}
    effective_config = _config_from_manifest(config, manifest)
    utility_weights = _utility_weights_from_manifest(manifest) if manifest else None
    bundle = None
    needs_bundle = (
        refresh_data
        or not config.feature_panel_path.exists()
        or not config.prices_path.exists()
        or not config.benchmarks_path.exists()
    )
    if needs_bundle:
        report_progress(progress_callback, "Refreshing market data for research validation...", 0.0)
        bundle = collect_market_data(
            config=_research_collection_config(effective_config),
            refresh=refresh_data,
            progress_callback=progress_callback,
            mode="full",
        )
        panel = build_feature_panel(
            prices=bundle.prices,
            benchmarks=bundle.benchmarks,
            config=effective_config,
            external_markets=bundle.external_markets,
            fundamentals=bundle.fundamentals,
            company_metadata=bundle.company_metadata if effective_config.use_company_metadata_features else None,
            flow_history=bundle.flow_history if effective_config.use_fireant_flow_features else None,
            event_history=bundle.event_history if effective_config.use_event_features else None,
            persist_path=config.feature_panel_path,
        )
        prices = bundle.prices
        benchmarks = bundle.benchmarks
    else:
        report_progress(progress_callback, "Loading cached feature panel for research validation...", 0.0)
        panel = pd.read_parquet(config.feature_panel_path)
        prices = pd.read_parquet(config.prices_path)
        benchmarks = pd.read_parquet(config.benchmarks_path)

    reference_policy = None
    policy_source = "default"
    payload = manifest.get("best_policy") if manifest else None
    if isinstance(payload, dict):
        reference_policy = _policy_from_manifest(effective_config, manifest)
        policy_source = "deployment_manifest"

    report_progress(progress_callback, "Running repeated holdouts, baselines, and stress scenarios...", 0.10)
    report = run_validation_matrix(
        panel=panel,
        prices=prices,
        benchmarks=benchmarks,
        config=effective_config,
        reference_policy=reference_policy,
        utility_weights=utility_weights,
        progress_callback=progress_callback,
        policy_source=policy_source,
    )
    payload = validation_matrix_to_dict(report)
    write_json(config.validation_matrix_path, payload)
    config.validation_matrix_markdown_path.write_text(render_validation_matrix_markdown(report), encoding="utf-8")
    report_progress(progress_callback, "Research validation report completed.", 1.0)
    return payload


def search_constants_system(
    config: TradingBotConfig | None = None,
    refresh_data: bool = False,
    trials: int = 60,
    timeout_seconds: int | None = None,
    target_summary: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = config or get_default_config()
    _ensure_layout(config)
    bundle = None
    needs_bundle = (
        refresh_data
        or not config.prices_path.exists()
        or not config.benchmarks_path.exists()
    )
    if needs_bundle:
        report_progress(progress_callback, "Refreshing market data for constant search...", 0.0)
        bundle = collect_market_data(
            config=_research_collection_config(config),
            refresh=refresh_data,
            progress_callback=progress_callback,
            mode="full",
        )
    else:
        report_progress(progress_callback, "Loading cached raw data for constant search...", 0.0)
        prices = pd.read_parquet(config.prices_path)
        benchmarks = pd.read_parquet(config.benchmarks_path)
        external_markets = pd.read_parquet(config.external_markets_path) if config.external_markets_path.exists() else None
        fundamentals = pd.read_parquet(config.fundamentals_path) if config.fundamentals_path.exists() else None
        company_metadata = pd.read_parquet(config.company_metadata_path) if config.company_metadata_path.exists() else None
        flow_history = pd.read_parquet(config.flow_path) if config.flow_path.exists() else None
        event_history = pd.read_parquet(config.events_path) if config.events_path.exists() else None
        bundle = type(
            "CachedBundle",
            (),
            {
                "prices": prices,
                "benchmarks": benchmarks,
                "external_markets": external_markets,
                "fundamentals": fundamentals,
                "company_metadata": company_metadata,
                "flow_history": flow_history,
                "event_history": event_history,
            },
        )()

    report_progress(progress_callback, "Precomputing holdouts and searching constants...", 0.05)
    report = run_constant_search(
        prices=bundle.prices,
        benchmarks=bundle.benchmarks,
        config=config,
        external_markets=bundle.external_markets,
        fundamentals=bundle.fundamentals,
        company_metadata=bundle.company_metadata,
        flow_history=bundle.flow_history,
        event_history=bundle.event_history,
        trials=trials,
        timeout_seconds=timeout_seconds,
        target_summary=target_summary,
        progress_callback=progress_callback,
    )
    payload = constant_search_to_dict(report)
    write_json(config.constant_search_path, payload)
    config.constant_search_markdown_path.write_text(render_constant_search_markdown(report), encoding="utf-8")
    report_progress(progress_callback, "Constant search report completed.", 1.0)
    return payload


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


def _config_from_overrides(config: TradingBotConfig, overrides: dict[str, Any] | None) -> TradingBotConfig:
    if not isinstance(overrides, dict) or not overrides:
        return config
    return replace(config, **overrides)


def _config_from_manifest(config: TradingBotConfig, manifest: dict[str, Any]) -> TradingBotConfig:
    return _config_from_overrides(config, manifest.get("best_config_overrides"))


def _policy_from_manifest(config: TradingBotConfig, manifest: dict[str, Any]):
    return _policy_from_payload(config, manifest.get("best_policy"))


def _policy_from_payload(config: TradingBotConfig, payload: dict[str, Any] | None):
    from rdtb.portfolio.optimizer import PolicyParameters, default_policy

    if not isinstance(payload, dict):
        return default_policy(config)
    return PolicyParameters(**payload)


def _utility_weights_from_manifest(manifest: dict[str, Any]) -> dict[str, float] | None:
    payload = manifest.get("best_utility_weights")
    if not isinstance(payload, dict):
        return None
    return normalize_utility_weights(payload)


def _research_collection_config(config: TradingBotConfig) -> TradingBotConfig:
    return replace(config, use_company_metadata_features=True)
