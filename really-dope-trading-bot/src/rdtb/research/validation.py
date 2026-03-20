from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import statistics

import numpy as np
import pandas as pd

from rdtb.backtest.engine import BacktestResult, run_backtest
from rdtb.config import TradingBotConfig
from rdtb.models.train import ModelStack, score_panel, train_model_stack
from rdtb.portfolio.optimizer import PolicyParameters, default_policy
from rdtb.research.search_scoring import score_search_summary
from rdtb.research.validation_matrix import _run_monthly_rebalance_baseline


@dataclass(slots=True)
class FoldSummary:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    annualized_return: float
    max_drawdown: float
    worst_year_return: float


@dataclass(slots=True)
class ValidationArtifacts:
    best_policy: PolicyParameters
    fold_summaries: list[FoldSummary]
    final_backtest: BacktestResult
    final_scored_panel: pd.DataFrame
    production_model_stack: ModelStack
    deployable: bool
    scored_folds: list[tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.DataFrame]]


def run_strict_validation(
    panel: pd.DataFrame,
    config: TradingBotConfig,
    reference_policy: PolicyParameters | None = None,
    utility_weights: Mapping[str, float] | None = None,
    tune_policy: bool = True,
) -> ValidationArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy()
    if trainable.empty:
        raise ValueError("No trainable rows available for strict validation.")
    development, final_test = _split_development_and_final_test(trainable, config)
    folds = _build_walk_forward_folds(development, config)
    scored_folds: list[tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.DataFrame]] = []
    fold_summaries: list[FoldSummary] = []
    for train_start, train_end, valid_start, valid_end in folds:
        fold_train = development.loc[(development["date"] >= train_start) & (development["date"] <= train_end)].copy()
        fold_train = _restrict_recent_training_window(fold_train, anchor_date=valid_start, config=config)
        fold_valid = development.loc[(development["date"] >= valid_start) & (development["date"] <= valid_end)].copy()
        model_artifacts = train_model_stack(fold_train, config, persist=False)
        scored_valid = score_panel(fold_valid, model_artifacts.model_stack, utility_weights=utility_weights)
        scored_folds.append(((train_start, train_end, valid_start, valid_end), scored_valid))
    best_policy = reference_policy or (_tune_policy(scored_folds, config) if tune_policy else default_policy(config))

    development_train = _prepare_final_training_frame(development, config)
    development_models = train_model_stack(development_train, config, persist=False)
    final_scored_panel = score_panel(final_test, development_models.model_stack, utility_weights=utility_weights)
    final_backtest = run_backtest(final_scored_panel, config, policy=best_policy)
    fold_summaries = _build_fold_summaries(scored_folds, config, best_policy)

    production_cutoff = max(config.final_test_years)
    production_train = trainable.loc[pd.to_datetime(trainable["date"]).dt.year <= production_cutoff].copy()
    production_train = _restrict_recent_training_window(production_train, anchor_date=pd.Timestamp(f"{production_cutoff + 1}-01-01"), config=config)
    production_model_stack = train_model_stack(production_train, config, persist=False).model_stack
    deployable = evaluate_deployability(final_backtest.metrics, config)
    return ValidationArtifacts(
        best_policy=best_policy,
        fold_summaries=fold_summaries,
        final_backtest=final_backtest,
        final_scored_panel=final_scored_panel,
        production_model_stack=production_model_stack,
        deployable=deployable,
        scored_folds=scored_folds,
    )


def evaluate_deployability(metrics: dict[str, object], config: TradingBotConfig) -> bool:
    yearly_returns = metrics.get("yearly_returns", {})
    if not isinstance(yearly_returns, dict):
        return False
    required_years = {str(year) for year in config.final_test_years}
    if not required_years.issubset(yearly_returns.keys()):
        return False
    if any(float(yearly_returns[year]) < config.deployment_min_year_return for year in required_years):
        return False
    return float(metrics.get("max_drawdown", 0.0)) >= config.deployment_max_drawdown


def validation_to_dict(artifacts: ValidationArtifacts) -> dict[str, object]:
    return {
        "best_policy": asdict(artifacts.best_policy),
        "fold_summaries": [asdict(summary) for summary in artifacts.fold_summaries],
        "final_metrics": artifacts.final_backtest.metrics,
        "deployable": artifacts.deployable,
    }


def replay_validation_artifacts(
    artifacts: ValidationArtifacts,
    config: TradingBotConfig,
    policy: PolicyParameters,
) -> ValidationArtifacts:
    if asdict(policy) == asdict(artifacts.best_policy):
        return artifacts
    final_backtest = run_backtest(artifacts.final_scored_panel, config, policy=policy)
    return ValidationArtifacts(
        best_policy=policy,
        fold_summaries=_build_fold_summaries(artifacts.scored_folds, config, policy),
        final_backtest=final_backtest,
        final_scored_panel=artifacts.final_scored_panel,
        production_model_stack=artifacts.production_model_stack,
        deployable=evaluate_deployability(final_backtest.metrics, config),
        scored_folds=artifacts.scored_folds,
    )


def _split_development_and_final_test(
    panel: pd.DataFrame,
    config: TradingBotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.to_datetime(panel["date"]).dt.year
    development = panel.loc[(years >= config.development_start_year) & (years <= config.development_end_year)].copy()
    final_test = panel.loc[years.isin(config.final_test_years)].copy()
    if development.empty or final_test.empty:
        raise ValueError("The panel does not contain enough history for the requested development/final-test split.")
    final_years = set(pd.to_datetime(final_test["date"]).dt.year.unique())
    if final_years != set(config.final_test_years):
        raise ValueError("The final test set must contain the full untouched years required by the deployment contract.")
    return development.sort_values(["date", "symbol"]).reset_index(drop=True), final_test.sort_values(["date", "symbol"]).reset_index(drop=True)


def _prepare_final_training_frame(panel: pd.DataFrame, config: TradingBotConfig) -> pd.DataFrame:
    final_test_start = pd.Timestamp(f"{min(config.final_test_years)}-01-01")
    train_cutoff = final_test_start - pd.Timedelta(days=config.fold_embargo_days)
    train_frame = panel.loc[pd.to_datetime(panel["date"]) <= train_cutoff].copy()
    train_frame = _restrict_recent_training_window(train_frame, anchor_date=final_test_start, config=config)
    if train_frame.empty:
        raise ValueError("No development rows remain after applying the final-test embargo.")
    return train_frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _build_walk_forward_folds(panel: pd.DataFrame, config: TradingBotConfig) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(panel["date"]).sort_values().unique()
    years = sorted(set(pd.DatetimeIndex(dates).year.tolist()))
    min_valid_year = config.development_start_year + config.fold_min_train_years
    validation_years = [year for year in years if year >= min_valid_year]
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for year in validation_years:
        valid_start = pd.Timestamp(f"{year}-01-01")
        valid_end = pd.Timestamp(f"{year}-12-31")
        train_start = pd.Timestamp(f"{config.development_start_year}-01-01")
        train_end = valid_start - pd.Timedelta(days=config.fold_embargo_days)
        if train_end <= train_start:
            continue
        if not ((panel["date"] >= valid_start) & (panel["date"] <= valid_end)).any():
            continue
        folds.append((train_start, train_end, valid_start, valid_end))
    if not folds:
        raise ValueError("No walk-forward folds could be generated for the development period.")
    return folds


def _restrict_recent_training_window(panel: pd.DataFrame, anchor_date: pd.Timestamp, config: TradingBotConfig) -> pd.DataFrame:
    if panel.empty:
        return panel
    window_start = anchor_date - pd.DateOffset(years=config.max_train_years)
    trimmed = panel.loc[pd.to_datetime(panel["date"]) >= window_start].copy()
    return trimmed if not trimmed.empty else panel


def _tune_policy(
    scored_folds: list[tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.DataFrame]],
    config: TradingBotConfig,
) -> PolicyParameters:
    base = default_policy(config)
    try:
        import optuna
    except Exception:  # pragma: no cover - optional runtime dependency
        return base

    def score_policy(policy: PolicyParameters) -> float:
        fold_returns: list[float] = []
        fold_sharpes: list[float] = []
        fold_exposures: list[float] = []
        equal_weight_gaps: list[float] = []
        momentum_gaps: list[float] = []
        fold_worst_years: list[float] = []
        worst_drawdown = 0.0
        for _, scored_valid in scored_folds:
            backtest = run_backtest(scored_valid, config, policy=policy)
            metrics = backtest.metrics
            equal_weight = _run_monthly_rebalance_baseline(scored_valid, config, signal_column=None, top_n=None)
            momentum = _run_monthly_rebalance_baseline(
                scored_valid,
                config,
                signal_column="relative_strength_20d",
                top_n=max(int(policy.max_positions), 1),
            )
            yearly = metrics.get("yearly_returns", {})
            if isinstance(yearly, dict) and yearly:
                fold_worst_years.append(min(float(value) for value in yearly.values()))
            else:
                fold_worst_years.append(float(metrics["annualized_return"]))
            worst_drawdown = min(worst_drawdown, float(metrics["max_drawdown"]))
            annualized_return = float(metrics["annualized_return"])
            fold_returns.append(annualized_return)
            fold_sharpes.append(float(metrics["sharpe"]))
            fold_exposures.append(float(metrics["avg_gross_exposure"]))
            equal_weight_gaps.append(annualized_return - float(equal_weight["annualized_return"]))
            momentum_gaps.append(annualized_return - float(momentum["annualized_return"]))

        median_return = float(statistics.median(fold_returns)) if fold_returns else 0.0
        median_sharpe = float(statistics.median(fold_sharpes)) if fold_sharpes else 0.0
        avg_exposure = float(np.mean(fold_exposures)) if fold_exposures else 0.0
        median_equal_weight_gap = float(statistics.median(equal_weight_gaps)) if equal_weight_gaps else 0.0
        median_momentum_gap = float(statistics.median(momentum_gaps)) if momentum_gaps else 0.0
        median_worst_year = float(statistics.median(fold_worst_years)) if fold_worst_years else 0.0
        worst_year = float(min(fold_worst_years)) if fold_worst_years else 0.0
        summary = {
            "median_holdout_annualized_return": median_return,
            "median_holdout_sharpe": median_sharpe,
            "median_holdout_worst_year_return": median_worst_year,
            "worst_holdout_year_return": worst_year,
            "worst_holdout_max_drawdown": worst_drawdown,
            "beat_benchmark_count": 0,
            "beat_equal_weight_count": sum(gap > 0.0 for gap in equal_weight_gaps),
            "beat_momentum_count": sum(gap > 0.0 for gap in momentum_gaps),
            "median_momentum_gap": median_momentum_gap,
            "median_high_friction_annualized_return": 0.0,
            "median_signal_delay_annualized_return": 0.0,
        }
        score = score_search_summary(summary, include_stress=False, config=config)
        score -= max(0.20 - avg_exposure, 0.0) * 0.75
        score -= max(avg_exposure - 0.90, 0.0) * 0.50
        score += median_equal_weight_gap * 1.25
        return score

    def objective(trial: optuna.Trial) -> float:
        policy = PolicyParameters(
            max_positions=trial.suggest_int("max_positions", 5, 8),
            max_weight=trial.suggest_float("max_weight", 0.10, 0.26),
            risk_penalty=trial.suggest_float("risk_penalty", 0.10, 0.70),
            turnover_penalty=trial.suggest_float("turnover_penalty", 0.05, 0.45),
            concentration_penalty=trial.suggest_float("concentration_penalty", 0.01, 0.18),
            cash_floor=trial.suggest_float("cash_floor", 0.0, 0.16),
            risk_on_threshold=trial.suggest_float("risk_on_threshold", 0.48, 0.70),
            defensive_threshold=trial.suggest_float("defensive_threshold", 0.20, 0.45),
            buy_threshold=trial.suggest_float("buy_threshold", 0.50, 0.72),
            add_threshold=trial.suggest_float("add_threshold", 0.60, 0.84),
            exit_threshold=trial.suggest_float("exit_threshold", 0.32, 0.55),
            trim_threshold=trial.suggest_float("trim_threshold", 0.40, 0.66),
            risk_reject_threshold=trial.suggest_float("risk_reject_threshold", 0.45, 0.74),
            defensive_gross_exposure=trial.suggest_float("defensive_gross_exposure", 0.08, 0.35),
            min_gross_exposure=trial.suggest_float("min_gross_exposure", 0.0, 0.06),
            risk_exit_threshold=trial.suggest_float("risk_exit_threshold", 0.64, 0.90),
            atr_stop_multiple=trial.suggest_float("atr_stop_multiple", 1.8, 3.8),
            regime_transition_slope=trial.suggest_float("regime_transition_slope", 3.0, 16.0),
        )
        return score_policy(policy)

    base_score = score_policy(base)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout_seconds)
    if study.best_trial is None or float(study.best_value) <= base_score:
        return base
    best = study.best_params
    return PolicyParameters(
        max_positions=int(best["max_positions"]),
        max_weight=float(best["max_weight"]),
        risk_penalty=float(best["risk_penalty"]),
        turnover_penalty=float(best["turnover_penalty"]),
        concentration_penalty=float(best["concentration_penalty"]),
        cash_floor=float(best["cash_floor"]),
        risk_on_threshold=float(best["risk_on_threshold"]),
        defensive_threshold=float(best["defensive_threshold"]),
        buy_threshold=float(best["buy_threshold"]),
        add_threshold=float(best["add_threshold"]),
        exit_threshold=float(best["exit_threshold"]),
        trim_threshold=float(best["trim_threshold"]),
        risk_reject_threshold=float(best["risk_reject_threshold"]),
        defensive_gross_exposure=float(best["defensive_gross_exposure"]),
        min_gross_exposure=float(best["min_gross_exposure"]),
        risk_exit_threshold=float(best["risk_exit_threshold"]),
        atr_stop_multiple=float(best["atr_stop_multiple"]),
        regime_transition_slope=float(best["regime_transition_slope"]),
    )


def _build_fold_summaries(
    scored_folds: list[tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.DataFrame]],
    config: TradingBotConfig,
    policy: PolicyParameters,
) -> list[FoldSummary]:
    fold_summaries: list[FoldSummary] = []
    for (train_start, train_end, valid_start, valid_end), scored_valid in scored_folds:
        fold_backtest = run_backtest(scored_valid, config, policy=policy)
        yearly = fold_backtest.metrics.get("yearly_returns", {})
        worst_year = min(yearly.values()) if yearly else 0.0
        fold_summaries.append(
            FoldSummary(
                train_start=pd.Timestamp(train_start).strftime("%Y-%m-%d"),
                train_end=pd.Timestamp(train_end).strftime("%Y-%m-%d"),
                valid_start=pd.Timestamp(valid_start).strftime("%Y-%m-%d"),
                valid_end=pd.Timestamp(valid_end).strftime("%Y-%m-%d"),
                annualized_return=float(fold_backtest.metrics["annualized_return"]),
                max_drawdown=float(fold_backtest.metrics["max_drawdown"]),
                worst_year_return=float(worst_year),
            )
        )
    return fold_summaries
