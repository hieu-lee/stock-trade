from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rdtb.backtest.engine import BacktestResult, run_backtest
from rdtb.config import TradingBotConfig
from rdtb.models.train import ModelStack, score_panel, train_model_stack
from rdtb.portfolio.optimizer import PolicyParameters, default_policy


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


def run_strict_validation(panel: pd.DataFrame, config: TradingBotConfig) -> ValidationArtifacts:
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
        scored_valid = score_panel(fold_valid, model_artifacts.model_stack)
        scored_folds.append(((train_start, train_end, valid_start, valid_end), scored_valid))
    best_policy = _tune_policy(scored_folds, config)

    development_train = _restrict_recent_training_window(development, anchor_date=pd.Timestamp(f"{min(config.final_test_years)}-01-01"), config=config)
    development_models = train_model_stack(development_train, config, persist=False)
    final_scored_panel = score_panel(final_test, development_models.model_stack)
    final_backtest = run_backtest(final_scored_panel, config, policy=best_policy)

    for (train_start, train_end, valid_start, valid_end), scored_valid in scored_folds:
        fold_backtest = run_backtest(scored_valid, config, policy=best_policy)
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

    def objective(trial: optuna.Trial) -> float:
        policy = PolicyParameters(
            max_positions=trial.suggest_int("max_positions", 5, 10),
            max_weight=trial.suggest_float("max_weight", 0.10, 0.25),
            risk_penalty=trial.suggest_float("risk_penalty", 0.10, 0.80),
            turnover_penalty=trial.suggest_float("turnover_penalty", 0.05, 0.50),
            concentration_penalty=trial.suggest_float("concentration_penalty", 0.01, 0.20),
            cash_floor=trial.suggest_float("cash_floor", 0.05, 0.20),
            risk_on_threshold=trial.suggest_float("risk_on_threshold", 0.50, 0.65),
            defensive_threshold=trial.suggest_float("defensive_threshold", 0.35, 0.50),
            buy_threshold=trial.suggest_float("buy_threshold", 0.50, 0.70),
            add_threshold=trial.suggest_float("add_threshold", 0.60, 0.80),
            exit_threshold=trial.suggest_float("exit_threshold", 0.35, 0.50),
            trim_threshold=trial.suggest_float("trim_threshold", 0.40, 0.60),
            risk_reject_threshold=trial.suggest_float("risk_reject_threshold", 0.45, 0.70),
        )
        fold_scores: list[float] = []
        fold_exposures: list[float] = []
        worst_year = 10.0
        worst_drawdown = 0.0
        for _, scored_valid in scored_folds:
            backtest = run_backtest(scored_valid, config, policy=policy)
            metrics = backtest.metrics
            yearly = metrics.get("yearly_returns", {})
            if isinstance(yearly, dict) and yearly:
                worst_year = min(worst_year, min(float(value) for value in yearly.values()))
            worst_drawdown = min(worst_drawdown, float(metrics["max_drawdown"]))
            fold_scores.append(float(metrics["annualized_return"]))
            fold_exposures.append(float(metrics["avg_gross_exposure"]))
        avg_return = float(np.mean(fold_scores)) if fold_scores else -1.0
        avg_exposure = float(np.mean(fold_exposures)) if fold_exposures else 0.0
        goal_penalty = max(config.deployment_min_year_return - worst_year, 0.0) * 6.0
        drawdown_penalty = max(config.deployment_max_drawdown - worst_drawdown, 0.0) * 8.0
        exposure_penalty = max(0.25 - avg_exposure, 0.0) * 2.0
        return avg_return + worst_year * 2.0 + avg_exposure * 0.20 - goal_penalty - drawdown_penalty - exposure_penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout_seconds)
    best = study.best_params if study.best_trial is not None else {}
    if not best:
        return base
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
    )
