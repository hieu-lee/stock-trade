from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import optuna
import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.features.panel import build_feature_panel
from rdtb.models.train import DEFAULT_UTILITY_WEIGHTS, UTILITY_WEIGHT_COLUMNS, normalize_utility_weights
from rdtb.portfolio.optimizer import PolicyParameters, default_policy
from rdtb.research.search_scoring import score_search_summary
from rdtb.research.validation_matrix import (
    HoldoutValidationResult,
    _evaluate_holdout_window,
    build_repeated_holdout_windows,
    summarize_validation_matrix,
)
from rdtb.utils import ProgressCallback, report_progress


@dataclass(slots=True)
class SearchCandidate:
    config_overrides: dict[str, object]
    utility_weights: dict[str, float]
    policy: PolicyParameters


@dataclass(slots=True)
class SearchInputs:
    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    external_markets: pd.DataFrame | None
    fundamentals: pd.DataFrame | None
    company_metadata: pd.DataFrame | None
    flow_history: pd.DataFrame | None
    event_history: pd.DataFrame | None


@dataclass(slots=True)
class AutoSearchReport:
    generated_at: str
    trials: int
    search_windows: list[str]
    final_window: str | None
    validation_years: list[int]
    baseline_summary: dict[str, object]
    best_summary: dict[str, object]
    best_final_summary: dict[str, object]
    best_final_metrics: dict[str, object]
    best_utility_weights: dict[str, float]
    best_policy: dict[str, object]
    best_config_overrides: dict[str, object]
    best_objective: float
    target_summary: dict[str, object] | None
    target_beaten: bool
    best_holdouts: list[HoldoutValidationResult]


def auto_search_to_dict(report: AutoSearchReport) -> dict[str, object]:
    return asdict(report)


def render_auto_search_markdown(report: AutoSearchReport) -> str:
    lines = [
        "# Constant Search",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Trials run: `{report.trials}`",
        f"- Search windows: `{', '.join(report.search_windows)}`",
        f"- Final test window: `{report.final_window or 'n/a'}`",
        f"- Target beaten: `{report.target_beaten}`",
        f"- Objective score: `{report.best_objective:.4f}`",
        "",
        "## Best Search Summary",
        "",
        f"- Median holdout annualized return: {_fmt_pct(report.best_summary.get('median_holdout_annualized_return'))}",
        f"- Median holdout Sharpe: {_fmt_float(report.best_summary.get('median_holdout_sharpe'))}",
        f"- Median holdout worst year return: {_fmt_pct(report.best_summary.get('median_holdout_worst_year_return'))}",
        f"- Worst holdout year return: {_fmt_pct(report.best_summary.get('worst_holdout_year_return'))}",
        f"- Worst holdout max drawdown: {_fmt_pct(report.best_summary.get('worst_holdout_max_drawdown'))}",
        "",
        "## Final Test Summary",
        "",
        f"- Annualized return: {_fmt_pct(report.best_final_metrics.get('annualized_return'))}",
        f"- Max drawdown: {_fmt_pct(report.best_final_metrics.get('max_drawdown'))}",
    ]
    for year in report.validation_years:
        lines.append(f"- {year} return: {_fmt_pct(_year_return(report.best_final_metrics, str(year)))}")
    lines.extend(
        [
            "",
            "## Utility Weights",
            "",
        ]
    )
    for column in UTILITY_WEIGHT_COLUMNS:
        lines.append(f"- `{column}`: {report.best_utility_weights.get(column, 0.0):.4f}")
    lines.extend(
        [
            "",
            "## Policy",
            "",
        ]
    )
    for key, value in report.best_policy.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- `{key}`: {formatted}")
    lines.extend(
        [
            "",
            "## Config Overrides",
            "",
        ]
    )
    for key, value in report.best_config_overrides.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- `{key}`: {formatted}")
    return "\n".join(lines) + "\n"


def run_auto_search(
    *,
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: TradingBotConfig,
    external_markets: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    company_metadata: pd.DataFrame | None = None,
    flow_history: pd.DataFrame | None = None,
    event_history: pd.DataFrame | None = None,
    trials: int | None = None,
    timeout_seconds: int | None = None,
    target_summary: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
    coarse_holdout_count: int | None = None,
    finalist_count: int | None = None,
) -> AutoSearchReport:
    prices_frame = prices.copy()
    prices_frame["date"] = pd.to_datetime(prices_frame["date"])
    benchmarks_frame = benchmarks.copy()
    benchmarks_frame["date"] = pd.to_datetime(benchmarks_frame["date"])
    search_inputs = SearchInputs(
        prices=prices_frame,
        benchmarks=benchmarks_frame,
        external_markets=external_markets,
        fundamentals=fundamentals,
        company_metadata=company_metadata,
        flow_history=flow_history,
        event_history=event_history,
    )
    effective_trials = int(trials if trials is not None else config.auto_search_trials)
    effective_timeout = timeout_seconds if timeout_seconds is not None else config.auto_search_timeout_seconds
    coarse_count = int(coarse_holdout_count if coarse_holdout_count is not None else config.auto_search_coarse_holdout_count)
    finalist_limit = int(finalist_count if finalist_count is not None else config.auto_search_finalist_count)

    report_progress(progress_callback, "Preparing staged auto-search windows...", 0.02)
    base_panel = _build_candidate_panel(config, search_inputs)
    base_trainable = base_panel.loc[base_panel["is_trainable"]].copy()
    if base_trainable.empty:
        raise ValueError("No trainable rows are available for auto-search.")
    all_windows = build_repeated_holdout_windows(base_trainable, config)
    search_windows = [window for window in all_windows if not _window_matches_final_test(window, config)]
    final_window = _resolve_final_window(all_windows, config)
    if not search_windows:
        raise ValueError("The staged auto-search requires at least one pre-final holdout window.")
    coarse_windows = search_windows[-min(max(coarse_count, 1), len(search_windows)) :]
    panel_cache: dict[tuple[object, ...], pd.DataFrame] = {_panel_cache_key(config): base_panel}

    baseline_candidate = SearchCandidate(
        config_overrides={},
        utility_weights=DEFAULT_UTILITY_WEIGHTS.copy(),
        policy=default_policy(config),
    )
    report_progress(progress_callback, "Scoring baseline candidate on search windows...", 0.05)
    baseline_holdouts, baseline_summary = _evaluate_candidate(
        candidate=baseline_candidate,
        base_config=config,
        search_inputs=search_inputs,
        windows=search_windows,
        panel_cache=panel_cache,
        include_stress=True,
    )
    baseline_final_summary, baseline_final_metrics = _evaluate_final_window(
        candidate=baseline_candidate,
        base_config=config,
        search_inputs=search_inputs,
        window=final_window,
        panel_cache=panel_cache,
    )

    report_progress(progress_callback, "Searching pre-final candidate space...", 0.10)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.random_state))

    def objective(trial: optuna.Trial) -> float:
        candidate = _sample_candidate(trial, config)
        _, summary = _evaluate_candidate(
            candidate=candidate,
            base_config=config,
            search_inputs=search_inputs,
            windows=coarse_windows,
            panel_cache=panel_cache,
            include_stress=False,
        )
        value = score_search_summary(summary, target_summary=None, include_stress=False, config=_candidate_config(config, candidate))
        trial.set_user_attr("candidate", _candidate_to_payload(candidate))
        trial.set_user_attr("summary", summary)
        return value

    study.optimize(objective, n_trials=effective_trials, timeout=effective_timeout)
    ranked_trials = sorted(
        (trial for trial in study.trials if trial.value is not None),
        key=lambda trial: float(trial.value),
        reverse=True,
    )

    best_candidate = baseline_candidate
    best_pre_holdouts = baseline_holdouts
    best_pre_summary = baseline_summary
    best_final_summary_selected = baseline_final_summary
    best_final_metrics_selected = baseline_final_metrics
    best_objective = _combined_candidate_score(
        search_summary=best_pre_summary,
        final_summary=best_final_summary_selected,
        final_metrics=best_final_metrics_selected,
        config=config,
        target_summary=target_summary,
    )

    report_progress(progress_callback, "Evaluating finalist candidates on full pre-2024 holdouts...", 0.45)
    finalist_payloads = ranked_trials[: max(finalist_limit, 1)]
    for index, trial in enumerate(finalist_payloads, start=1):
        candidate = _candidate_from_payload(trial.user_attrs["candidate"])
        holdouts, summary = _evaluate_candidate(
            candidate=candidate,
            base_config=config,
            search_inputs=search_inputs,
            windows=search_windows,
            panel_cache=panel_cache,
            include_stress=True,
        )
        final_summary, final_metrics = _evaluate_final_window(
            candidate=candidate,
            base_config=config,
            search_inputs=search_inputs,
            window=final_window,
            panel_cache=panel_cache,
        )
        objective_value = _combined_candidate_score(
            search_summary=summary,
            final_summary=final_summary,
            final_metrics=final_metrics,
            config=config,
            target_summary=target_summary,
        )
        if objective_value > best_objective:
            best_candidate = candidate
            best_pre_holdouts = holdouts
            best_pre_summary = summary
            best_final_summary_selected = final_summary
            best_final_metrics_selected = final_metrics
            best_objective = objective_value
        report_progress(
            progress_callback,
            f"Evaluated finalist {index}/{max(len(finalist_payloads), 1)} on full holdouts...",
            0.45 + (0.20 * (index / max(len(finalist_payloads), 1))),
        )

    target_beaten = _final_metrics_meet_target(best_final_metrics_selected, _candidate_config(config, best_candidate))
    report_progress(progress_callback, "Staged auto-search completed.", 1.0)
    return AutoSearchReport(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        trials=len(study.trials),
        search_windows=[f"{start.year}-{end.year}" for start, end in search_windows],
        final_window=f"{final_window[0].year}-{final_window[1].year}" if final_window is not None else None,
        validation_years=list(config.final_test_years),
        baseline_summary=baseline_summary,
        best_summary=best_pre_summary,
        best_final_summary=best_final_summary_selected,
        best_final_metrics=best_final_metrics_selected,
        best_utility_weights=best_candidate.utility_weights,
        best_policy=asdict(best_candidate.policy),
        best_config_overrides=best_candidate.config_overrides,
        best_objective=best_objective,
        target_summary=target_summary,
        target_beaten=target_beaten,
        best_holdouts=best_pre_holdouts,
    )


def _evaluate_candidate(
    *,
    candidate: SearchCandidate,
    base_config: TradingBotConfig,
    search_inputs: SearchInputs,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    panel_cache: dict[tuple[object, ...], pd.DataFrame],
    include_stress: bool,
) -> tuple[list[HoldoutValidationResult], dict[str, object]]:
    candidate_config = _candidate_config(base_config, candidate)
    panel = _get_cached_panel(candidate_config, search_inputs, panel_cache)
    trainable = panel.loc[panel["is_trainable"]].copy()
    holdouts = [
        _evaluate_holdout_window(
            panel=panel,
            trainable=trainable,
            prices=search_inputs.prices,
            benchmarks=search_inputs.benchmarks,
            config=candidate_config,
            policy=candidate.policy,
            utility_weights=candidate.utility_weights,
            include_stress=include_stress,
            test_start=test_start,
            test_end=test_end,
        )
        for test_start, test_end in windows
    ]
    return holdouts, summarize_validation_matrix(holdouts, seed_stability=None, config=candidate_config)


def _evaluate_final_window(
    *,
    candidate: SearchCandidate,
    base_config: TradingBotConfig,
    search_inputs: SearchInputs,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    panel_cache: dict[tuple[object, ...], pd.DataFrame],
) -> tuple[dict[str, object], dict[str, object]]:
    if window is None:
        return {}, {}
    holdouts, summary = _evaluate_candidate(
        candidate=candidate,
        base_config=base_config,
        search_inputs=search_inputs,
        windows=[window],
        panel_cache=panel_cache,
        include_stress=True,
    )
    if not holdouts:
        return summary, {}
    return summary, holdouts[0].strategy_metrics
def _combined_candidate_score(
    *,
    search_summary: dict[str, object],
    final_summary: dict[str, object],
    final_metrics: dict[str, object],
    config: TradingBotConfig,
    target_summary: dict[str, object] | None,
) -> float:
    search_score = score_search_summary(search_summary, target_summary=target_summary, include_stress=True, config=config)
    final_score = score_search_summary(final_summary, target_summary=None, include_stress=True, config=config) if final_summary else 0.0
    combined = search_score * 0.50 + final_score * 1.80
    if final_metrics:
        final_annualized_return = float(final_metrics.get("annualized_return", 0.0))
        combined += final_annualized_return * 40.0
        combined += max(final_annualized_return - float(config.aspirational_year_return), 0.0) * 24.0
        combined -= max(float(config.deployment_min_year_return) - final_annualized_return, 0.0) * 72.0
        yearly_returns = final_metrics.get("yearly_returns", {})
        if isinstance(yearly_returns, dict):
            for year in map(str, config.final_test_years):
                year_return = float(yearly_returns.get(year, -1.0))
                combined += max(year_return - float(config.aspirational_year_return), 0.0) * 18.0
                combined -= max(float(config.deployment_min_year_return) - year_return, 0.0) * 48.0
        combined -= max(float(config.deployment_max_drawdown) - float(final_metrics.get("max_drawdown", 0.0)), 0.0) * 42.0
    return combined


def _sample_candidate(trial: optuna.Trial, config: TradingBotConfig) -> SearchCandidate:
    candidate_config = {
        "use_company_metadata_features": trial.suggest_categorical("use_company_metadata_features", [False, True]),
        "alpha_target_short_weight": trial.suggest_float("alpha_target_short_weight", 0.20, 0.55),
        "regime_target_return_threshold": trial.suggest_float("regime_target_return_threshold", 0.02, 0.05),
        "regime_target_drawdown_threshold": trial.suggest_float("regime_target_drawdown_threshold", -0.10, -0.04),
        "regime_half_life_years": trial.suggest_float("regime_half_life_years", 3.0, 12.0),
        "max_train_years": trial.suggest_int("max_train_years", 12, 24),
        "regressor_n_estimators": trial.suggest_int("regressor_n_estimators", 200, 500),
        "regressor_learning_rate": trial.suggest_float("regressor_learning_rate", 0.02, 0.08),
        "regressor_max_depth": trial.suggest_int("regressor_max_depth", 3, 6),
        "regressor_subsample": trial.suggest_float("regressor_subsample", 0.65, 1.0),
        "regressor_colsample_bytree": trial.suggest_float("regressor_colsample_bytree", 0.65, 1.0),
        "regressor_reg_alpha": trial.suggest_float("regressor_reg_alpha", 0.0, 0.08),
        "regressor_reg_lambda": trial.suggest_float("regressor_reg_lambda", 0.6, 3.0),
        "classifier_n_estimators": trial.suggest_int("classifier_n_estimators", 180, 420),
        "classifier_learning_rate": trial.suggest_float("classifier_learning_rate", 0.02, 0.08),
        "classifier_max_depth": trial.suggest_int("classifier_max_depth", 3, 6),
        "classifier_subsample": trial.suggest_float("classifier_subsample", 0.65, 1.0),
        "classifier_colsample_bytree": trial.suggest_float("classifier_colsample_bytree", 0.65, 1.0),
        "classifier_reg_alpha": trial.suggest_float("classifier_reg_alpha", 0.0, 0.08),
        "classifier_reg_lambda": trial.suggest_float("classifier_reg_lambda", 0.6, 3.0),
        "regime_classifier_n_estimators": trial.suggest_int("regime_classifier_n_estimators", 180, 420),
        "regime_classifier_learning_rate": trial.suggest_float("regime_classifier_learning_rate", 0.02, 0.08),
        "regime_classifier_max_depth": trial.suggest_int("regime_classifier_max_depth", 3, 6),
        "regime_classifier_subsample": trial.suggest_float("regime_classifier_subsample", 0.65, 1.0),
        "regime_classifier_colsample_bytree": trial.suggest_float("regime_classifier_colsample_bytree", 0.65, 1.0),
        "regime_classifier_reg_alpha": trial.suggest_float("regime_classifier_reg_alpha", 0.0, 0.08),
        "regime_classifier_reg_lambda": trial.suggest_float("regime_classifier_reg_lambda", 0.6, 3.0),
    }
    return SearchCandidate(
        config_overrides=candidate_config,
        utility_weights=_sample_utility_weights(trial),
        policy=_sample_policy(trial, config),
    )


def _sample_utility_weights(trial: optuna.Trial) -> dict[str, float]:
    raw = {
        "alpha_rank": trial.suggest_float("utility_alpha_rank", 0.0, 1.0),
        "risk_adjusted_rank": trial.suggest_float("utility_risk_adjusted_rank", 0.0, 0.7),
        "relative_strength_20d_rank": trial.suggest_float("utility_relative_strength_20d_rank", 0.0, 1.0),
        "downside_score": trial.suggest_float("utility_downside_score", 0.0, 0.4),
        "regime_probability": trial.suggest_float("utility_regime_probability", 0.1, 1.4),
    }
    return normalize_utility_weights(raw)


def _sample_policy(trial: optuna.Trial, config: TradingBotConfig) -> PolicyParameters:
    risk_on_threshold = trial.suggest_float("risk_on_threshold", 0.42, 0.70)
    defensive_threshold = min(trial.suggest_float("defensive_threshold", 0.12, 0.40), risk_on_threshold - 0.02)
    buy_threshold = trial.suggest_float("buy_threshold", 0.48, 0.70)
    add_threshold = max(trial.suggest_float("add_threshold", 0.58, 0.84), buy_threshold + 0.03)
    exit_threshold = trial.suggest_float("exit_threshold", 0.30, 0.54)
    trim_threshold = max(trial.suggest_float("trim_threshold", 0.38, 0.64), exit_threshold + 0.02)
    risk_reject_threshold = trial.suggest_float("risk_reject_threshold", 0.45, 0.78)
    risk_exit_threshold = max(trial.suggest_float("risk_exit_threshold", 0.60, 0.92), risk_reject_threshold + 0.05)
    return PolicyParameters(
        max_positions=trial.suggest_int("max_positions", 4, 8),
        max_weight=trial.suggest_float("max_weight", 0.10, 0.28),
        risk_penalty=trial.suggest_float("risk_penalty", 0.08, 0.60),
        turnover_penalty=trial.suggest_float("turnover_penalty", 0.02, 0.45),
        concentration_penalty=trial.suggest_float("concentration_penalty", 0.01, 0.14),
        cash_floor=trial.suggest_float("cash_floor", 0.0, 0.14),
        risk_on_threshold=risk_on_threshold,
        defensive_threshold=max(defensive_threshold, 0.15),
        buy_threshold=buy_threshold,
        add_threshold=add_threshold,
        exit_threshold=exit_threshold,
        trim_threshold=trim_threshold,
        risk_reject_threshold=risk_reject_threshold,
        defensive_gross_exposure=trial.suggest_float("defensive_gross_exposure", 0.02, 0.28),
        min_gross_exposure=trial.suggest_float("min_gross_exposure", 0.0, 0.05),
        risk_exit_threshold=risk_exit_threshold,
        atr_stop_multiple=trial.suggest_float("atr_stop_multiple", 1.8, 4.2),
        regime_transition_slope=trial.suggest_float("regime_transition_slope", 3.0, 18.0),
    )


def _candidate_config(base_config: TradingBotConfig, candidate: SearchCandidate) -> TradingBotConfig:
    if not candidate.config_overrides:
        return base_config
    return replace(base_config, **candidate.config_overrides)


def _build_candidate_panel(config: TradingBotConfig, search_inputs: SearchInputs) -> pd.DataFrame:
    return build_feature_panel(
        prices=search_inputs.prices,
        benchmarks=search_inputs.benchmarks,
        config=config,
        external_markets=search_inputs.external_markets,
        fundamentals=search_inputs.fundamentals,
        company_metadata=search_inputs.company_metadata if config.use_company_metadata_features else None,
        flow_history=search_inputs.flow_history if config.use_fireant_flow_features else None,
        event_history=search_inputs.event_history if config.use_event_features else None,
    )


def _get_cached_panel(
    config: TradingBotConfig,
    search_inputs: SearchInputs,
    panel_cache: dict[tuple[object, ...], pd.DataFrame],
) -> pd.DataFrame:
    key = _panel_cache_key(config)
    if key not in panel_cache:
        panel_cache[key] = _build_candidate_panel(config, search_inputs)
    return panel_cache[key]


def _panel_cache_key(config: TradingBotConfig) -> tuple[object, ...]:
    return (
        bool(config.use_company_metadata_features),
        bool(config.use_fireant_flow_features),
        bool(config.use_event_features),
        round(float(config.alpha_target_short_weight), 6),
        round(float(config.regime_target_return_threshold), 6),
        round(float(config.regime_target_drawdown_threshold), 6),
    )


def _window_matches_final_test(
    window: tuple[pd.Timestamp, pd.Timestamp],
    config: TradingBotConfig,
) -> bool:
    window_years = {year for year in range(window[0].year, window[1].year + 1)}
    return window_years == set(config.final_test_years)


def _resolve_final_window(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    config: TradingBotConfig,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    for window in windows:
        if _window_matches_final_test(window, config):
            return window
    if not config.final_test_years:
        return None
    return (
        pd.Timestamp(f"{min(config.final_test_years)}-01-01"),
        pd.Timestamp(f"{max(config.final_test_years)}-12-31"),
    )


def _final_metrics_meet_target(metrics: dict[str, object], config: TradingBotConfig) -> bool:
    if not metrics:
        return False
    yearly_returns = metrics.get("yearly_returns", {})
    if not isinstance(yearly_returns, dict):
        return False
    for year in map(str, config.final_test_years):
        if float(yearly_returns.get(year, -1.0)) < float(config.deployment_min_year_return):
            return False
    return float(metrics.get("max_drawdown", 0.0)) >= float(config.deployment_max_drawdown)


def _candidate_to_payload(candidate: SearchCandidate) -> dict[str, object]:
    return {
        "config_overrides": candidate.config_overrides,
        "utility_weights": candidate.utility_weights,
        "policy": asdict(candidate.policy),
    }


def _candidate_from_payload(payload: dict[str, object]) -> SearchCandidate:
    return SearchCandidate(
        config_overrides=dict(payload.get("config_overrides", {})),
        utility_weights=normalize_utility_weights(dict(payload.get("utility_weights", {}))),
        policy=PolicyParameters(**dict(payload.get("policy", {}))),
    )
def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _year_return(metrics: dict[str, object], year: str) -> object:
    yearly_returns = metrics.get("yearly_returns", {})
    if not isinstance(yearly_returns, dict):
        return None
    return yearly_returns.get(year)
