from __future__ import annotations

from rdtb.config import TradingBotConfig


def score_search_summary(
    summary: dict[str, object],
    target_summary: dict[str, object] | None = None,
    include_stress: bool = True,
    config: TradingBotConfig | None = None,
) -> float:
    min_year_floor = float(config.deployment_min_year_return) if config is not None else 0.20
    aspirational_year_return = float(config.aspirational_year_return) if config is not None else 0.30
    drawdown_floor = float(config.deployment_max_drawdown) if config is not None else -0.15

    median_return = float(summary.get("median_holdout_annualized_return", 0.0))
    median_sharpe = float(summary.get("median_holdout_sharpe", 0.0))
    median_worst_year = float(summary.get("median_holdout_worst_year_return", median_return))
    worst_year = float(summary.get("worst_holdout_year_return", median_worst_year))
    worst_drawdown = float(summary.get("worst_holdout_max_drawdown", 0.0))
    beat_benchmark_count = float(summary.get("beat_benchmark_count", 0.0))
    beat_equal_weight_count = float(summary.get("beat_equal_weight_count", 0.0))
    beat_momentum_count = float(summary.get("beat_momentum_count", 0.0))
    median_momentum_gap = float(summary.get("median_momentum_gap", 0.0))

    score = (
        median_return * 12.0
        + median_sharpe * 1.0
        + median_worst_year * 15.0
        + worst_year * 22.0
        + worst_drawdown * 12.0
        + beat_benchmark_count * 0.30
        + beat_equal_weight_count * 0.25
        + beat_momentum_count * 0.10
        + median_momentum_gap * 0.60
    )
    score -= max(min_year_floor - median_return, 0.0) * 28.0
    score -= max(min_year_floor - median_worst_year, 0.0) * 36.0
    score -= max(min_year_floor - worst_year, 0.0) * 46.0
    score -= max(drawdown_floor - worst_drawdown, 0.0) * 40.0
    score += max(median_return - aspirational_year_return, 0.0) * 16.0
    score += max(median_worst_year - aspirational_year_return, 0.0) * 12.0
    score += max(worst_year - aspirational_year_return, 0.0) * 16.0

    if include_stress:
        high_friction = float(summary.get("median_high_friction_annualized_return", 0.0))
        signal_delay = float(summary.get("median_signal_delay_annualized_return", 0.0))
        score += high_friction * 4.0 + signal_delay * 2.0
        score -= max(-high_friction, 0.0) * 8.0
        score -= max(-signal_delay, 0.0) * 4.0

    if not target_summary:
        return score

    comparison_weights = {
        "median_holdout_annualized_return": 14.0,
        "median_holdout_sharpe": 2.5,
        "median_holdout_worst_year_return": 16.0,
        "worst_holdout_year_return": 18.0,
        "worst_holdout_max_drawdown": 10.0,
        "beat_benchmark_count": 0.8,
        "beat_equal_weight_count": 0.7,
        "beat_momentum_count": 0.4,
        "median_high_friction_annualized_return": 6.0,
        "median_signal_delay_annualized_return": 3.0,
    }
    for key, weight in comparison_weights.items():
        if key not in target_summary:
            continue
        score -= max(float(target_summary[key]) - float(summary.get(key, 0.0)), 0.0) * weight
    return score


def search_summary_beats_target(
    summary: dict[str, object],
    target_summary: dict[str, object] | None,
) -> bool:
    if not target_summary:
        return False
    for key, value in target_summary.items():
        if key not in summary:
            continue
        if float(summary[key]) < float(value):
            return False
    return True
