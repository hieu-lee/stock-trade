from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import statistics

import pandas as pd

from rdtb.backtest.engine import run_backtest
from rdtb.config import TradingBotConfig
from rdtb.models.train import score_panel, train_model_stack
from rdtb.portfolio.optimizer import PolicyParameters, default_policy
from rdtb.utils import ProgressCallback, annualized_return, max_drawdown, report_progress, sharpe_ratio

MIN_TRAIN_YEARS = 8
SEED_OFFSETS = (0, 7, 19)


@dataclass(slots=True)
class HoldoutValidationResult:
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    strategy_metrics: dict[str, object]
    benchmark_metrics: dict[str, object]
    equal_weight_metrics: dict[str, object]
    momentum_metrics: dict[str, object]
    stress_metrics: dict[str, dict[str, object]]


@dataclass(slots=True)
class SeedStabilityRun:
    seed: int
    metrics: dict[str, object]


@dataclass(slots=True)
class SeedStabilitySummary:
    runs: list[SeedStabilityRun]
    annualized_return_mean: float
    annualized_return_std: float
    annualized_return_min: float
    annualized_return_max: float
    sharpe_mean: float
    sharpe_std: float
    max_drawdown_worst: float


@dataclass(slots=True)
class ValidationMatrixReport:
    generated_at: str
    policy_source: str
    holdouts: list[HoldoutValidationResult]
    seed_stability: SeedStabilitySummary | None
    summary: dict[str, object]


def run_validation_matrix(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: TradingBotConfig,
    reference_policy: PolicyParameters | None = None,
    utility_weights: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
    policy_source: str = "default",
) -> ValidationMatrixReport:
    panel_frame = panel.copy()
    panel_frame["date"] = pd.to_datetime(panel_frame["date"])
    trainable = panel_frame.loc[panel_frame["is_trainable"]].copy()
    if trainable.empty:
        raise ValueError("No trainable rows are available for the validation matrix.")

    prices_frame = prices.copy()
    prices_frame["date"] = pd.to_datetime(prices_frame["date"])
    benchmarks_frame = benchmarks.copy()
    benchmarks_frame["date"] = pd.to_datetime(benchmarks_frame["date"])

    policy = reference_policy or default_policy(config)
    windows = build_repeated_holdout_windows(trainable, config)
    if not windows:
        raise ValueError("No repeated holdout windows could be generated for the validation matrix.")

    holdouts: list[HoldoutValidationResult] = []
    total_steps = max(len(windows) + 1, 1)
    for index, (test_start, test_end) in enumerate(windows, start=1):
        label = f"{test_start.year}-{test_end.year}"
        report_progress(
            progress_callback,
            f"Running validation holdout {label} ({index}/{len(windows)})...",
            0.05 + 0.75 * ((index - 1) / total_steps),
        )
        holdouts.append(
            _evaluate_holdout_window(
                panel=panel_frame,
                trainable=trainable,
                prices=prices_frame,
                benchmarks=benchmarks_frame,
                config=config,
                policy=policy,
                utility_weights=utility_weights,
                test_start=test_start,
                test_end=test_end,
            )
        )

    latest_window = windows[-1]
    report_progress(progress_callback, "Running seed-stability checks on the latest holdout...", 0.85)
    seed_stability = _evaluate_seed_stability(
        panel=panel_frame,
        trainable=trainable,
        config=config,
        policy=policy,
        utility_weights=utility_weights,
        test_start=latest_window[0],
        test_end=latest_window[1],
    )
    summary = summarize_validation_matrix(holdouts, seed_stability, config=config)
    report_progress(progress_callback, "Validation matrix completed.", 1.0)
    return ValidationMatrixReport(
        generated_at=pd.Timestamp.utcnow().isoformat(),
        policy_source=policy_source,
        holdouts=holdouts,
        seed_stability=seed_stability,
        summary=summary,
    )


def build_repeated_holdout_windows(trainable: pd.DataFrame, config: TradingBotConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    years = sorted(
        year
        for year in set(pd.to_datetime(trainable["date"]).dt.year.tolist())
        if year < int(config.paper_trade_year)
    )
    if not years:
        return []
    holdout_years = max(int(config.validation_holdout_years), 1)
    holdout_step_years = max(int(config.validation_holdout_step_years), 1)
    max_holdout_windows = max(int(config.validation_max_holdout_windows), 1)
    earliest_start = years[0] + max(MIN_TRAIN_YEARS, config.fold_min_train_years + holdout_years)
    latest_start = years[-1] - holdout_years + 1
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_year in range(earliest_start, latest_start + 1, holdout_step_years):
        end_year = start_year + holdout_years - 1
        if not all(year in years for year in range(start_year, end_year + 1)):
            continue
        windows.append((pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31")))
    return windows[-max_holdout_windows:]


def validation_matrix_to_dict(report: ValidationMatrixReport) -> dict[str, object]:
    return asdict(report)


def render_validation_matrix_markdown(report: ValidationMatrixReport) -> str:
    summary = report.summary
    lines = [
        "# Validation Matrix",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Policy source: `{report.policy_source}`",
        f"- Overall pass: `{summary.get('overall_pass', False)}`",
        "",
        "## Summary",
        "",
        f"- Median holdout annualized return: {_fmt_pct(summary.get('median_holdout_annualized_return'))}",
        f"- Median holdout Sharpe: {_fmt_float(summary.get('median_holdout_sharpe'))}",
        f"- Worst holdout max drawdown: {_fmt_pct(summary.get('worst_holdout_max_drawdown'))}",
        f"- Worst holdout year return: {_fmt_pct(summary.get('worst_holdout_year_return'))}",
        f"- Holdouts beating VNINDEX: `{summary.get('beat_benchmark_count', 0)}/{summary.get('holdout_count', 0)}`",
        f"- Holdouts beating equal-weight VN60: `{summary.get('beat_equal_weight_count', 0)}/{summary.get('holdout_count', 0)}`",
        f"- Holdouts beating momentum: `{summary.get('beat_momentum_count', 0)}/{summary.get('holdout_count', 0)}`",
        f"- Median momentum gap: {_fmt_pct(summary.get('median_momentum_gap'))}",
        f"- Median high-friction annualized return: {_fmt_pct(summary.get('median_high_friction_annualized_return'))}",
        f"- Median 1D-delay annualized return: {_fmt_pct(summary.get('median_signal_delay_annualized_return'))}",
        "",
        "## Holdouts",
        "",
        "| Window | Strategy Ann. | Strategy Sharpe | VNINDEX Ann. | Equal-Weight Ann. | Momentum Ann. | High Friction Ann. | 1D Delay Ann. |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for holdout in report.holdouts:
        lines.append(
            "| "
            f"{holdout.label} | "
            f"{_fmt_pct(holdout.strategy_metrics.get('annualized_return'))} | "
            f"{_fmt_float(holdout.strategy_metrics.get('sharpe'))} | "
            f"{_fmt_pct(holdout.benchmark_metrics.get('annualized_return'))} | "
            f"{_fmt_pct(holdout.equal_weight_metrics.get('annualized_return'))} | "
            f"{_fmt_pct(holdout.momentum_metrics.get('annualized_return'))} | "
            f"{_fmt_pct(holdout.stress_metrics.get('high_friction', {}).get('annualized_return'))} | "
            f"{_fmt_pct(holdout.stress_metrics.get('signal_delay_1d', {}).get('annualized_return'))} |"
        )
    if report.seed_stability is not None:
        lines.extend(
            [
                "",
                "## Seed Stability",
                "",
                f"- Mean annualized return: {_fmt_pct(report.seed_stability.annualized_return_mean)}",
                f"- Annualized return std dev: {_fmt_pct(report.seed_stability.annualized_return_std)}",
                f"- Mean Sharpe: {_fmt_float(report.seed_stability.sharpe_mean)}",
                f"- Sharpe std dev: {_fmt_float(report.seed_stability.sharpe_std)}",
                f"- Worst max drawdown: {_fmt_pct(report.seed_stability.max_drawdown_worst)}",
                "",
                "| Seed | Annualized Return | Sharpe | Max Drawdown |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for run in report.seed_stability.runs:
            lines.append(
                "| "
                f"{run.seed} | "
                f"{_fmt_pct(run.metrics.get('annualized_return'))} | "
                f"{_fmt_float(run.metrics.get('sharpe'))} | "
                f"{_fmt_pct(run.metrics.get('max_drawdown'))} |"
            )
    return "\n".join(lines) + "\n"


def summarize_validation_matrix(
    holdouts: list[HoldoutValidationResult],
    seed_stability: SeedStabilitySummary | None,
    config: TradingBotConfig | None = None,
) -> dict[str, object]:
    strategy_ann = [float(holdout.strategy_metrics["annualized_return"]) for holdout in holdouts]
    strategy_sharpe = [float(holdout.strategy_metrics["sharpe"]) for holdout in holdouts]
    strategy_drawdowns = [float(holdout.strategy_metrics["max_drawdown"]) for holdout in holdouts]
    holdout_worst_year_returns = [_extract_worst_year_return(holdout.strategy_metrics) for holdout in holdouts]
    high_friction_ann = [
        float(holdout.stress_metrics["high_friction"]["annualized_return"])
        for holdout in holdouts
        if "high_friction" in holdout.stress_metrics
    ]
    signal_delay_ann = [
        float(holdout.stress_metrics["signal_delay_1d"]["annualized_return"])
        for holdout in holdouts
        if "signal_delay_1d" in holdout.stress_metrics
    ]
    beat_benchmark_count = sum(
        float(holdout.strategy_metrics["annualized_return"]) > float(holdout.benchmark_metrics["annualized_return"])
        for holdout in holdouts
    )
    beat_equal_weight_count = sum(
        float(holdout.strategy_metrics["annualized_return"]) > float(holdout.equal_weight_metrics["annualized_return"])
        for holdout in holdouts
    )
    beat_momentum_count = sum(
        float(holdout.strategy_metrics["annualized_return"]) > float(holdout.momentum_metrics["annualized_return"])
        for holdout in holdouts
    )
    momentum_gaps = [
        float(holdout.strategy_metrics["annualized_return"]) - float(holdout.momentum_metrics["annualized_return"])
        for holdout in holdouts
    ]
    holdout_count = len(holdouts)
    min_year_return_floor = float(config.deployment_min_year_return) if config is not None else 0.20
    drawdown_floor = float(config.deployment_max_drawdown) if config is not None else -0.15
    pass_criteria = {
        "median_holdout_annualized_return_gt_min_year_return": (
            statistics.median(strategy_ann) > min_year_return_floor if strategy_ann else False
        ),
        "median_holdout_sharpe_gt_1": statistics.median(strategy_sharpe) > 1.0 if strategy_sharpe else False,
        "median_holdout_worst_year_return_gt_min_year_return": (
            statistics.median(holdout_worst_year_returns) > min_year_return_floor if holdout_worst_year_returns else False
        ),
        "worst_holdout_drawdown_gt_drawdown_floor": min(strategy_drawdowns) > drawdown_floor if strategy_drawdowns else False,
        "beats_vnindex_in_majority": beat_benchmark_count >= ((holdout_count // 2) + 1) if holdout_count else False,
        "beats_equal_weight_in_majority": beat_equal_weight_count >= ((holdout_count // 2) + 1) if holdout_count else False,
        "median_high_friction_annualized_return_positive": statistics.median(high_friction_ann) > 0.0 if high_friction_ann else False,
        "median_signal_delay_annualized_return_positive": statistics.median(signal_delay_ann) > 0.0 if signal_delay_ann else False,
        "seed_stability_annualized_return_std_lt_10pct": (
            seed_stability.annualized_return_std < 0.10 if seed_stability is not None else False
        ),
    }
    overall_pass = all(pass_criteria.values()) if pass_criteria else False
    return {
        "holdout_count": holdout_count,
        "median_holdout_annualized_return": statistics.median(strategy_ann) if strategy_ann else 0.0,
        "median_holdout_sharpe": statistics.median(strategy_sharpe) if strategy_sharpe else 0.0,
        "worst_holdout_max_drawdown": min(strategy_drawdowns) if strategy_drawdowns else 0.0,
        "median_holdout_worst_year_return": statistics.median(holdout_worst_year_returns) if holdout_worst_year_returns else 0.0,
        "worst_holdout_year_return": min(holdout_worst_year_returns) if holdout_worst_year_returns else 0.0,
        "beat_benchmark_count": beat_benchmark_count,
        "beat_equal_weight_count": beat_equal_weight_count,
        "beat_momentum_count": beat_momentum_count,
        "median_momentum_gap": statistics.median(momentum_gaps) if momentum_gaps else 0.0,
        "median_high_friction_annualized_return": statistics.median(high_friction_ann) if high_friction_ann else 0.0,
        "median_signal_delay_annualized_return": statistics.median(signal_delay_ann) if signal_delay_ann else 0.0,
        "pass_criteria": pass_criteria,
        "overall_pass": overall_pass,
    }


def delay_scored_panel(scored_panel: pd.DataFrame, days: int = 1) -> pd.DataFrame:
    delayed = scored_panel.copy()
    delayed["date"] = pd.to_datetime(delayed["date"])
    delayed = delayed.sort_values(["symbol", "date"]).reset_index(drop=True)
    for column in [
        "alpha_prediction",
        "risk_probability",
        "utility_score",
        "regime_probability",
        "regime_anchor_probability",
        "regime_participation_probability",
    ]:
        if column in delayed.columns:
            delayed[column] = delayed.groupby("symbol")[column].shift(days)
    return delayed.sort_values(["date", "utility_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def _evaluate_holdout_window(
    panel: pd.DataFrame,
    trainable: pd.DataFrame,
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: TradingBotConfig,
    policy: PolicyParameters,
    utility_weights: dict[str, float] | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    include_stress: bool = True,
) -> HoldoutValidationResult:
    train_cutoff = test_start - pd.Timedelta(days=config.fold_embargo_days)
    pre_holdout = trainable.loc[pd.to_datetime(trainable["date"]) <= train_cutoff].copy()
    pre_holdout = _restrict_recent_training_window(pre_holdout, anchor_date=test_start, config=config)
    if pre_holdout.empty:
        raise ValueError(f"No pre-holdout training rows are available before {test_start.date()}.")

    holdout_panel = panel.loc[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    if holdout_panel.empty:
        raise ValueError(f"No holdout rows are available for {test_start.date()} to {test_end.date()}.")

    holdout_benchmarks = benchmarks.loc[(benchmarks["date"] >= test_start) & (benchmarks["date"] <= test_end)].copy()
    model_stack = train_model_stack(pre_holdout, config, persist=False).model_stack
    scored_holdout = score_panel(holdout_panel, model_stack, utility_weights=utility_weights)
    strategy_metrics = run_backtest(scored_holdout, config, policy=policy).metrics
    stress_metrics = _run_stress_scenarios(scored_holdout, config, policy) if include_stress else {}
    benchmark_metrics = _run_benchmark_buy_and_hold(holdout_benchmarks, config)
    equal_weight_metrics = _run_monthly_rebalance_baseline(holdout_panel, config, signal_column=None, top_n=None)
    momentum_metrics = _run_monthly_rebalance_baseline(
        holdout_panel,
        config,
        signal_column="relative_strength_20d",
        top_n=max(int(policy.max_positions), 1),
    )
    return HoldoutValidationResult(
        label=f"{test_start.year}-{test_end.year}",
        train_start=pd.to_datetime(pre_holdout["date"]).min().strftime("%Y-%m-%d"),
        train_end=pd.to_datetime(pre_holdout["date"]).max().strftime("%Y-%m-%d"),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=test_end.strftime("%Y-%m-%d"),
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        equal_weight_metrics=equal_weight_metrics,
        momentum_metrics=momentum_metrics,
        stress_metrics=stress_metrics,
    )


def _evaluate_seed_stability(
    panel: pd.DataFrame,
    trainable: pd.DataFrame,
    config: TradingBotConfig,
    policy: PolicyParameters,
    utility_weights: dict[str, float] | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> SeedStabilitySummary:
    train_cutoff = test_start - pd.Timedelta(days=config.fold_embargo_days)
    pre_holdout = trainable.loc[pd.to_datetime(trainable["date"]) <= train_cutoff].copy()
    pre_holdout = _restrict_recent_training_window(pre_holdout, anchor_date=test_start, config=config)
    holdout_panel = panel.loc[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()

    runs: list[SeedStabilityRun] = []
    for offset in SEED_OFFSETS:
        seed = int(config.random_state + offset)
        seed_config = replace(config, random_state=seed)
        model_stack = train_model_stack(pre_holdout, seed_config, persist=False).model_stack
        scored_holdout = score_panel(holdout_panel, model_stack, utility_weights=utility_weights)
        metrics = run_backtest(scored_holdout, seed_config, policy=policy).metrics
        runs.append(SeedStabilityRun(seed=seed, metrics=metrics))

    ann_returns = [float(run.metrics["annualized_return"]) for run in runs]
    sharpe_values = [float(run.metrics["sharpe"]) for run in runs]
    drawdowns = [float(run.metrics["max_drawdown"]) for run in runs]
    return SeedStabilitySummary(
        runs=runs,
        annualized_return_mean=float(statistics.fmean(ann_returns)),
        annualized_return_std=float(statistics.pstdev(ann_returns)) if len(ann_returns) > 1 else 0.0,
        annualized_return_min=float(min(ann_returns)),
        annualized_return_max=float(max(ann_returns)),
        sharpe_mean=float(statistics.fmean(sharpe_values)),
        sharpe_std=float(statistics.pstdev(sharpe_values)) if len(sharpe_values) > 1 else 0.0,
        max_drawdown_worst=float(min(drawdowns)),
    )


def _run_stress_scenarios(
    scored_holdout: pd.DataFrame,
    config: TradingBotConfig,
    policy: PolicyParameters,
) -> dict[str, dict[str, object]]:
    high_friction_config = replace(
        config,
        buy_transaction_fee_bps=float(config.buy_transaction_fee_bps * 3.0),
        sell_transaction_fee_bps=float(config.sell_transaction_fee_bps * 3.0),
        commission_bps=float(config.commission_bps * 2.0),
        slippage_bps=float(config.slippage_bps * 2.0),
    )
    delayed_panel = delay_scored_panel(scored_holdout, days=1)
    return {
        "high_friction": run_backtest(scored_holdout, high_friction_config, policy=policy).metrics,
        "signal_delay_1d": run_backtest(delayed_panel, config, policy=policy).metrics,
        "high_friction_and_delay": run_backtest(delayed_panel, high_friction_config, policy=policy).metrics,
    }


def _run_benchmark_buy_and_hold(benchmarks: pd.DataFrame, config: TradingBotConfig) -> dict[str, object]:
    benchmark_symbol = config.benchmark_symbols[0]
    frame = benchmarks.loc[benchmarks["symbol"] == benchmark_symbol].copy().sort_values("date").reset_index(drop=True)
    if len(frame) < 2:
        return _empty_metrics()
    dates = pd.to_datetime(frame["date"]).tolist()
    cash = float(config.initial_cash)
    shares = 0.0
    fee_rate = (config.buy_transaction_fee_bps + config.commission_bps) / 10_000.0
    slippage_rate = config.slippage_bps / 10_000.0
    equity_log = [
        {
            "date": dates[0],
            "equity": cash,
            "cash": cash,
            "gross_exposure": 0.0,
            "position_count": 0,
        }
    ]
    entry_price = float(frame.loc[1, "open"]) * (1.0 + slippage_rate)
    if entry_price > 0:
        shares = cash / (entry_price * (1.0 + fee_rate))
        cash -= shares * entry_price * (1.0 + fee_rate)
    for index in range(1, len(frame)):
        close_price = float(frame.loc[index, "close"])
        equity = cash + shares * close_price
        equity_log.append(
            {
                "date": dates[index],
                "equity": equity,
                "cash": cash,
                "gross_exposure": 0.0 if equity <= 0 else (shares * close_price) / equity,
                "position_count": 1 if shares > 0 else 0,
            }
        )
    return _compute_research_metrics(pd.DataFrame(equity_log), trade_count=1 if shares > 0 else 0)


def _run_monthly_rebalance_baseline(
    panel: pd.DataFrame,
    config: TradingBotConfig,
    signal_column: str | None,
    top_n: int | None,
) -> dict[str, object]:
    frame = panel.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        return _empty_metrics()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = sorted(frame["date"].drop_duplicates().tolist())
    if len(dates) < 2:
        return _empty_metrics()

    decision_dates = _build_rebalance_decision_dates(dates)
    open_prices = frame.pivot(index="date", columns="symbol", values="open").sort_index()
    close_prices = frame.pivot(index="date", columns="symbol", values="close").sort_index()
    holdings = pd.Series(dtype=float)
    cash = float(config.initial_cash)
    fee_rate = (
        config.buy_transaction_fee_bps
        + config.sell_transaction_fee_bps
        + 2.0 * config.commission_bps
        + 2.0 * config.slippage_bps
    ) / 20_000.0
    trade_count = 0
    equity_log = [
        {
            "date": dates[0],
            "equity": cash,
            "cash": cash,
            "gross_exposure": 0.0,
            "position_count": 0,
        }
    ]

    for current_date, next_date in zip(dates[:-1], dates[1:]):
        current_close = close_prices.loc[current_date].reindex(holdings.index, fill_value=0.0) if not holdings.empty else pd.Series(dtype=float)
        equity = cash + float((holdings * current_close).sum()) if not holdings.empty else cash
        if current_date in decision_dates:
            target_weights = _baseline_target_weights(
                frame=frame,
                current_date=current_date,
                signal_column=signal_column,
                top_n=top_n,
            )
            execution_open = open_prices.loc[next_date]
            target_shares = _target_shares_from_weights(
                equity=equity,
                weights=target_weights,
                execution_open=execution_open,
            )
            price_reference = execution_open.reindex(target_shares.index.union(holdings.index), fill_value=0.0)
            current_shares = holdings.reindex(price_reference.index, fill_value=0.0)
            aligned_target = target_shares.reindex(price_reference.index, fill_value=0.0)
            turnover_notional = float(((aligned_target - current_shares).abs() * price_reference).sum())
            fees = turnover_notional * fee_rate
            invested = float((target_shares * execution_open.reindex(target_shares.index)).sum())
            cash = equity - invested - fees
            holdings = target_shares
            if turnover_notional > 0:
                trade_count += 1
        next_close = close_prices.loc[next_date].reindex(holdings.index, fill_value=0.0) if not holdings.empty else pd.Series(dtype=float)
        position_value = float((holdings * next_close).sum()) if not holdings.empty else 0.0
        equity = cash + position_value
        equity_log.append(
            {
                "date": next_date,
                "equity": equity,
                "cash": cash,
                "gross_exposure": 0.0 if equity <= 0 else position_value / equity,
                "position_count": int((holdings > 0).sum()),
            }
        )
    return _compute_research_metrics(pd.DataFrame(equity_log), trade_count=trade_count)


def _baseline_target_weights(
    frame: pd.DataFrame,
    current_date: pd.Timestamp,
    signal_column: str | None,
    top_n: int | None,
) -> dict[str, float]:
    snapshot = frame.loc[frame["date"] == current_date, ["symbol", "close"] + ([signal_column] if signal_column else [])].copy()
    snapshot = snapshot.loc[snapshot["close"].notna() & (snapshot["close"] > 0)].copy()
    if snapshot.empty:
        return {}
    if signal_column is not None:
        snapshot = snapshot.loc[snapshot[signal_column].notna()].sort_values(signal_column, ascending=False)
    else:
        snapshot = snapshot.sort_values("symbol")
    if top_n is not None and top_n > 0:
        snapshot = snapshot.head(top_n)
    if snapshot.empty:
        return {}
    weight = 1.0 / len(snapshot)
    return {str(symbol): weight for symbol in snapshot["symbol"]}


def _target_shares_from_weights(
    equity: float,
    weights: dict[str, float],
    execution_open: pd.Series,
) -> pd.Series:
    shares: dict[str, float] = {}
    for symbol, weight in weights.items():
        price = float(execution_open.get(symbol, 0.0))
        if price <= 0:
            continue
        shares[symbol] = (equity * float(weight)) / price
    return pd.Series(shares, dtype=float)


def _build_rebalance_decision_dates(dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    frame["year_month"] = frame["date"].dt.to_period("M")
    decision_dates = set(frame.groupby("year_month")["date"].max().tolist())
    decision_dates.add(pd.Timestamp(dates[0]))
    return {pd.Timestamp(value) for value in decision_dates}


def _extract_worst_year_return(metrics: dict[str, object]) -> float:
    yearly_returns = metrics.get("yearly_returns", {})
    if isinstance(yearly_returns, dict) and yearly_returns:
        return float(min(float(value) for value in yearly_returns.values()))
    return float(metrics.get("annualized_return", 0.0))


def _compute_research_metrics(equity_curve: pd.DataFrame, trade_count: int) -> dict[str, object]:
    if equity_curve.empty:
        return _empty_metrics()
    curve = equity_curve.copy().sort_values("date").reset_index(drop=True)
    curve["return_1d"] = curve["equity"].pct_change().fillna(0.0)
    yearly_returns: dict[str, float] = {}
    for year, part in curve.groupby(curve["date"].dt.year):
        if len(part) < 2:
            continue
        yearly_returns[str(year)] = float(part["equity"].iloc[-1] / part["equity"].iloc[0] - 1.0)
    return {
        "total_return": float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0),
        "annualized_return": float(annualized_return(curve["equity"])),
        "max_drawdown": float(max_drawdown(curve["equity"])),
        "sharpe": float(sharpe_ratio(curve["return_1d"])),
        "trade_count": int(trade_count),
        "avg_gross_exposure": float(curve["gross_exposure"].mean()),
        "avg_position_count": float(curve["position_count"].mean()),
        "yearly_returns": yearly_returns,
    }


def _restrict_recent_training_window(panel: pd.DataFrame, anchor_date: pd.Timestamp, config: TradingBotConfig) -> pd.DataFrame:
    if panel.empty:
        return panel
    window_start = anchor_date - pd.DateOffset(years=config.max_train_years)
    trimmed = panel.loc[pd.to_datetime(panel["date"]) >= window_start].copy()
    return trimmed if not trimmed.empty else panel


def _empty_metrics() -> dict[str, object]:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "trade_count": 0,
        "avg_gross_exposure": 0.0,
        "avg_position_count": 0.0,
        "yearly_returns": {},
    }


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"
