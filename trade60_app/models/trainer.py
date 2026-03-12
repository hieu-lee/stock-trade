from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from trade60_app.backtest.engine import BacktestArtifacts, StrategyParameters, run_daily_backtest
from trade60_app.config import Trade60Config

ALPHA_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_40d",
    "gap_open",
    "intraday_return",
    "range_pct",
    "volatility_10d",
    "volatility_20d",
    "avg_volume_20d",
    "avg_turnover_20d",
    "volume_zscore_20d",
    "distance_ma10",
    "distance_ma20",
    "distance_ma50",
    "distance_ma200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "atr_pct",
    "adx",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "benchmark_ret_20d",
    "benchmark_vol_20d",
    "benchmark_drawdown_252d",
    "benchmark_distance_ma200",
    "relative_strength_10d",
    "relative_strength_20d",
    "beta_60d",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "breadth_positive_10d",
    "breadth_ret_20d",
    "breadth_turnover_20d",
]

REGIME_FEATURE_COLUMNS = [
    "benchmark_ret_1d",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "benchmark_ret_20d",
    "benchmark_vol_20d",
    "benchmark_drawdown_252d",
    "benchmark_distance_ma200",
    "benchmark_gap_open",
    "benchmark_intraday_return",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "breadth_positive_10d",
    "breadth_ret_20d",
    "breadth_turnover_20d",
]


@dataclass(slots=True)
class ModelBundle:
    model: Pipeline
    feature_columns: list[str]


@dataclass(slots=True)
class TrainingArtifacts:
    alpha_bundle: ModelBundle
    regime_bundle: ModelBundle
    best_params: dict
    split_summary: dict
    validation_backtest: BacktestArtifacts
    holdout_backtest: BacktestArtifacts
    validation_scored_panel: pd.DataFrame
    holdout_scored_panel: pd.DataFrame
    feature_importance: pd.DataFrame


@dataclass(slots=True)
class CleanWalkforwardArtifacts:
    production_alpha_bundle: ModelBundle
    production_regime_bundle: ModelBundle
    selected_params: dict
    split_summary: dict
    fold_summaries: list[dict]
    final_backtest: BacktestArtifacts
    final_scored_panel: pd.DataFrame


def train_models(panel: pd.DataFrame, config: Trade60Config) -> TrainingArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy()
    if trainable.empty:
        raise ValueError("No trainable rows were produced for the trade60 panel.")

    split_summary = _resolve_split(trainable["date"], config)
    train_mask = trainable["date"] < split_summary["validation_start"]
    validation_mask = (trainable["date"] >= split_summary["validation_start"]) & (trainable["date"] < split_summary["holdout_start"])
    holdout_mask = trainable["date"] >= split_summary["holdout_start"]

    train_panel = trainable.loc[train_mask].copy()
    validation_panel = trainable.loc[validation_mask].copy()
    development_panel = trainable.loc[trainable["date"] < split_summary["holdout_start"]].copy()
    holdout_panel = trainable.loc[holdout_mask].copy()

    if train_panel.empty or validation_panel.empty or holdout_panel.empty:
        raise ValueError("Date split produced an empty train, validation, or holdout partition.")

    alpha_bundle = _fit_alpha_model(train_panel, config)
    regime_bundle = _fit_regime_model(train_panel, config)
    validation_scored_panel = score_panel(validation_panel, alpha_bundle, regime_bundle)
    best_params, validation_backtest = _search_best_parameters(validation_scored_panel, config)

    final_alpha_bundle = _fit_alpha_model(development_panel, config)
    final_regime_bundle = _fit_regime_model(development_panel, config)
    holdout_scored_panel = score_panel(holdout_panel, final_alpha_bundle, final_regime_bundle)
    holdout_backtest = run_daily_backtest(holdout_scored_panel, config, _params_from_dict(best_params))
    feature_importance = _extract_feature_importance(final_alpha_bundle)

    return TrainingArtifacts(
        alpha_bundle=final_alpha_bundle,
        regime_bundle=final_regime_bundle,
        best_params=best_params,
        split_summary={
            **split_summary,
            "train_rows": int(len(train_panel)),
            "validation_rows": int(len(validation_panel)),
            "holdout_rows": int(len(holdout_panel)),
        },
        validation_backtest=validation_backtest,
        holdout_backtest=holdout_backtest,
        validation_scored_panel=validation_scored_panel,
        holdout_scored_panel=holdout_scored_panel,
        feature_importance=feature_importance,
    )


def run_clean_walkforward_evaluation(panel: pd.DataFrame, config: Trade60Config) -> CleanWalkforwardArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy().sort_values("date")
    if trainable.empty:
        raise ValueError("No trainable rows were produced for clean walk-forward evaluation.")

    split_summary = _resolve_clean_walkforward_split(trainable["date"])
    development_panel = trainable.loc[trainable["date"] < split_summary["final_test_start"]].copy()
    final_test_panel = trainable.loc[trainable["date"] >= split_summary["final_test_start"]].copy()

    fold_frames: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    development_dates = sorted(pd.to_datetime(development_panel["date"]).unique())
    for valid_start_idx in range(
        split_summary["min_train_days"],
        len(development_dates) - split_summary["validation_days"] + 1,
        split_summary["step_days"],
    ):
        valid_start = pd.Timestamp(development_dates[valid_start_idx])
        valid_end = pd.Timestamp(development_dates[min(valid_start_idx + split_summary["validation_days"], len(development_dates) - 1)])
        fold_train = development_panel.loc[development_panel["date"] < valid_start].copy()
        fold_valid = development_panel.loc[
            (development_panel["date"] >= valid_start) & (development_panel["date"] < valid_end)
        ].copy()
        if fold_train.empty or fold_valid.empty:
            continue
        alpha_bundle, regime_bundle = fit_model_bundles(fold_train, config)
        fold_scored = score_panel(fold_valid, alpha_bundle, regime_bundle)
        fold_frames.append((valid_start, valid_end, fold_scored))

    if not fold_frames:
        raise ValueError("Clean walk-forward evaluation could not create any development folds.")

    selected_params, fold_summaries = _search_parameters_across_folds(fold_frames, config)
    evaluation_alpha_bundle, evaluation_regime_bundle = fit_model_bundles(development_panel, config)
    final_scored_panel = score_panel(final_test_panel, evaluation_alpha_bundle, evaluation_regime_bundle)
    final_backtest = run_daily_backtest(final_scored_panel, config, _params_from_dict(selected_params))
    production_alpha_bundle, production_regime_bundle = fit_model_bundles(trainable, config)

    return CleanWalkforwardArtifacts(
        production_alpha_bundle=production_alpha_bundle,
        production_regime_bundle=production_regime_bundle,
        selected_params=selected_params,
        split_summary={
            **split_summary,
            "development_rows": int(len(development_panel)),
            "final_test_rows": int(len(final_test_panel)),
            "fold_count": int(len(fold_frames)),
        },
        fold_summaries=fold_summaries,
        final_backtest=final_backtest,
        final_scored_panel=final_scored_panel,
    )


def score_panel(panel: pd.DataFrame, alpha_bundle: ModelBundle, regime_bundle: ModelBundle) -> pd.DataFrame:
    scored = panel.copy()
    alpha_features = scored.reindex(columns=alpha_bundle.feature_columns).replace([np.inf, -np.inf], np.nan)
    alpha_probabilities = _positive_class_probability(alpha_bundle.model, alpha_features)
    scored["alpha_probability"] = alpha_probabilities

    regime_daily = scored[["date"] + regime_bundle.feature_columns].drop_duplicates(subset=["date"]).sort_values("date")
    regime_features = regime_daily[regime_bundle.feature_columns].replace([np.inf, -np.inf], np.nan)
    regime_probabilities = _positive_class_probability(regime_bundle.model, regime_features)
    regime_frame = regime_daily[["date"]].assign(regime_probability=regime_probabilities)
    scored = scored.merge(regime_frame, on="date", how="left")

    scored["composite_score"] = np.clip(
        scored["alpha_probability"] * 0.75
        + scored["regime_probability"] * 0.25
        + scored["relative_strength_20d"].fillna(0.0) * 0.10,
        0.0,
        1.0,
    )
    return scored.sort_values(["date", "composite_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def load_model_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)


def fit_model_bundles(panel: pd.DataFrame, config: Trade60Config) -> tuple[ModelBundle, ModelBundle]:
    train_frame = panel.loc[panel["is_trainable"]] if "is_trainable" in panel.columns else panel
    train_frame = train_frame.copy()
    return _fit_alpha_model(train_frame, config), _fit_regime_model(train_frame, config)


def calibrate_deployment_parameters(
    scored_panel: pd.DataFrame,
    base_params: dict,
    config: Trade60Config,
) -> tuple[dict, BacktestArtifacts]:
    best_score = float("-inf")
    best_params = base_params.copy()
    best_backtest = run_daily_backtest(scored_panel, config, _params_from_dict(base_params))

    regime_candidates = sorted(
        {
            round(max(0.40, float(base_params["regime_threshold"]) - 0.08), 2),
            round(max(0.42, float(base_params["regime_threshold"]) - 0.06), 2),
            round(max(0.44, float(base_params["regime_threshold"]) - 0.04), 2),
            round(float(base_params["regime_threshold"]), 2),
        }
    )
    max_position_candidates = sorted({int(base_params["max_positions"]), max(int(base_params["max_positions"]), 7)})

    for regime_threshold, max_positions in product(regime_candidates, max_position_candidates):
        candidate_params = {
            **base_params,
            "regime_threshold": regime_threshold,
            "max_positions": max_positions,
        }
        backtest = run_daily_backtest(scored_panel, config, _params_from_dict(candidate_params))
        score = _objective(backtest.metrics, config)
        if score > best_score:
            best_score = score
            best_params = {**candidate_params, "deployment_objective_score": score}
            best_backtest = backtest

    return best_params, best_backtest


def _search_parameters_across_folds(
    fold_frames: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    config: Trade60Config,
) -> tuple[dict, list[dict]]:
    best_score = float("-inf")
    best_params: dict | None = None
    best_fold_summaries: list[dict] = []
    rebalance_profiles = _rebalance_profiles(config)

    parameter_grid = product(
        [0.50],
        [0.96],
        [0.42],
        [0.40],
        [5],
        [2],
        [0.07],
        [0.24],
    )

    for entry_threshold, entry_quantile, exit_threshold, regime_threshold, max_positions, min_holding_days, stop_loss_pct, take_profit_pct in parameter_grid:
        for rebalance_profile in rebalance_profiles:
            fold_scores: list[float] = []
            fold_summaries: list[dict] = []
            for fold_index, (valid_start, valid_end, scored_panel) in enumerate(fold_frames, start=1):
                params = StrategyParameters(
                    entry_threshold=entry_threshold,
                    entry_quantile=entry_quantile,
                    exit_threshold=exit_threshold,
                    regime_threshold=regime_threshold,
                    max_positions=max_positions,
                    max_holding_days=config.max_holding_days,
                    min_holding_days=min_holding_days,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    hold_alpha_buffer=rebalance_profile["hold_alpha_buffer"],
                    rank_keep_fraction=rebalance_profile["rank_keep_fraction"],
                    defensive_trim_fraction=rebalance_profile["defensive_trim_fraction"],
                    weak_alpha_trim_fraction=rebalance_profile["weak_alpha_trim_fraction"],
                    profit_trim_fraction=rebalance_profile["profit_trim_fraction"],
                )
                backtest = run_daily_backtest(scored_panel, config, params)
                objective = _objective(backtest.metrics, config)
                fold_scores.append(objective)
                fold_summaries.append(
                    {
                        "fold_index": fold_index,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "objective_score": objective,
                        **backtest.metrics,
                    }
                )

            weights = np.arange(1, len(fold_scores) + 1)
            weighted_score = float(np.average(fold_scores, weights=weights))
            if weighted_score > best_score:
                best_score = weighted_score
                best_params = {
                    "entry_threshold": entry_threshold,
                    "entry_quantile": entry_quantile,
                    "exit_threshold": exit_threshold,
                    "regime_threshold": regime_threshold,
                    "max_positions": max_positions,
                    "max_holding_days": config.max_holding_days,
                    "min_holding_days": min_holding_days,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    **rebalance_profile,
                    "clean_walkforward_objective_score": weighted_score,
                }
                best_fold_summaries = fold_summaries

    if best_params is None:
        raise RuntimeError("Clean walk-forward evaluation could not select any strategy parameters.")
    return best_params, best_fold_summaries


def _fit_alpha_model(panel: pd.DataFrame, config: Trade60Config) -> ModelBundle:
    train_frame = panel.dropna(subset=["target_long"]).copy()
    train_frame[ALPHA_FEATURE_COLUMNS] = train_frame[ALPHA_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    classifier = _build_classifier(
        target=train_frame["target_long"].astype(int),
        random_state=config.random_state,
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=8,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    model.fit(train_frame[ALPHA_FEATURE_COLUMNS], train_frame["target_long"].astype(int))
    return ModelBundle(model=model, feature_columns=list(ALPHA_FEATURE_COLUMNS))


def _fit_regime_model(panel: pd.DataFrame, config: Trade60Config) -> ModelBundle:
    regime_frame = panel[["date", "target_regime"] + REGIME_FEATURE_COLUMNS].drop_duplicates(subset=["date"]).dropna(subset=["target_regime"]).copy()
    regime_frame[REGIME_FEATURE_COLUMNS] = regime_frame[REGIME_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    classifier = _build_classifier(
        target=regime_frame["target_regime"].astype(int),
        random_state=config.random_state,
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=6,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    model.fit(regime_frame[REGIME_FEATURE_COLUMNS], regime_frame["target_regime"].astype(int))
    return ModelBundle(model=model, feature_columns=list(REGIME_FEATURE_COLUMNS))


def _search_best_parameters(panel: pd.DataFrame, config: Trade60Config) -> tuple[dict, BacktestArtifacts]:
    best_score = float("-inf")
    best_params: dict | None = None
    best_backtest: BacktestArtifacts | None = None
    rebalance_profiles = _rebalance_profiles(config)

    parameter_grid = product(
        [0.50],
        [0.96],
        [0.42],
        [0.40],
        [5],
        [2],
        [0.07],
        [0.24],
    )

    for entry_threshold, entry_quantile, exit_threshold, regime_threshold, max_positions, min_holding_days, stop_loss_pct, take_profit_pct in parameter_grid:
        for rebalance_profile in rebalance_profiles:
            params = StrategyParameters(
                entry_threshold=entry_threshold,
                entry_quantile=entry_quantile,
                exit_threshold=exit_threshold,
                regime_threshold=regime_threshold,
                max_positions=max_positions,
                max_holding_days=config.max_holding_days,
                min_holding_days=min_holding_days,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                hold_alpha_buffer=rebalance_profile["hold_alpha_buffer"],
                rank_keep_fraction=rebalance_profile["rank_keep_fraction"],
                defensive_trim_fraction=rebalance_profile["defensive_trim_fraction"],
                weak_alpha_trim_fraction=rebalance_profile["weak_alpha_trim_fraction"],
                profit_trim_fraction=rebalance_profile["profit_trim_fraction"],
            )
            backtest = run_daily_backtest(panel, config, params)
            score = _objective(backtest.metrics, config)
            if score > best_score:
                best_score = score
                best_params = {
                    "entry_threshold": entry_threshold,
                    "entry_quantile": entry_quantile,
                    "exit_threshold": exit_threshold,
                    "regime_threshold": regime_threshold,
                    "max_positions": max_positions,
                    "max_holding_days": config.max_holding_days,
                    "min_holding_days": min_holding_days,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    **rebalance_profile,
                    "objective_score": score,
                }
                best_backtest = backtest

    if best_params is None or best_backtest is None:
        raise RuntimeError("Could not determine the best trade60 strategy parameters.")
    return best_params, best_backtest


def _objective(metrics: dict, config: Trade60Config) -> float:
    annualized = float(metrics["annualized_return"])
    excess = float(metrics["excess_return_vs_benchmark"])
    max_drawdown = float(metrics["max_drawdown"])
    avg_cash_ratio = float(metrics.get("avg_cash_ratio", 0.0))
    positive_benchmark_flat_ratio = float(metrics.get("positive_benchmark_flat_ratio", 0.0))
    benchmark_return = float(metrics.get("benchmark_return", 0.0))
    drawdown_penalty = abs(max_drawdown) * 2.5
    trade_penalty = 0.0 if int(metrics["trade_count"]) >= 15 else 0.5
    goal_penalty = 0.0
    if annualized < config.target_annual_return:
        goal_penalty += (config.target_annual_return - annualized) * 6.0
    if excess < 0:
        goal_penalty += abs(excess) * 6.0
    if max_drawdown < -0.10:
        goal_penalty += abs(max_drawdown + 0.10) * 28.0
    if benchmark_return > 0.10:
        goal_penalty += positive_benchmark_flat_ratio * 6.0
        goal_penalty += max(avg_cash_ratio - 0.55, 0.0) * 1.0
    return (
        annualized * 5.5
        + excess * 4.0
        + float(metrics["win_rate"]) * 1.25
        + float(metrics["sharpe"]) * 0.25
        - drawdown_penalty
        - trade_penalty
        - goal_penalty
    )


def _resolve_split(dates: pd.Series, config: Trade60Config) -> dict:
    unique_dates = sorted(pd.to_datetime(dates).unique())
    total = len(unique_dates)
    if total < 600:
        raise ValueError("Not enough daily history to create robust train/validation/holdout splits.")

    holdout_days = min(config.holdout_days, max(126, total // 5))
    validation_days = min(config.validation_days, max(126, total // 5))
    while total - holdout_days - validation_days < 252:
        if validation_days > 126:
            validation_days -= 21
        elif holdout_days > 126:
            holdout_days -= 21
        else:
            break

    validation_start_idx = total - holdout_days - validation_days
    holdout_start_idx = total - holdout_days
    return {
        "train_start": pd.Timestamp(unique_dates[0]),
        "validation_start": pd.Timestamp(unique_dates[validation_start_idx]),
        "holdout_start": pd.Timestamp(unique_dates[holdout_start_idx]),
        "end_date": pd.Timestamp(unique_dates[-1]),
        "validation_days": validation_days,
        "holdout_days": holdout_days,
    }


def _resolve_clean_walkforward_split(dates: pd.Series) -> dict:
    unique_dates = sorted(pd.to_datetime(dates).unique())
    final_test_days = 252
    validation_days = 252
    step_days = 252
    min_train_days = 1512
    if len(unique_dates) <= final_test_days + min_train_days + validation_days:
        raise ValueError("Not enough history for clean walk-forward evaluation with an untouched final year.")
    final_test_start = pd.Timestamp(unique_dates[-final_test_days])
    return {
        "train_start": pd.Timestamp(unique_dates[0]),
        "final_test_start": final_test_start,
        "final_test_end": pd.Timestamp(unique_dates[-1]),
        "final_test_days": final_test_days,
        "validation_days": validation_days,
        "step_days": step_days,
        "min_train_days": min_train_days,
    }


def _extract_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    classifier = bundle.model.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": bundle.feature_columns, "importance": importances}).sort_values(
        "importance",
        ascending=False,
    )


def _build_classifier(
    target: pd.Series,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
):
    if target.nunique() < 2:
        return DummyClassifier(strategy="constant", constant=int(target.iloc[0]))
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def _positive_class_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(features)
    classes = getattr(classifier, "classes_", np.array([0, 1]))
    if probabilities.shape[1] == 1:
        return np.full(len(features), float(classes[0]))
    positive_index = int(np.where(classes == 1)[0][0]) if 1 in classes else probabilities.shape[1] - 1
    return probabilities[:, positive_index]


def _params_from_dict(payload: dict) -> StrategyParameters:
    return StrategyParameters(
        entry_threshold=float(payload["entry_threshold"]),
        entry_quantile=float(payload["entry_quantile"]),
        exit_threshold=float(payload["exit_threshold"]),
        regime_threshold=float(payload["regime_threshold"]),
        max_positions=int(payload["max_positions"]),
        max_holding_days=int(payload["max_holding_days"]),
        min_holding_days=int(payload["min_holding_days"]),
        stop_loss_pct=float(payload["stop_loss_pct"]),
        take_profit_pct=float(payload["take_profit_pct"]),
        hold_alpha_buffer=float(payload.get("hold_alpha_buffer", 0.06)),
        rank_keep_fraction=float(payload.get("rank_keep_fraction", 1.0)),
        defensive_trim_fraction=float(payload.get("defensive_trim_fraction", 0.35)),
        weak_alpha_trim_fraction=float(payload.get("weak_alpha_trim_fraction", 0.5)),
        profit_trim_fraction=float(payload.get("profit_trim_fraction", 0.5)),
    )


def _entry_quantile(panel: pd.DataFrame, entry_threshold: float) -> float:
    selected_share = float((panel["alpha_probability"] >= entry_threshold).mean())
    return float(np.clip(1.0 - selected_share, 0.80, 0.98))


def _rebalance_profiles(config: Trade60Config) -> list[dict]:
    return [
        {
            "hold_alpha_buffer": 0.08,
            "rank_keep_fraction": 1.0,
            "defensive_trim_fraction": 0.55,
            "weak_alpha_trim_fraction": 0.5,
            "profit_trim_fraction": 0.7,
        },
        {
            "hold_alpha_buffer": 0.08,
            "rank_keep_fraction": 1.0,
            "defensive_trim_fraction": 0.55,
            "weak_alpha_trim_fraction": 0.5,
            "profit_trim_fraction": 0.5,
        },
    ]
