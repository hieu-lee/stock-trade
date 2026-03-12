from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from vn30_strategy.backtest.engine import BacktestResult, run_monthly_backtest
from vn30_strategy.config import StrategyConfig
from vn30_strategy.models.ranker import RankerArtifacts, score_ranker, train_ranker
from vn30_strategy.models.regime import RegimeModelArtifacts, score_regime, train_regime_model


@dataclass(slots=True)
class WalkForwardArtifacts:
    best_params: dict
    validation_summary: dict
    in_sample_backtest: BacktestResult
    holdout_backtest: BacktestResult
    regime_artifacts: RegimeModelArtifacts
    ranker_artifacts: RankerArtifacts
    holdout_scored_panel: pd.DataFrame


def run_walkforward(panel: pd.DataFrame, config: StrategyConfig) -> WalkForwardArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    unique_dates = trainable["date"].drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 96:
        raise RuntimeError("Not enough monthly observations to run the requested walk-forward process.")

    holdout_dates = unique_dates[-12:]
    development_dates = unique_dates[:-12]
    development = trainable.loc[trainable["date"].isin(development_dates)].copy()
    holdout = trainable.loc[trainable["date"].isin(holdout_dates)].copy()
    windows = _build_validation_windows(development_dates)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: _objective(trial, development, windows, config), n_trials=config.optuna_trials)
    best_params = dict(study.best_params)
    best_params["regime_threshold"] = config.regime_threshold
    best_params["rank_threshold"] = config.rank_threshold
    best_params["max_positions"] = config.max_positions

    regime_artifacts, ranker_artifacts = _fit_models(development, best_params)
    dev_scored = _score_panel(development, regime_artifacts, ranker_artifacts)
    holdout_scored = _score_panel(holdout, regime_artifacts, ranker_artifacts)

    in_sample_backtest = run_monthly_backtest(
        dev_scored,
        config,
        regime_threshold=float(best_params["regime_threshold"]),
        rank_threshold=float(best_params["rank_threshold"]),
        max_positions=int(best_params["max_positions"]),
    )
    holdout_backtest = run_monthly_backtest(
        holdout_scored,
        config,
        regime_threshold=float(best_params["regime_threshold"]),
        rank_threshold=float(best_params["rank_threshold"]),
        max_positions=int(best_params["max_positions"]),
    )

    validation_summary = {
        "best_score": float(study.best_value),
        "best_params": best_params,
        "validation_windows": [
            {"train_start": str(train[0].date()), "train_end": str(train[-1].date()), "valid_start": str(valid[0].date()), "valid_end": str(valid[-1].date())}
            for train, valid in windows
        ],
    }
    return WalkForwardArtifacts(
        best_params=best_params,
        validation_summary=validation_summary,
        in_sample_backtest=in_sample_backtest,
        holdout_backtest=holdout_backtest,
        regime_artifacts=regime_artifacts,
        ranker_artifacts=ranker_artifacts,
        holdout_scored_panel=holdout_scored,
    )


def score_unlabeled_panel(panel: pd.DataFrame, regime_artifacts: RegimeModelArtifacts, ranker_artifacts: RankerArtifacts) -> pd.DataFrame:
    return _score_panel(panel, regime_artifacts, ranker_artifacts)


def _objective(trial: optuna.Trial, development: pd.DataFrame, windows: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]], config: StrategyConfig) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": config.random_state,
        "regime_threshold": config.regime_threshold,
        "rank_threshold": config.rank_threshold,
        "max_positions": config.max_positions,
    }

    scores: list[float] = []
    hit_rates: list[float] = []
    for train_dates, valid_dates in windows:
        train_df = development.loc[development["date"].isin(train_dates)].copy()
        valid_df = development.loc[development["date"].isin(valid_dates)].copy()
        if train_df["target_hit"].nunique() < 2 or valid_df.empty:
            continue
        try:
            regime_artifacts, ranker_artifacts = _fit_models(train_df, params)
            valid_scored = _score_panel(valid_df, regime_artifacts, ranker_artifacts)
            backtest = run_monthly_backtest(
                valid_scored,
                config,
                regime_threshold=float(params["regime_threshold"]),
                rank_threshold=float(params["rank_threshold"]),
                max_positions=int(params["max_positions"]),
            )
        except Exception:
            continue

        metrics = backtest.metrics
        traded_months = metrics["months_traded"]
        traded_share = metrics["months_traded"] / max(metrics["months_total"], 1)
        hit_rate = metrics["monthly_hit_rate"]
        undertrade_penalty = max(config.min_validation_traded_months - traded_months, 0) * 0.35
        score = (
            hit_rate * 2.5
            + metrics["average_monthly_return"] * 2.0
            - abs(traded_share - config.target_validation_traded_share) * 1.25
            - undertrade_penalty
            - abs(metrics["max_drawdown"]) * 0.5
        )
        scores.append(score)
        hit_rates.append(hit_rate)

    if not scores:
        return -10.0
    trial.set_user_attr("avg_validation_hit_rate", float(np.mean(hit_rates)))
    return float(np.mean(scores))


def _fit_models(train_df: pd.DataFrame, params: dict) -> tuple[RegimeModelArtifacts, RankerArtifacts]:
    regime_artifacts = train_regime_model(train_df, random_state=int(params.get("random_state", 42)))
    ranker_artifacts = train_ranker(train_df, params=params)
    return regime_artifacts, ranker_artifacts


def _score_panel(panel: pd.DataFrame, regime_artifacts: RegimeModelArtifacts, ranker_artifacts: RankerArtifacts) -> pd.DataFrame:
    scored = score_ranker(ranker_artifacts, panel)
    regime_scores = score_regime(regime_artifacts, panel)
    return scored.merge(regime_scores, on="date", how="left")


def _build_validation_windows(development_dates: list[pd.Timestamp], train_months: int = 72, valid_months: int = 12, step: int = 6) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    windows: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    start = train_months
    while start + valid_months <= len(development_dates):
        train_dates = development_dates[:start]
        valid_dates = development_dates[start : start + valid_months]
        windows.append((train_dates, valid_dates))
        start += step
    if not windows:
        raise RuntimeError("Could not build validation windows for walk-forward evaluation.")
    return windows
