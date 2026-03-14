from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from rdtb.config import TradingBotConfig

ALPHA_FEATURES = [
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
    "relative_strength_10d",
    "relative_strength_20d",
    "relative_strength_20d_rank",
    "distance_ma50_rank",
    "volume_zscore_20d_rank",
    "benchmark_ret_20d",
    "benchmark_distance_ma200",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "breadth_positive_10d",
    "breadth_ret_20d",
    "beta_60d",
    "revenue_yoy_pct",
    "parent_profit_yoy_pct",
    "roe_pct",
    "roa_pct",
    "gross_profit_margin_pct",
    "net_profit_margin_pct",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "interest_coverage",
    "asset_turnover",
    "inventory_turnover",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "eps_vnd",
    "bvps_vnd",
    "fundamental_growth_score",
    "fundamental_quality_score",
    "fundamental_value_score",
    "fundamental_freshness_days",
    "sector_ret_20d",
    "sector_breadth_above_ma50",
    "sector_ret_20d_rank",
    "sector_turnover_20d",
    "quality_breadth",
    "growth_breadth",
    "value_breadth",
    "recent_event_count_20d",
    "recent_event_count_60d",
    "recent_dividend_event_count_252d",
    "recent_issue_event_count_252d",
    "recent_listing_event_count_252d",
    "days_since_last_event",
    "days_since_last_dividend_event",
    "days_since_last_issue_event",
    "latest_dividend_value",
    "latest_issue_ratio",
    "upcoming_record_days",
    "upcoming_exright_days",
    "event_score",
    "event_score_rank",
    "foreign_buy_ratio",
    "foreign_sell_ratio",
    "foreign_net_ratio",
    "foreign_value_net_ratio",
    "foreign_room_ratio",
    "foreign_flow_5d",
    "foreign_flow_20d",
    "room_change_5d",
    "order_quantity_imbalance",
    "order_count_imbalance",
    "putthrough_ratio",
    "deal_ratio",
    "market_cap_turnover_ratio",
    "foreign_flow_score",
    "order_pressure_score",
    "foreign_flow_score_rank",
    "order_pressure_score_rank",
    "spy_ret_5d",
    "spy_ret_20d",
    "qqq_ret_20d",
    "eem_ret_20d",
    "fxi_ret_20d",
    "tlt_ret_20d",
    "gld_ret_20d",
    "uup_ret_20d",
    "global_equity_momentum_5d",
    "global_equity_momentum_20d",
    "global_defensive_momentum_20d",
    "global_equity_trend_ma50",
    "global_risk_on_score",
]

RISK_FEATURES = [
    "volatility_10d",
    "volatility_20d",
    "range_pct",
    "atr_pct",
    "distance_ma20",
    "distance_ma50",
    "distance_ma200",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rsi_14",
    "benchmark_ret_20d",
    "benchmark_vol_20d",
    "benchmark_drawdown_252d",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "beta_60d",
    "debt_to_equity",
    "current_ratio",
    "interest_coverage",
    "fundamental_quality_score",
    "fundamental_growth_score",
    "fundamental_value_score",
    "fundamental_freshness_days",
    "sector_ret_20d",
    "sector_breadth_above_ma50",
    "quality_breadth",
    "growth_breadth",
    "recent_event_count_20d",
    "recent_dividend_event_count_252d",
    "recent_issue_event_count_252d",
    "days_since_last_event",
    "days_since_last_issue_event",
    "latest_issue_ratio",
    "upcoming_record_days",
    "upcoming_exright_days",
    "event_score",
    "foreign_net_ratio",
    "foreign_value_net_ratio",
    "foreign_room_ratio",
    "foreign_flow_5d",
    "foreign_flow_20d",
    "room_change_5d",
    "order_quantity_imbalance",
    "putthrough_ratio",
    "deal_ratio",
    "market_cap_turnover_ratio",
    "foreign_flow_score",
    "order_pressure_score",
    "spy_ret_5d",
    "spy_ret_20d",
    "eem_ret_20d",
    "fxi_ret_20d",
    "tlt_ret_20d",
    "uup_ret_20d",
    "global_equity_momentum_20d",
    "global_defensive_momentum_20d",
    "global_risk_on_score",
]

REGIME_FEATURES = [
    "benchmark_ret_1d",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "benchmark_ret_20d",
    "benchmark_gap_open",
    "benchmark_intraday_return",
    "benchmark_vol_20d",
    "benchmark_distance_ma200",
    "benchmark_drawdown_252d",
    "breadth_above_ma50",
    "breadth_above_ma200",
    "breadth_positive_10d",
    "breadth_ret_20d",
    "breadth_turnover_20d",
    "quality_breadth",
    "growth_breadth",
    "value_breadth",
    "event_breadth_20d",
    "dividend_breadth_252d",
    "foreign_net_ratio",
    "foreign_value_net_ratio",
    "foreign_room_ratio",
    "foreign_flow_5d",
    "foreign_flow_20d",
    "room_change_5d",
    "order_quantity_imbalance",
    "putthrough_ratio",
    "deal_ratio",
    "market_cap_turnover_ratio",
    "foreign_flow_score",
    "order_pressure_score",
    "spy_ret_5d",
    "spy_ret_20d",
    "eem_ret_20d",
    "fxi_ret_20d",
    "tlt_ret_20d",
    "uup_ret_20d",
    "global_equity_momentum_20d",
    "global_defensive_momentum_20d",
    "global_equity_trend_ma50",
    "global_risk_on_score",
]


@dataclass(slots=True)
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    target_name: str


@dataclass(slots=True)
class ModelStack:
    alpha: ModelBundle
    risk: ModelBundle
    regime: ModelBundle


@dataclass(slots=True)
class ModelArtifacts:
    model_stack: ModelStack
    scored_panel: pd.DataFrame
    feature_importance: pd.DataFrame


def train_model_stack(
    panel: pd.DataFrame,
    config: TradingBotConfig,
    persist: bool = False,
) -> ModelArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy()
    if trainable.empty:
        raise ValueError("No trainable rows found in feature panel.")
    alpha_bundle = _fit_alpha_model(trainable, config)
    risk_bundle = _fit_risk_model(trainable, config)
    regime_bundle = _fit_regime_model(trainable, config)
    model_stack = ModelStack(alpha=alpha_bundle, risk=risk_bundle, regime=regime_bundle)
    scored_panel = score_panel(panel, model_stack)
    feature_importance = _extract_feature_importance(alpha_bundle)
    artifacts = ModelArtifacts(model_stack=model_stack, scored_panel=scored_panel, feature_importance=feature_importance)
    if persist:
        save_model_artifacts(artifacts, config)
    return artifacts


def score_panel(panel: pd.DataFrame, model_stack: ModelStack) -> pd.DataFrame:
    scored = panel.copy()
    alpha_features = _clean_features(scored.reindex(columns=model_stack.alpha.feature_columns))
    risk_features = _clean_features(scored.reindex(columns=model_stack.risk.feature_columns))
    scored["alpha_prediction"] = np.clip(model_stack.alpha.pipeline.predict(alpha_features), 0.0, 1.0)
    scored["risk_probability"] = _predict_probability(model_stack.risk.pipeline, risk_features)

    regime_daily = pd.concat(
        [
            scored[["date"]].copy(),
            scored.reindex(columns=model_stack.regime.feature_columns),
        ],
        axis=1,
    ).drop_duplicates(subset=["date"]).sort_values("date")
    regime_daily["regime_probability"] = _predict_probability(
        model_stack.regime.pipeline,
        _clean_features(regime_daily[model_stack.regime.feature_columns]),
    )
    scored = scored.merge(regime_daily[["date", "regime_probability"]], on="date", how="left")
    scored["alpha_rank"] = scored.groupby("date")["alpha_prediction"].rank(pct=True, ascending=True)
    scored["utility_score"] = (
        scored["alpha_rank"].fillna(0.0) * 0.75
        + scored["relative_strength_20d_rank"].fillna(0.0) * 0.20
        + scored["regime_probability"].fillna(0.0) * 0.05
    )
    return scored.sort_values(["date", "utility_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def save_model_artifacts(artifacts: ModelArtifacts, config: TradingBotConfig) -> None:
    joblib.dump(artifacts.model_stack.alpha, config.alpha_model_path)
    joblib.dump(artifacts.model_stack.risk, config.risk_model_path)
    joblib.dump(artifacts.model_stack.regime, config.regime_model_path)
    artifacts.scored_panel.to_parquet(config.scored_panel_path, index=False)
    artifacts.feature_importance.to_parquet(config.reports_dir / "feature_importance.parquet", index=False)


def load_model_artifacts(config: TradingBotConfig) -> ModelStack:
    return ModelStack(
        alpha=joblib.load(config.alpha_model_path),
        risk=joblib.load(config.risk_model_path),
        regime=joblib.load(config.regime_model_path),
    )


def _fit_alpha_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    estimator = _build_regressor(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    alpha_weights = _time_decay_weights(trainable["date"], half_life_years=config.alpha_half_life_years)
    pipeline.fit(
        _clean_features(trainable.reindex(columns=ALPHA_FEATURES)),
        trainable["target_alpha_blend"],
        estimator__sample_weight=alpha_weights,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=ALPHA_FEATURES, target_name="target_alpha_blend")


def _fit_risk_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    estimator = _build_classifier(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    risk_weights = _time_decay_weights(trainable["date"], half_life_years=config.risk_half_life_years)
    pipeline.fit(
        _clean_features(trainable.reindex(columns=RISK_FEATURES)),
        trainable["target_downside"],
        estimator__sample_weight=risk_weights,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=RISK_FEATURES, target_name="target_downside")


def _fit_regime_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    daily = pd.concat(
        [
            trainable[["date", "target_regime"]].copy(),
            trainable.reindex(columns=REGIME_FEATURES),
        ],
        axis=1,
    ).drop_duplicates(subset=["date"]).sort_values("date")
    estimator = _build_classifier(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    regime_weights = _time_decay_weights(daily["date"], half_life_years=config.regime_half_life_years)
    pipeline.fit(
        _clean_features(daily.reindex(columns=REGIME_FEATURES)),
        daily["target_regime"],
        estimator__sample_weight=regime_weights,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=REGIME_FEATURES, target_name="target_regime")


def _build_regressor(config: TradingBotConfig):
    try:
        from xgboost import XGBRegressor
    except Exception:  # pragma: no cover - optional runtime dependency
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=250,
            random_state=config.random_state,
        )
    return XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.02,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=1,
    )


def _build_classifier(config: TradingBotConfig):
    try:
        from xgboost import XGBClassifier
    except Exception:  # pragma: no cover - optional runtime dependency
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=220,
            random_state=config.random_state,
        )
    return XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=1,
    )


def _predict_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = pipeline.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    raw = pipeline.predict(features)
    return np.clip(raw, 0.0, 1.0)


def _clean_features(features: pd.DataFrame) -> pd.DataFrame:
    return features.replace([np.inf, -np.inf], np.nan)


def _time_decay_weights(dates: pd.Series, half_life_years: float) -> np.ndarray:
    timestamps = pd.to_datetime(dates)
    if timestamps.empty:
        return np.array([], dtype=float)
    latest = timestamps.max()
    age_days = (latest - timestamps).dt.days.clip(lower=0)
    half_life_days = max(int(half_life_years * 252), 1)
    weights = 0.5 ** (age_days / half_life_days)
    return np.clip(weights.to_numpy(dtype=float), 0.15, 1.0)


def _extract_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    estimator = bundle.pipeline.named_steps["estimator"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame({"feature": bundle.feature_columns, "importance": np.nan})
    return (
        pd.DataFrame({"feature": bundle.feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
