from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
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

REGIME_MARKET_FEATURES = [
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

REGIME_RELATIVE_FEATURES = [
    "relative_strength_10d",
    "relative_strength_20d",
    "relative_strength_20d_rank",
    "beta_60d",
]

REGIME_FLOW_FEATURES = [
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
]

REGIME_SYMBOL_FEATURES = REGIME_RELATIVE_FEATURES + REGIME_FLOW_FEATURES

REGIME_FEATURES = REGIME_MARKET_FEATURES + REGIME_SYMBOL_FEATURES

COMPANY_OPTIONAL_FEATURES = {
    "sector_ret_20d",
    "sector_breadth_above_ma50",
    "sector_ret_20d_rank",
    "sector_turnover_20d",
}

EVENT_OPTIONAL_FEATURES = {
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
    "event_breadth_20d",
    "dividend_breadth_252d",
}

FLOW_OPTIONAL_FEATURES = {
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
}

UTILITY_WEIGHT_COLUMNS = (
    "alpha_rank",
    "risk_adjusted_rank",
    "relative_strength_20d_rank",
    "downside_score",
    "regime_probability",
)

DEFAULT_UTILITY_WEIGHTS = {
    "alpha_rank": 0.33,
    "risk_adjusted_rank": 0.11,
    "relative_strength_20d_rank": 0.33,
    "downside_score": 0.07,
    "regime_probability": 0.16,
}


class ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def fit(self, X, y=None, sample_weight=None):  # noqa: ANN001 - sklearn style
        self.classes_ = np.array([0.0, 1.0], dtype=float)
        self.n_features_in_ = int(getattr(X, "shape", [0, 0])[1]) if hasattr(X, "shape") else 0
        return self

    def predict(self, X) -> np.ndarray:  # noqa: ANN001 - sklearn style
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)

    def predict_proba(self, X) -> np.ndarray:  # noqa: ANN001 - sklearn style
        probability = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(getattr(self, "n_features_in_", 0), dtype=float)


@dataclass(slots=True)
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    target_name: str


@dataclass(slots=True)
class RegimeModelStack:
    anchor: ModelBundle
    participation: ModelBundle


@dataclass(slots=True)
class ModelStack:
    alpha: ModelBundle
    risk: ModelBundle
    regime: ModelBundle | RegimeModelStack


@dataclass(slots=True)
class ModelArtifacts:
    model_stack: ModelStack
    scored_panel: pd.DataFrame
    feature_importance: pd.DataFrame


def get_alpha_features(config: TradingBotConfig) -> list[str]:
    return _filter_optional_features(ALPHA_FEATURES, config)


def get_risk_features(config: TradingBotConfig) -> list[str]:
    return _filter_optional_features(RISK_FEATURES, config)


def get_regime_market_features(config: TradingBotConfig) -> list[str]:
    return _filter_optional_features(list(REGIME_MARKET_FEATURES), config)


def get_regime_symbol_features(config: TradingBotConfig) -> list[str]:
    if config.regime_use_market_only_features:
        return []
    return _filter_optional_features(list(REGIME_SYMBOL_FEATURES), config)


def get_regime_features(config: TradingBotConfig) -> list[str]:
    features = get_regime_market_features(config)
    features.extend(get_regime_symbol_features(config))
    return _filter_optional_features(features, config)


def build_regime_anchor_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    include_target: bool = False,
) -> pd.DataFrame:
    ordered_columns = ["date"]
    if include_target and "target_regime" in frame.columns:
        ordered_columns.append("target_regime")
    ordered_columns.extend(feature_columns)
    regime_frame = frame.reindex(columns=ordered_columns).copy()
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    return regime_frame.sort_values("date").drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)


def build_regime_symbol_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    include_target: bool = False,
    target_column: str = "target_regime",
) -> pd.DataFrame:
    ordered_columns = ["date", "symbol"]
    if include_target and target_column in frame.columns:
        ordered_columns.append(target_column)
    ordered_columns.extend(feature_columns)
    regime_frame = frame.reindex(columns=ordered_columns).copy()
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    regime_frame["symbol"] = regime_frame["symbol"].astype(str)
    return regime_frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_regime_sample_weights(frame: pd.DataFrame, config: TradingBotConfig) -> np.ndarray:
    if "symbol" not in frame.columns:
        return _time_decay_weights(frame["date"], half_life_years=config.regime_half_life_years)
    date_counts = frame.groupby("date")["symbol"].transform("count").clip(lower=1)
    time_weights = _time_decay_weights(frame["date"], half_life_years=config.regime_half_life_years)
    return time_weights / date_counts.to_numpy(dtype=float)


def _filter_optional_features(features: list[str], config: TradingBotConfig) -> list[str]:
    feature_columns = list(features)
    if not config.use_company_metadata_features:
        feature_columns = [column for column in feature_columns if column not in COMPANY_OPTIONAL_FEATURES]
    if not config.use_event_features:
        feature_columns = [column for column in feature_columns if column not in EVENT_OPTIONAL_FEATURES]
    if not config.use_fireant_flow_features:
        feature_columns = [column for column in feature_columns if column not in FLOW_OPTIONAL_FEATURES]
    return feature_columns


def train_model_stack(
    panel: pd.DataFrame,
    config: TradingBotConfig,
    persist: bool = False,
    utility_weights: Mapping[str, float] | None = None,
) -> ModelArtifacts:
    trainable = panel.loc[panel["is_trainable"]].copy()
    if trainable.empty:
        raise ValueError("No trainable rows found in feature panel.")
    alpha_bundle = _fit_alpha_model(trainable, config)
    risk_bundle = _fit_risk_model(trainable, config)
    regime_stack = _fit_regime_model(trainable, config)
    model_stack = ModelStack(alpha=alpha_bundle, risk=risk_bundle, regime=regime_stack)
    scored_panel = score_panel(panel, model_stack, utility_weights=utility_weights)
    feature_importance = _extract_feature_importance(alpha_bundle)
    artifacts = ModelArtifacts(model_stack=model_stack, scored_panel=scored_panel, feature_importance=feature_importance)
    if persist:
        save_model_artifacts(artifacts, config)
    return artifacts


def normalize_utility_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    normalized = {column: 0.0 for column in UTILITY_WEIGHT_COLUMNS}
    if weights is not None:
        for column, value in weights.items():
            if column not in normalized or not pd.notna(value):
                continue
            normalized[column] = max(float(value), 0.0)
    total = float(sum(normalized.values()))
    if total <= 0:
        return DEFAULT_UTILITY_WEIGHTS.copy()
    return {column: value / total for column, value in normalized.items()}


def apply_utility_weights(
    scored: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    utility_weights = normalize_utility_weights(weights)
    rescored = scored.copy()
    rescored["utility_score"] = sum(
        rescored[column].fillna(0.0) * utility_weights[column] for column in UTILITY_WEIGHT_COLUMNS
    )
    return rescored.sort_values(["date", "utility_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def score_panel(
    panel: pd.DataFrame,
    model_stack: ModelStack,
    utility_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    scored = panel.copy()
    alpha_features = _clean_features(scored.reindex(columns=model_stack.alpha.feature_columns))
    risk_features = _clean_features(scored.reindex(columns=model_stack.risk.feature_columns))
    scored["alpha_prediction"] = np.clip(model_stack.alpha.pipeline.predict(alpha_features), 0.0, 1.0)
    scored["risk_probability"] = _predict_probability(model_stack.risk.pipeline, risk_features)
    if isinstance(model_stack.regime, RegimeModelStack):
        anchor_features = _clean_features(scored.reindex(columns=model_stack.regime.anchor.feature_columns))
        participation_features = _clean_features(scored.reindex(columns=model_stack.regime.participation.feature_columns))
        scored["regime_anchor_probability"] = _predict_probability(model_stack.regime.anchor.pipeline, anchor_features)
        scored["regime_participation_probability"] = _predict_probability(
            model_stack.regime.participation.pipeline,
            participation_features,
        )
        scored["regime_probability"] = (
            scored["regime_anchor_probability"].fillna(0.5).clip(lower=0.0, upper=1.0)
            * scored["regime_participation_probability"].fillna(1.0).clip(lower=0.0, upper=1.0)
        )
    else:
        regime_features = _clean_features(scored.reindex(columns=model_stack.regime.feature_columns))
        scored["regime_probability"] = _predict_probability(model_stack.regime.pipeline, regime_features)
        scored["regime_anchor_probability"] = scored["regime_probability"]
        scored["regime_participation_probability"] = 1.0
    scored["alpha_rank"] = scored.groupby("date")["alpha_prediction"].rank(pct=True, ascending=True)
    scored["risk_adjusted_alpha"] = scored["alpha_prediction"].fillna(0.0) * (
        1.0 - scored["risk_probability"].fillna(1.0).clip(lower=0.0, upper=1.0)
    )
    scored["risk_adjusted_rank"] = scored.groupby("date")["risk_adjusted_alpha"].rank(pct=True, ascending=True)
    scored["downside_score"] = 1.0 - scored["risk_probability"].fillna(1.0).clip(lower=0.0, upper=1.0)
    return apply_utility_weights(scored, weights=utility_weights)


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
    feature_columns = get_alpha_features(config)
    estimator = _build_regressor(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    alpha_weights = _time_decay_weights(trainable["date"], half_life_years=config.alpha_half_life_years)
    pipeline.fit(
        _clean_features(trainable.reindex(columns=feature_columns)),
        trainable["target_alpha_blend"],
        estimator__sample_weight=alpha_weights,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=feature_columns, target_name="target_alpha_blend")


def _fit_risk_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    feature_columns = get_risk_features(config)
    estimator = _build_classifier(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    risk_weights = _time_decay_weights(trainable["date"], half_life_years=config.risk_half_life_years)
    pipeline.fit(
        _clean_features(trainable.reindex(columns=feature_columns)),
        trainable["target_downside"],
        estimator__sample_weight=risk_weights,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=feature_columns, target_name="target_downside")


def _fit_regime_model(trainable: pd.DataFrame, config: TradingBotConfig) -> RegimeModelStack:
    return RegimeModelStack(
        anchor=_fit_regime_anchor_model(trainable, config),
        participation=_fit_regime_participation_model(trainable, config),
    )


def _fit_regime_anchor_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    feature_columns = get_regime_market_features(config)
    regime_train = build_regime_anchor_frame(trainable, feature_columns, include_target=True)
    return _fit_probability_model_bundle(
        frame=regime_train,
        feature_columns=feature_columns,
        target_column="target_regime",
        target_name="target_regime",
        config=config,
        fallback_probability=0.5,
    )


def _fit_regime_participation_model(trainable: pd.DataFrame, config: TradingBotConfig) -> ModelBundle:
    feature_columns = get_regime_symbol_features(config)
    if not feature_columns:
        return _constant_probability_bundle(
            feature_columns=feature_columns,
            target_name="target_regime_participation",
            probability=1.0,
        )
    regime_train = build_regime_symbol_frame(
        trainable,
        feature_columns,
        include_target=True,
        target_column="target_regime_participation",
    )
    regime_train = regime_train.loc[regime_train["target_regime_participation"].notna()].reset_index(drop=True)
    if len(regime_train) < max(len(feature_columns) * 4, 128):
        return _constant_probability_bundle(
            feature_columns=feature_columns,
            target_name="target_regime_participation",
            probability=1.0,
        )
    if regime_train["target_regime_participation"].nunique(dropna=True) < 2:
        return _constant_probability_bundle(
            feature_columns=feature_columns,
            target_name="target_regime_participation",
            probability=1.0,
        )
    estimator = _build_regime_classifier(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    regime_weights = build_regime_sample_weights(regime_train, config)
    pipeline.fit(
        _clean_features(regime_train.reindex(columns=feature_columns)),
        regime_train["target_regime_participation"],
        estimator__sample_weight=regime_weights,
    )
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_name="target_regime_participation",
    )


def _fit_probability_model_bundle(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_name: str,
    config: TradingBotConfig,
    fallback_probability: float,
) -> ModelBundle:
    train_frame = frame.loc[frame[target_column].notna()].reset_index(drop=True)
    if train_frame.empty or train_frame[target_column].nunique(dropna=True) < 2:
        return _constant_probability_bundle(
            feature_columns=feature_columns,
            target_name=target_name,
            probability=fallback_probability,
        )
    estimator = _build_regime_classifier(config)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("estimator", estimator),
        ]
    )
    sample_weight = build_regime_sample_weights(train_frame, config)
    pipeline.fit(
        _clean_features(train_frame.reindex(columns=feature_columns)),
        train_frame[target_column],
        estimator__sample_weight=sample_weight,
    )
    return ModelBundle(pipeline=pipeline, feature_columns=feature_columns, target_name=target_name)


def _constant_probability_bundle(
    feature_columns: list[str],
    target_name: str,
    probability: float,
) -> ModelBundle:
    pipeline = Pipeline([("estimator", ConstantProbabilityClassifier(probability=probability))])
    dummy_features = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    pipeline.fit(dummy_features, np.array([1.0], dtype=float))
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_name=target_name,
    )


def _build_regressor(config: TradingBotConfig):
    try:
        from xgboost import XGBRegressor
    except Exception:  # pragma: no cover - optional runtime dependency
        return HistGradientBoostingRegressor(
            learning_rate=float(config.regressor_learning_rate),
            max_depth=int(config.regressor_max_depth),
            max_iter=int(config.regressor_n_estimators),
            random_state=config.random_state,
        )
    return XGBRegressor(
        n_estimators=int(config.regressor_n_estimators),
        learning_rate=float(config.regressor_learning_rate),
        max_depth=int(config.regressor_max_depth),
        subsample=float(config.regressor_subsample),
        colsample_bytree=float(config.regressor_colsample_bytree),
        reg_alpha=float(config.regressor_reg_alpha),
        reg_lambda=float(config.regressor_reg_lambda),
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=1,
    )


def _build_classifier(config: TradingBotConfig):
    try:
        from xgboost import XGBClassifier
    except Exception:  # pragma: no cover - optional runtime dependency
        return HistGradientBoostingClassifier(
            learning_rate=float(config.classifier_learning_rate),
            max_depth=int(config.classifier_max_depth),
            max_iter=int(config.classifier_n_estimators),
            random_state=config.random_state,
        )
    return XGBClassifier(
        n_estimators=int(config.classifier_n_estimators),
        learning_rate=float(config.classifier_learning_rate),
        max_depth=int(config.classifier_max_depth),
        subsample=float(config.classifier_subsample),
        colsample_bytree=float(config.classifier_colsample_bytree),
        reg_alpha=float(config.classifier_reg_alpha),
        reg_lambda=float(config.classifier_reg_lambda),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=1,
    )


def _build_regime_classifier(config: TradingBotConfig):
    try:
        from xgboost import XGBClassifier
    except Exception:  # pragma: no cover - optional runtime dependency
        return HistGradientBoostingClassifier(
            learning_rate=float(config.regime_classifier_learning_rate),
            max_depth=int(config.regime_classifier_max_depth),
            max_iter=int(config.regime_classifier_n_estimators),
            random_state=config.random_state,
        )
    return XGBClassifier(
        n_estimators=int(config.regime_classifier_n_estimators),
        learning_rate=float(config.regime_classifier_learning_rate),
        max_depth=int(config.regime_classifier_max_depth),
        subsample=float(config.regime_classifier_subsample),
        colsample_bytree=float(config.regime_classifier_colsample_bytree),
        reg_alpha=float(config.regime_classifier_reg_alpha),
        reg_lambda=float(config.regime_classifier_reg_lambda),
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
