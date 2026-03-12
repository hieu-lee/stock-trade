from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class RegimeModelArtifacts:
    feature_columns: list[str]
    model: Pipeline


def train_regime_model(train_df: pd.DataFrame, random_state: int = 42) -> RegimeModelArtifacts:
    month_level = _to_month_level(train_df)
    candidate_features = [
        col for col in month_level.columns if col not in {"date", "month_target"} and month_level[col].notna().any()
    ]
    sanitized = _sanitize_features(month_level[candidate_features])
    features = [col for col in sanitized.columns if sanitized[col].notna().any() and sanitized[col].nunique(dropna=True) > 1]
    if not features:
        features = [col for col in sanitized.columns if sanitized[col].notna().any()]
    if not features:
        month_level["fallback_feature"] = 0.0
        sanitized = _sanitize_features(month_level[["fallback_feature"]])
        features = ["fallback_feature"]
    classifier = (
        DummyClassifier(strategy="constant", constant=int(month_level["month_target"].iloc[0]))
        if month_level["month_target"].nunique() < 2
        else RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    train_x = sanitized[features]
    model.fit(train_x, month_level["month_target"])
    return RegimeModelArtifacts(feature_columns=features, model=model)


def score_regime(model_artifacts: RegimeModelArtifacts, scored_df: pd.DataFrame) -> pd.DataFrame:
    month_level = _to_month_level(scored_df)
    for column in model_artifacts.feature_columns:
        if column not in month_level.columns:
            month_level[column] = np.nan
    score_x = _sanitize_features(month_level[model_artifacts.feature_columns])
    month_level["regime_probability"] = model_artifacts.model.predict_proba(score_x)[:, 1]
    return month_level[["date", "regime_probability"]]


def _to_month_level(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "benchmark_ret_20d": "median",
        "benchmark_ret_60d": "median",
        "benchmark_vol_20d": "median",
        "benchmark_distance_ma200": "median",
        "benchmark_drawdown_252d": "median",
        "breadth_above_ma200": "median",
        "breadth_positive_20d": "median",
        "breadth_ret_20d": "median",
        "breadth_ret_60d": "median",
        "ret_20d": "median",
        "ret_60d": "median",
        "volatility_20d": "median",
        "distance_ma200": "median",
        "avg_turnover_20d": "median",
        "target_hit": "max",
    }
    available = {key: value for key, value in aggregations.items() if key in df.columns}
    month_level = df.groupby("date", as_index=False).agg(available)
    month_level = month_level.rename(columns={"target_hit": "month_target"})
    month_level["month_target"] = month_level["month_target"].fillna(0.0)
    return month_level.sort_values("date").reset_index(drop=True)


def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.clip(lower=-100.0, upper=100.0)
