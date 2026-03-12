from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


EXCLUDED_COLUMNS = {
    "symbol",
    "date",
    "month",
    "entry_price",
    "exit_price",
    "exit_date",
    "forward_return_20d",
    "target_hit",
    "is_trainable",
    "report_date",
}


@dataclass(slots=True)
class RankerArtifacts:
    feature_columns: list[str]
    model: Pipeline


def train_ranker(train_df: pd.DataFrame, params: dict | None = None) -> RankerArtifacts:
    params = params or {}
    features = [
        col
        for col in train_df.columns
        if col not in EXCLUDED_COLUMNS and train_df[col].notna().any()
    ]
    classifier = (
        DummyClassifier(strategy="constant", constant=int(train_df["target_hit"].iloc[0]))
        if train_df["target_hit"].nunique() < 2
        else XGBClassifier(
            n_estimators=int(params.get("n_estimators", 250)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            min_child_weight=float(params.get("min_child_weight", 2.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(params.get("random_state", 42)),
        )
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    train_x = _sanitize_features(train_df[features])
    model.fit(train_x, train_df["target_hit"].astype(int))
    return RankerArtifacts(feature_columns=features, model=model)


def score_ranker(artifacts: RankerArtifacts, scored_df: pd.DataFrame) -> pd.DataFrame:
    scored = scored_df.copy()
    score_x = _sanitize_features(scored[artifacts.feature_columns])
    scored["rank_probability"] = artifacts.model.predict_proba(score_x)[:, 1]
    return scored


def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).clip(lower=-1_000_000, upper=1_000_000)
