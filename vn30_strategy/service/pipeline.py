from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from vn30_strategy.backtest.walkforward import WalkForwardArtifacts, run_walkforward, score_unlabeled_panel
from vn30_strategy.config import StrategyConfig
from vn30_strategy.data.adjustments import prepare_research_data
from vn30_strategy.data.collector import DataBundle, VN30DataCollector, load_cached_bundle
from vn30_strategy.features.fundamental import build_fundamental_features
from vn30_strategy.features.technical import build_technical_features
from vn30_strategy.models.labels import attach_forward_targets
from vn30_strategy.reporting.plots import generate_report_artifacts
from vn30_strategy.reporting.report import export_strategy_report
from vn30_strategy.utils import ensure_directories, write_json

LOGGER = logging.getLogger(__name__)


def download_data(config: StrategyConfig, refresh: bool = False) -> DataBundle:
    collector = VN30DataCollector(config)
    return collector.collect_all(refresh=refresh)


def build_feature_store(config: StrategyConfig, refresh_data: bool = False) -> pd.DataFrame:
    try:
        bundle = download_data(config, refresh=refresh_data) if refresh_data else load_cached_bundle(config)
    except FileNotFoundError:
        LOGGER.info("Cached raw data not found. Downloading fresh data.")
        bundle = download_data(config, refresh=True)
    prepared = prepare_research_data(bundle, config)
    daily_features, monthly_features = build_technical_features(prepared.prices, prepared.benchmarks)
    monthly_features = build_fundamental_features(monthly_features, bundle.overviews, bundle.ratios)
    panel = attach_forward_targets(prepared.prices, monthly_features, config.target_holding_days, config.target_return)
    panel.to_parquet(config.processed_dir / "monthly_feature_panel.parquet", index=False)
    return panel


def train_strategy(config: StrategyConfig, refresh_data: bool = False) -> dict:
    ensure_directories([config.artifacts_dir, config.models_dir, config.reports_dir, config.recommendations_dir])
    panel = build_feature_store(config, refresh_data=refresh_data)
    artifacts = run_walkforward(panel, config)
    feature_importance = _extract_feature_importance(artifacts)
    current_recommendations = generate_current_recommendations(panel, artifacts, config)

    feature_importance.to_csv(config.models_dir / "feature_importance.csv", index=False)
    joblib.dump(artifacts.ranker_artifacts, config.models_dir / "ranker.joblib")
    joblib.dump(artifacts.regime_artifacts, config.models_dir / "regime.joblib")
    write_json(artifacts.best_params, config.models_dir / "best_params.json")
    artifacts.holdout_scored_panel.to_parquet(config.models_dir / "holdout_scored_panel.parquet", index=False)

    report_images = generate_report_artifacts(
        config.reports_dir,
        artifacts.in_sample_backtest,
        artifacts.holdout_backtest,
        feature_importance,
    )
    current_recommendations.to_csv(config.recommendations_dir / "current_recommendations.csv", index=False)
    current_recommendations.to_json(config.recommendations_dir / "current_recommendations.json", orient="records", indent=2)

    summary = {
        "config": asdict(config),
        "best_params": artifacts.best_params,
        "validation_summary": artifacts.validation_summary,
        "in_sample_metrics": artifacts.in_sample_backtest.metrics,
        "holdout_metrics": artifacts.holdout_backtest.metrics,
        "report_images": report_images,
        "recommendation_count": int(len(current_recommendations)),
    }
    write_json(summary, config.artifacts_dir / "summary.json")
    export_strategy_report(config.reports_dir / "strategy_report.md", summary, current_recommendations)
    return summary


def load_trained_artifacts(config: StrategyConfig) -> tuple[object, object]:
    ranker = joblib.load(config.models_dir / "ranker.joblib")
    regime = joblib.load(config.models_dir / "regime.joblib")
    return ranker, regime


def generate_current_recommendations(panel: pd.DataFrame, artifacts: WalkForwardArtifacts, config: StrategyConfig) -> pd.DataFrame:
    unlabeled = panel.loc[~panel["is_trainable"]].copy()
    if unlabeled.empty:
        unlabeled = panel.groupby("symbol", as_index=False).tail(1).copy()
    latest_date = unlabeled["date"].max()
    latest_panel = unlabeled.loc[unlabeled["date"] == latest_date].copy()
    scored = score_unlabeled_panel(latest_panel, artifacts.regime_artifacts, artifacts.ranker_artifacts)
    recommendations = scored.loc[
        (scored["regime_probability"] >= float(artifacts.best_params["regime_threshold"]))
        & (scored["rank_probability"] >= float(artifacts.best_params["rank_threshold"]))
    ].copy()
    recommendations = recommendations.sort_values("rank_probability", ascending=False).head(int(artifacts.best_params["max_positions"]))
    recommendations["explanation"] = recommendations.apply(_explain_row, axis=1)
    columns = ["date", "symbol", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"]
    for column in columns:
        if column not in recommendations.columns:
            recommendations[column] = pd.NA
    return recommendations[columns].reset_index(drop=True)


def backtest_from_saved_panel(config: StrategyConfig) -> dict:
    if not (config.processed_dir / "monthly_feature_panel.parquet").exists():
        raise FileNotFoundError("Run `main.py build-features` or `main.py train` before backtesting saved data.")
    return train_strategy(config, refresh_data=False)


def _extract_feature_importance(artifacts: WalkForwardArtifacts) -> pd.DataFrame:
    classifier = artifacts.ranker_artifacts.model.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(
        {"feature": artifacts.ranker_artifacts.feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)


def _explain_row(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("ret_60d", 0) > 0:
        reasons.append("strong 3-month momentum")
    if row.get("distance_ma200", -1) > 0:
        reasons.append("trading above 200-day trend")
    if row.get("roe", 0) > 0.15:
        reasons.append("high ROE")
    pe = row.get("pe")
    if pd.notna(pe) and pe < 15:
        reasons.append("reasonable valuation")
    return ", ".join(reasons) if reasons else "composite model conviction"
