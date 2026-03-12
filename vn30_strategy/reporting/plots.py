from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vn30_strategy.backtest.engine import BacktestResult
from vn30_strategy.utils import ensure_directories, write_json


def generate_report_artifacts(
    reports_dir: Path,
    in_sample: BacktestResult,
    holdout: BacktestResult,
    feature_importance: pd.DataFrame,
) -> dict[str, str]:
    ensure_directories([reports_dir])
    outputs = {
        "in_sample_equity": str(_plot_equity_curve(in_sample.monthly_returns, reports_dir / "in_sample_equity.png", "In-sample equity curve")),
        "holdout_equity": str(_plot_equity_curve(holdout.monthly_returns, reports_dir / "holdout_equity.png", "Holdout equity curve")),
        "holdout_hit_rate": str(_plot_yearly_hit_rate(holdout.monthly_returns, reports_dir / "holdout_yearly_hit_rate.png")),
        "feature_importance": str(_plot_feature_importance(feature_importance, reports_dir / "feature_importance.png")),
    }
    write_json(in_sample.metrics, reports_dir / "in_sample_metrics.json")
    write_json(holdout.metrics, reports_dir / "holdout_metrics.json")
    in_sample.monthly_returns.to_csv(reports_dir / "in_sample_monthly_returns.csv", index=False)
    holdout.monthly_returns.to_csv(reports_dir / "holdout_monthly_returns.csv", index=False)
    if not holdout.trades.empty:
        holdout.trades.to_csv(reports_dir / "holdout_trades.csv", index=False)
    return outputs


def _plot_equity_curve(monthly_returns: pd.DataFrame, path: Path, title: str) -> Path:
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_returns["date"], monthly_returns["equity_curve"], linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_yearly_hit_rate(monthly_returns: pd.DataFrame, path: Path) -> Path:
    yearly = monthly_returns.loc[monthly_returns["is_traded"]].copy()
    if yearly.empty:
        yearly = monthly_returns.copy()
    yearly["year"] = yearly["date"].dt.year
    summary = yearly.groupby("year", as_index=False)["target_hit"].mean()
    plt.figure(figsize=(10, 4))
    sns.barplot(data=summary, x="year", y="target_hit", color="#2a9d8f")
    plt.title("Yearly target hit rate")
    plt.xlabel("Year")
    plt.ylabel("Hit rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _plot_feature_importance(feature_importance: pd.DataFrame, path: Path) -> Path:
    if feature_importance.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
    top_features = feature_importance.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x="importance", y="feature", color="#264653")
    plt.title("Top ranker features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
