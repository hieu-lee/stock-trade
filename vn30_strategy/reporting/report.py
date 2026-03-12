from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_strategy_report(
    path: Path,
    summary: dict,
    current_recommendations: pd.DataFrame,
) -> Path:
    lines = [
        "# VN30 Monthly Alpha Report",
        "",
        "## Summary",
        f"- Out-of-sample monthly hit rate: `{summary['holdout_metrics']['monthly_hit_rate']:.2%}`",
        f"- Passes 80% target: `{summary['holdout_metrics']['pass_80pct_requirement']}`",
        f"- Holdout max drawdown: `{summary['holdout_metrics']['max_drawdown']:.2%}`",
        f"- Holdout CAGR-like return: `{summary['holdout_metrics']['cagr_like']:.2%}`",
        "",
        "## Tactic",
        "- Trade only when the market-regime gate signals a favorable month.",
        "- Rank VN30 symbols by the probability of achieving at least 10% return over the next 20 trading days.",
        "- Hold only the top names above the probability threshold and keep the rest of the capital in cash.",
        "",
        "## Current Recommendations",
    ]

    if current_recommendations.empty:
        lines.append("- No current recommendation was produced.")
    else:
        for row in current_recommendations.itertuples():
            lines.append(
                f"- `{row.symbol}`: probability `{row.rank_probability:.2%}`, regime `{row.regime_probability:.2%}`, score reason `{row.explanation}`"
            )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
