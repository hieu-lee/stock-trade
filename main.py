from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from vn30_strategy.config import UNIVERSE_SYMBOLS, get_default_config
from vn30_strategy.service.pipeline import (
    build_feature_store,
    download_data,
    generate_current_recommendations,
    load_trained_artifacts,
    train_strategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VN30 monthly alpha strategy CLI")
    parser.add_argument("--workspace", default=Path(__file__).resolve().parent, help="Workspace directory")
    parser.add_argument("--profile", default="vn30_core", choices=sorted(UNIVERSE_SYMBOLS), help="Universe profile to run")
    parser.add_argument("--refresh", action="store_true", help="Refresh downloaded data before running")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in [
        ("download-data", "Download and cache raw market data"),
        ("build-features", "Build the research feature store"),
        ("train", "Train models, backtest, and export reports"),
        ("recommend", "Show the latest monthly recommendations"),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--refresh", action="store_true", help="Refresh downloaded data before running")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = get_default_config(args.workspace, profile_name=args.profile)

    if args.command == "download-data":
        bundle = download_data(config, refresh=args.refresh)
        print(
            json.dumps(
                {
                    "symbols": len(bundle.universe),
                    "price_rows": len(bundle.prices),
                    "benchmark_rows": len(bundle.benchmarks),
                    "overview_rows": len(bundle.overviews),
                    "ratio_rows": len(bundle.ratios),
                },
                indent=2,
                default=str,
            )
        )
        return

    if args.command == "build-features":
        panel = build_feature_store(config, refresh_data=args.refresh)
        print(panel.tail(10).to_string(index=False))
        return

    if args.command == "train":
        summary = train_strategy(config, refresh_data=args.refresh)
        print(json.dumps(summary["holdout_metrics"], indent=2, default=str))
        return

    if args.command == "recommend":
        panel = pd.read_parquet(config.processed_dir / "monthly_feature_panel.parquet")
        ranker, regime = load_trained_artifacts(config)
        from vn30_strategy.backtest.walkforward import score_unlabeled_panel

        unlabeled = panel.loc[~panel["is_trainable"]].copy()
        if unlabeled.empty:
            unlabeled = panel.groupby("symbol", as_index=False).tail(1).copy()
        latest_panel = unlabeled.loc[unlabeled["date"] == unlabeled["date"].max()].copy()
        scored = score_unlabeled_panel(latest_panel, regime, ranker)
        scored = scored.sort_values("rank_probability", ascending=False)
        cols = ["date", "symbol", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200"]
        print(scored[cols].head(10).to_string(index=False))
        return


if __name__ == "__main__":
    main()
