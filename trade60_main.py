from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from trade60_app import get_default_config
from trade60_app.service import (
    build_feature_store,
    build_holdings_template,
    download_data,
    generate_trade_plan,
    train_strategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trade60 long-only next-day trading system")
    parser.add_argument("--workspace", default=Path(__file__).resolve().parent, help="Workspace directory")
    parser.add_argument("--refresh", action="store_true", help="Refresh market data before running")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download-data", help="Download and cache raw daily data")
    subparsers.add_parser("build-features", help="Build the daily feature panel")
    subparsers.add_parser("train", help="Train models, backtest, and export reports")
    subparsers.add_parser("template", help="Print an empty holdings template")

    recommend = subparsers.add_parser("recommend", help="Generate tomorrow's buy/sell/hold-adjustment plan")
    recommend.add_argument("--budget", type=float, default=100_000.0, help="Available cash budget")
    recommend.add_argument(
        "--portfolio",
        type=Path,
        help="CSV file with symbol,quantity,avg_cost,buy_date columns; blank buy_date defaults to about one month ago",
    )
    recommend.add_argument("--json", action="store_true", help="Print the full recommendation as JSON")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = get_default_config(args.workspace)

    if args.command == "download-data":
        bundle = download_data(config, refresh=args.refresh)
        print(
            json.dumps(
                {
                    "symbols": len(bundle.universe),
                    "price_rows": len(bundle.prices),
                    "benchmark_rows": len(bundle.benchmarks),
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

    if args.command == "template":
        template = build_holdings_template(config)
        print(template.to_csv(index=False))
        return

    if args.command == "recommend":
        holdings = build_holdings_template(config)
        if args.portfolio:
            holdings = pd.read_csv(args.portfolio)
        result = generate_trade_plan(config, budget=args.budget, holdings=holdings, refresh_data=args.refresh)
        if args.json:
            payload = {
                "latest_signal_date": result["latest_signal_date"],
                "cash_after_actions": result["cash_after_actions"],
                "notes": result["notes"],
                "actions": result["actions"].to_dict("records"),
                "position_status": result["position_status"].to_dict("records"),
            }
            print(json.dumps(payload, indent=2, default=str))
        else:
            if result["actions"].empty:
                print("No buy/sell adjustments are needed for the next session.")
            else:
                print("Actions:")
                print(result["actions"].to_string(index=False))
            if not result["position_status"].empty:
                print("\nCurrent position status:")
                print(result["position_status"].to_string(index=False))
            if result["notes"]:
                print("\nNotes:")
                for note in result["notes"]:
                    print(f"- {note}")
        return


if __name__ == "__main__":
    main()
