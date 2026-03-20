from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rdtb.api.app import create_app
from rdtb.config import get_default_config
from rdtb.service.pipeline import generate_daily_decisions, search_constants_system, train_system, validate_system


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train the Really Dope Trading Bot.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--search-trials", type=int, default=None)
    parser.add_argument("--search-timeout-seconds", type=int, default=None)
    args = parser.parse_args()
    config = get_default_config(args.project_dir)
    summary = train_system(
        config=config,
        refresh_data=args.refresh_data,
        search_trials=args.search_trials,
        search_timeout_seconds=args.search_timeout_seconds,
    )
    print(summary)


def decide_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate daily decisions.")
    parser.add_argument("--cash", type=float, default=1_000_000_000.0)
    parser.add_argument("--holdings-csv", default=None)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--project-dir", default=None)
    args = parser.parse_args()
    config = get_default_config(args.project_dir)
    holdings = pd.read_csv(args.holdings_csv) if args.holdings_csv else pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "buy_date"])
    decisions = generate_daily_decisions(
        cash=args.cash,
        holdings=holdings,
        config=config,
        refresh_data=args.refresh_data,
    )
    print(decisions)


def validate_cli() -> None:
    parser = argparse.ArgumentParser(description="Run deeper validation for the Really Dope Trading Bot.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--project-dir", default=None)
    args = parser.parse_args()
    config = get_default_config(args.project_dir)
    report = validate_system(config=config, refresh_data=args.refresh_data)
    print(report)


def search_constants_cli() -> None:
    parser = argparse.ArgumentParser(description="Run staged auto-search with final-test prioritization.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--target-summary-path", default=None)
    args = parser.parse_args()
    config = get_default_config(args.project_dir)
    target_summary = None
    if args.target_summary_path:
        target_summary = json.loads(Path(args.target_summary_path).read_text(encoding="utf-8"))
    report = search_constants_system(
        config=config,
        refresh_data=args.refresh_data,
        trials=args.trials,
        timeout_seconds=args.timeout_seconds,
        target_summary=target_summary,
    )
    print(report)


def ui_cli() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(project_dir / "src" / "rdtb" / "ui" / "app.py")], check=True)


def api_cli() -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("`uvicorn` is required to run the API server.") from exc
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
