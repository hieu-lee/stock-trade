from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rdtb.api.app import create_app
from rdtb.config import get_default_config
from rdtb.service.pipeline import generate_daily_decisions, train_system


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train the Really Dope Trading Bot.")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--project-dir", default=None)
    args = parser.parse_args()
    config = get_default_config(args.project_dir)
    summary = train_system(config=config, refresh_data=args.refresh_data)
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
