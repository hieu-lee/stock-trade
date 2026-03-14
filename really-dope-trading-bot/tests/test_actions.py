from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.portfolio.actions import build_daily_actions


def _config() -> TradingBotConfig:
    return TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("VCB", "FPT", "HPG"))


def _snapshot(rows: list[dict], date: str = "2026-03-11") -> pd.DataFrame:
    base = {
        "date": pd.Timestamp(date),
        "close": 10.0,
        "utility_score": 0.60,
        "risk_probability": 0.20,
        "regime_probability": 0.65,
        "alpha_prediction": 0.08,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


class ActionTests(unittest.TestCase):
    def test_buy_add_trim_and_exit_semantics(self) -> None:
        config = _config()

        trim_bundle = build_daily_actions(
            scored_snapshot=_snapshot([{"symbol": "VCB", "close": 10.0, "utility_score": 0.50}]),
            holdings=pd.DataFrame([{"symbol": "VCB", "quantity": 10_000, "avg_cost": 8.0, "buy_date": "2026-01-01"}]),
            cash=0.0,
            config=config,
        )
        self.assertEqual(trim_bundle.actions.iloc[0]["action"], "TRIM")

        add_bundle = build_daily_actions(
            scored_snapshot=_snapshot([{"symbol": "VCB", "close": 10.0, "utility_score": 0.95, "risk_probability": 0.10}]),
            holdings=pd.DataFrame([{"symbol": "VCB", "quantity": 1_000, "avg_cost": 10.0, "buy_date": "2026-02-10"}]),
            cash=1_000_000.0,
            config=config,
        )
        self.assertEqual(add_bundle.actions.iloc[0]["action"], "ADD")

        rotate_bundle = build_daily_actions(
            scored_snapshot=_snapshot(
                [
                    {"symbol": "VCB", "close": 10.0, "utility_score": 0.10, "risk_probability": 0.70},
                    {"symbol": "FPT", "close": 20.0, "utility_score": 0.92, "risk_probability": 0.10},
                    {"symbol": "HPG", "close": 15.0, "utility_score": 0.88, "risk_probability": 0.12},
                ]
            ),
            holdings=pd.DataFrame([{"symbol": "VCB", "quantity": 2_000, "avg_cost": 10.0, "buy_date": "2026-02-03"}]),
            cash=1_000_000.0,
            config=config,
        )
        self.assertIn("VCB", set(rotate_bundle.actions["symbol"]))
        self.assertIn("FPT", set(rotate_bundle.actions["symbol"]))
        self.assertEqual(len(rotate_bundle.actions["symbol"]), len(set(rotate_bundle.actions["symbol"])))


if __name__ == "__main__":
    unittest.main()
