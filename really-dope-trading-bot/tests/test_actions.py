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
        "regime_anchor_probability": 0.65,
        "regime_participation_probability": 1.0,
        "regime_probability": 0.65,
        "alpha_prediction": 0.08,
        "atr_pct": 0.02,
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

    def test_risk_spike_exits_existing_position(self) -> None:
        config = _config()

        bundle = build_daily_actions(
            scored_snapshot=_snapshot([{"symbol": "VCB", "risk_probability": 0.90, "regime_probability": 0.40, "utility_score": 0.52}]),
            holdings=pd.DataFrame([{"symbol": "VCB", "quantity": 1_000, "avg_cost": 10.0, "buy_date": "2026-02-10"}]),
            cash=500_000.0,
            config=config,
        )

        self.assertEqual(bundle.actions.iloc[0]["action"], "EXIT")

    def test_defensive_regime_can_stay_in_cash(self) -> None:
        config = _config()

        bundle = build_daily_actions(
            scored_snapshot=_snapshot(
                [
                    {"symbol": "VCB", "utility_score": 0.40, "risk_probability": 0.70, "regime_anchor_probability": 0.35, "regime_probability": 0.35},
                    {"symbol": "FPT", "utility_score": 0.42, "risk_probability": 0.66, "regime_anchor_probability": 0.35, "regime_probability": 0.35},
                    {"symbol": "HPG", "utility_score": 0.38, "risk_probability": 0.72, "regime_anchor_probability": 0.35, "regime_probability": 0.35},
                ]
            ),
            holdings=pd.DataFrame(),
            cash=1_000_000.0,
            config=config,
        )

        self.assertTrue(bundle.actions.empty)
        self.assertAlmostEqual(float(bundle.target_weights["target_weight"].sum()), 0.0)

    def test_symbol_regime_exposure_is_order_invariant(self) -> None:
        config = _config()
        rows = [
            {"symbol": "VCB", "utility_score": 0.72, "risk_probability": 0.10, "regime_anchor_probability": 0.55, "regime_probability": 0.25},
            {"symbol": "FPT", "utility_score": 0.72, "risk_probability": 0.10, "regime_anchor_probability": 0.55, "regime_probability": 0.85},
        ]

        first_bundle = build_daily_actions(
            scored_snapshot=_snapshot(rows),
            holdings=pd.DataFrame(),
            cash=1_000_000.0,
            config=config,
        )
        second_bundle = build_daily_actions(
            scored_snapshot=_snapshot(list(reversed(rows))),
            holdings=pd.DataFrame(),
            cash=1_000_000.0,
            config=config,
        )

        self.assertGreater(float(first_bundle.target_weights["target_weight"].sum()), 0.0)
        self.assertAlmostEqual(
            float(first_bundle.target_weights["target_weight"].sum()),
            float(second_bundle.target_weights["target_weight"].sum()),
            places=6,
        )

    def test_anchor_regime_controls_exposure_while_combined_regime_controls_ranking(self) -> None:
        config = _config()

        defensive_bundle = build_daily_actions(
            scored_snapshot=_snapshot(
                [
                    {
                        "symbol": "VCB",
                        "utility_score": 0.92,
                        "risk_probability": 0.10,
                        "regime_anchor_probability": 0.25,
                        "regime_participation_probability": 1.0,
                        "regime_probability": 0.92,
                    },
                    {
                        "symbol": "FPT",
                        "utility_score": 0.91,
                        "risk_probability": 0.10,
                        "regime_anchor_probability": 0.25,
                        "regime_participation_probability": 0.9,
                        "regime_probability": 0.90,
                    },
                ]
            ),
            holdings=pd.DataFrame(),
            cash=1_000_000.0,
            config=config,
        )
        risk_on_bundle = build_daily_actions(
            scored_snapshot=_snapshot(
                [
                    {
                        "symbol": "VCB",
                        "utility_score": 0.92,
                        "risk_probability": 0.10,
                        "regime_anchor_probability": 0.80,
                        "regime_participation_probability": 1.0,
                        "regime_probability": 0.92,
                    },
                    {
                        "symbol": "FPT",
                        "utility_score": 0.91,
                        "risk_probability": 0.10,
                        "regime_anchor_probability": 0.80,
                        "regime_participation_probability": 0.9,
                        "regime_probability": 0.90,
                    },
                ]
            ),
            holdings=pd.DataFrame(),
            cash=1_000_000.0,
            config=config,
        )

        self.assertLess(
            float(defensive_bundle.target_weights["target_weight"].sum()),
            float(risk_on_bundle.target_weights["target_weight"].sum()),
        )


if __name__ == "__main__":
    unittest.main()
