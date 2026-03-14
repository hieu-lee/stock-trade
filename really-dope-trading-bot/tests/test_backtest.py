from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from rdtb.backtest.engine import run_backtest
from rdtb.config import TradingBotConfig


class BacktestTests(unittest.TestCase):
    def test_backtest_executes_next_open_and_grows_equity(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("AAA",), benchmark_symbols=("VNINDEX",))
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-09"),
                    "symbol": "AAA",
                    "open": 10.0,
                    "close": 10.0,
                    "utility_score": 0.90,
                    "risk_probability": 0.10,
                    "regime_probability": 0.65,
                    "alpha_prediction": 0.08,
                },
                {
                    "date": pd.Timestamp("2026-03-10"),
                    "symbol": "AAA",
                    "open": 10.0,
                    "close": 11.0,
                    "utility_score": 0.92,
                    "risk_probability": 0.10,
                    "regime_probability": 0.66,
                    "alpha_prediction": 0.09,
                },
                {
                    "date": pd.Timestamp("2026-03-11"),
                    "symbol": "AAA",
                    "open": 11.0,
                    "close": 12.0,
                    "utility_score": 0.60,
                    "risk_probability": 0.15,
                    "regime_probability": 0.62,
                    "alpha_prediction": 0.07,
                },
            ]
        )

        result = run_backtest(panel, config=config, initial_cash=100_000.0)

        self.assertGreater(result.metrics["total_return"], 0.0)
        self.assertGreaterEqual(result.metrics["trade_count"], 1)
        self.assertGreater(result.equity_curve["equity"].iloc[-1], result.equity_curve["equity"].iloc[0])

    def test_backtest_respects_settlement_delays(self) -> None:
        config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            symbols=("AAA", "BBB"),
            benchmark_symbols=("VNINDEX",),
            max_positions=1,
            max_weight=1.0,
            optimizer_cash_floor=0.0,
        )
        dates = pd.bdate_range("2026-03-09", periods=9)
        rows: list[dict] = []
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "symbol": "AAA",
                    "open": 10.0,
                    "close": 10.0,
                    "utility_score": 0.95 if date == dates[0] else 0.10,
                    "risk_probability": 0.10,
                    "regime_probability": 0.70,
                    "alpha_prediction": 0.90 if date == dates[0] else 0.10,
                }
            )
            rows.append(
                {
                    "date": date,
                    "symbol": "BBB",
                    "open": 10.0,
                    "close": 10.0,
                    "utility_score": 0.10 if date == dates[0] else 0.95,
                    "risk_probability": 0.10,
                    "regime_probability": 0.70,
                    "alpha_prediction": 0.10 if date == dates[0] else 0.90,
                }
            )
        panel = pd.DataFrame(rows)

        result = run_backtest(panel, config=config, initial_cash=100_000.0)
        actions = result.actions.copy()
        first_buy_a = actions.loc[(actions["symbol"] == "AAA") & (actions["action"] == "BUY"), "execution_date"].min()
        first_sell_a = actions.loc[(actions["symbol"] == "AAA") & (actions["action"].isin(["EXIT", "TRIM"])), "execution_date"].min()
        first_buy_b = actions.loc[(actions["symbol"] == "BBB") & (actions["action"] == "BUY"), "execution_date"].min()

        self.assertEqual(pd.Timestamp(first_buy_a), pd.Timestamp("2026-03-10"))
        self.assertEqual(pd.Timestamp(first_sell_a), pd.Timestamp("2026-03-16"))
        self.assertEqual(pd.Timestamp(first_buy_b), pd.Timestamp("2026-03-19"))


if __name__ == "__main__":
    unittest.main()
