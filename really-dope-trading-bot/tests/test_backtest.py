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

    def test_backtest_applies_commission_and_slippage_in_main_path(self) -> None:
        base_config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            symbols=("AAA",),
            benchmark_symbols=("VNINDEX",),
            commission_bps=0.0,
            slippage_bps=0.0,
            buy_transaction_fee_bps=0.0,
            sell_transaction_fee_bps=0.0,
            max_positions=1,
            max_weight=1.0,
            optimizer_cash_floor=0.0,
            buy_score_threshold=0.0,
            add_score_threshold=0.0,
            exit_score_threshold=0.0,
            trim_score_threshold=0.0,
            min_trade_weight_delta=0.0,
        )
        costly_config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            symbols=("AAA",),
            benchmark_symbols=("VNINDEX",),
            commission_bps=10.0,
            slippage_bps=15.0,
            buy_transaction_fee_bps=3.0,
            sell_transaction_fee_bps=13.0,
            max_positions=1,
            max_weight=1.0,
            optimizer_cash_floor=0.0,
            buy_score_threshold=0.0,
            add_score_threshold=0.0,
            exit_score_threshold=0.0,
            trim_score_threshold=0.0,
            min_trade_weight_delta=0.0,
        )
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-09"),
                    "symbol": "AAA",
                    "open": 10.0,
                    "close": 10.0,
                    "utility_score": 0.95,
                    "risk_probability": 0.10,
                    "regime_probability": 0.70,
                    "alpha_prediction": 0.90,
                },
                {
                    "date": pd.Timestamp("2026-03-10"),
                    "symbol": "AAA",
                    "open": 10.5,
                    "close": 11.0,
                    "utility_score": 0.95,
                    "risk_probability": 0.10,
                    "regime_probability": 0.70,
                    "alpha_prediction": 0.90,
                },
                {
                    "date": pd.Timestamp("2026-03-11"),
                    "symbol": "AAA",
                    "open": 11.0,
                    "close": 11.5,
                    "utility_score": 0.20,
                    "risk_probability": 0.60,
                    "regime_probability": 0.40,
                    "alpha_prediction": 0.10,
                },
                {
                    "date": pd.Timestamp("2026-03-12"),
                    "symbol": "AAA",
                    "open": 11.3,
                    "close": 11.3,
                    "utility_score": 0.10,
                    "risk_probability": 0.80,
                    "regime_probability": 0.35,
                    "alpha_prediction": 0.05,
                },
            ]
        )

        no_cost_result = run_backtest(panel, config=base_config, initial_cash=100_000.0)
        costly_result = run_backtest(panel, config=costly_config, initial_cash=100_000.0)

        self.assertGreater(no_cost_result.metrics["total_return"], costly_result.metrics["total_return"])
        self.assertGreater(float(costly_result.actions["fees"].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
