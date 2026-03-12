from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from trade60_app.backtest.engine import StrategyParameters, run_daily_backtest
from trade60_app.config import Trade60Config
from trade60_app.service.pipeline import _build_actions, _normalize_holdings


def _make_config() -> Trade60Config:
    return Trade60Config(
        workspace_dir=Path(__file__).resolve().parents[1],
        symbols=("VCB", "FPT", "HPG"),
    )


def _make_scored(rows: list[dict], date: str = "2026-03-11") -> pd.DataFrame:
    base = {
        "date": pd.Timestamp(date),
        "close": 10.0,
        "alpha_probability": 0.6,
        "regime_probability": 0.6,
        "composite_score": 0.6,
        "relative_strength_20d": 0.1,
        "distance_ma50": 0.1,
        "breadth_above_ma50": 0.6,
        "volume_zscore_20d": 0.1,
        "benchmark_distance_ma200": 0.05,
        "breadth_above_ma200": 0.55,
        "benchmark_ret_20d": 0.05,
    }
    payload = [{**base, **row} for row in rows]
    return pd.DataFrame(payload)


def _make_calendar(end_date: str = "2026-03-11") -> pd.Series:
    return pd.Series(pd.bdate_range(end=end_date, periods=60))


class Trade60RebalanceTests(unittest.TestCase):
    def test_partial_sell_existing_position(self) -> None:
        config = _make_config()
        scored = _make_scored(
            [
                {
                    "symbol": "VCB",
                    "close": 60.5,
                    "alpha_probability": 0.50,
                    "composite_score": 0.52,
                }
            ]
        )
        holdings = pd.DataFrame(
            [
                {
                    "symbol": "VCB",
                    "quantity": 200,
                    "avg_cost": 61.7,
                    "buy_date": "2026-02-03",
                    "buy_date_defaulted": False,
                }
            ]
        )

        actions, position_status, _, _ = _build_actions(scored, holdings, 100_000.0, {}, config, _make_calendar())

        sell_row = actions.loc[actions["symbol"] == "VCB"].iloc[0]
        status_row = position_status.loc[position_status["symbol"] == "VCB"].iloc[0]
        self.assertEqual(sell_row["action"], "SELL")
        self.assertEqual(int(sell_row["quantity"]), 100)
        self.assertEqual(status_row["status"], "TRIM")
        self.assertEqual(int(status_row["next_quantity"]), 100)

    def test_top_up_existing_position(self) -> None:
        config = _make_config()
        scored = _make_scored(
            [
                {
                    "symbol": "VCB",
                    "close": 10.0,
                    "alpha_probability": 0.80,
                    "composite_score": 0.82,
                }
            ]
        )
        holdings = pd.DataFrame(
            [
                {
                    "symbol": "VCB",
                    "quantity": 100,
                    "avg_cost": 9.5,
                    "buy_date": "2026-02-03",
                    "buy_date_defaulted": False,
                }
            ]
        )

        actions, position_status, _, _ = _build_actions(scored, holdings, 1_000.0, {}, config, _make_calendar())

        buy_row = actions.loc[actions["symbol"] == "VCB"].iloc[0]
        status_row = position_status.loc[position_status["symbol"] == "VCB"].iloc[0]
        self.assertEqual(buy_row["action"], "BUY")
        self.assertGreater(int(buy_row["quantity"]), 0)
        self.assertEqual(status_row["status"], "TOP_UP")
        self.assertGreater(int(status_row["next_quantity"]), int(status_row["current_quantity"]))

    def test_mixed_symbol_sell_and_buy(self) -> None:
        config = _make_config()
        scored = _make_scored(
            [
                {
                    "symbol": "VCB",
                    "close": 60.5,
                    "alpha_probability": 0.50,
                    "composite_score": 0.52,
                },
                {
                    "symbol": "FPT",
                    "close": 20.0,
                    "alpha_probability": 0.82,
                    "composite_score": 0.84,
                },
            ]
        )
        holdings = pd.DataFrame(
            [
                {
                    "symbol": "VCB",
                    "quantity": 200,
                    "avg_cost": 61.7,
                    "buy_date": "2026-02-03",
                    "buy_date_defaulted": False,
                }
            ]
        )

        actions, _, _, _ = _build_actions(scored, holdings, 5_000.0, {}, config, _make_calendar())

        self.assertSetEqual(set(actions["symbol"]), {"VCB", "FPT"})
        self.assertEqual(actions.loc[actions["symbol"] == "VCB", "action"].iloc[0], "SELL")
        self.assertEqual(actions.loc[actions["symbol"] == "FPT", "action"].iloc[0], "BUY")

    def test_no_same_day_rebuy_for_trimmed_symbol(self) -> None:
        config = _make_config()
        scored = _make_scored(
            [
                {
                    "symbol": "VCB",
                    "close": 60.5,
                    "alpha_probability": 0.50,
                    "composite_score": 0.90,
                },
                {
                    "symbol": "FPT",
                    "close": 20.0,
                    "alpha_probability": 0.82,
                    "composite_score": 0.84,
                },
            ]
        )
        holdings = pd.DataFrame(
            [
                {
                    "symbol": "VCB",
                    "quantity": 200,
                    "avg_cost": 61.7,
                    "buy_date": "2026-02-03",
                    "buy_date_defaulted": False,
                }
            ]
        )

        actions, _, _, _ = _build_actions(scored, holdings, 5_000.0, {}, config, _make_calendar())

        buy_symbols = set(actions.loc[actions["action"] == "BUY", "symbol"])
        self.assertNotIn("VCB", buy_symbols)

    def test_blank_buy_date_defaults_to_one_month_ago(self) -> None:
        config = _make_config()
        holdings = pd.DataFrame(
            [
                {
                    "symbol": "VCB",
                    "quantity": 200,
                    "avg_cost": 61.7,
                    "buy_date": "",
                }
            ]
        )

        normalized, notes = _normalize_holdings(holdings, config, pd.Timestamp("2026-03-11"))

        self.assertFalse(notes)
        self.assertEqual(normalized.loc[0, "buy_date"], "2026-02-11")
        self.assertTrue(bool(normalized.loc[0, "buy_date_defaulted"]))

    def test_backtest_supports_partial_trim_and_rotation(self) -> None:
        config = _make_config()
        panel = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-09"),
                    "symbol": "VCB",
                    "open": 10.0,
                    "close": 10.0,
                    "alpha_probability": 0.82,
                    "regime_probability": 0.60,
                    "composite_score": 0.82,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
                {
                    "date": pd.Timestamp("2026-03-09"),
                    "symbol": "FPT",
                    "open": 20.0,
                    "close": 20.0,
                    "alpha_probability": 0.40,
                    "regime_probability": 0.60,
                    "composite_score": 0.40,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
                {
                    "date": pd.Timestamp("2026-03-10"),
                    "symbol": "VCB",
                    "open": 10.0,
                    "close": 10.5,
                    "alpha_probability": 0.50,
                    "regime_probability": 0.60,
                    "composite_score": 0.50,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
                {
                    "date": pd.Timestamp("2026-03-10"),
                    "symbol": "FPT",
                    "open": 20.0,
                    "close": 20.0,
                    "alpha_probability": 0.82,
                    "regime_probability": 0.60,
                    "composite_score": 0.82,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
                {
                    "date": pd.Timestamp("2026-03-11"),
                    "symbol": "VCB",
                    "open": 10.4,
                    "close": 10.3,
                    "alpha_probability": 0.50,
                    "regime_probability": 0.60,
                    "composite_score": 0.50,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
                {
                    "date": pd.Timestamp("2026-03-11"),
                    "symbol": "FPT",
                    "open": 20.2,
                    "close": 20.7,
                    "alpha_probability": 0.82,
                    "regime_probability": 0.60,
                    "composite_score": 0.82,
                    "benchmark_ret_1d": 0.0,
                    "benchmark_distance_ma200": 0.05,
                    "breadth_above_ma200": 0.55,
                    "benchmark_ret_20d": 0.05,
                },
            ]
        )
        params = StrategyParameters(
            entry_threshold=0.58,
            entry_quantile=0.90,
            exit_threshold=0.46,
            regime_threshold=0.53,
            max_positions=3,
            max_holding_days=40,
            min_holding_days=1,
            stop_loss_pct=0.05,
            take_profit_pct=0.18,
            hold_alpha_buffer=0.06,
            rank_keep_fraction=1.0,
            defensive_trim_fraction=0.35,
            weak_alpha_trim_fraction=0.5,
            profit_trim_fraction=0.5,
        )

        backtest = run_daily_backtest(panel, config, params)

        self.assertIn("weak_alpha_trim", set(backtest.trades["exit_reason"]))
        self.assertIn("FPT", set(backtest.trades["symbol"]))


if __name__ == "__main__":
    unittest.main()
