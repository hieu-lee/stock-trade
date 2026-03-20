from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.features.panel import build_feature_panel


class FeaturePanelTests(unittest.TestCase):
    def test_build_feature_panel_creates_asof_targets(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("AAA",), benchmark_symbols=("VNINDEX",))
        dates = pd.bdate_range("2022-01-03", periods=280)
        prices = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": np.linspace(10, 20, len(dates)),
                "high": np.linspace(10.5, 20.5, len(dates)),
                "low": np.linspace(9.5, 19.5, len(dates)),
                "close": np.linspace(10.1, 20.1, len(dates)),
                "volume": np.linspace(1000, 2000, len(dates)),
                "source": "TEST",
            }
        )
        benchmarks = pd.DataFrame(
            {
                "symbol": "VNINDEX",
                "date": dates,
                "open": np.linspace(100, 120, len(dates)),
                "high": np.linspace(100.5, 120.5, len(dates)),
                "low": np.linspace(99.5, 119.5, len(dates)),
                "close": np.linspace(100.2, 120.2, len(dates)),
                "volume": np.linspace(10000, 20000, len(dates)),
                "source": "TEST",
            }
        )

        panel = build_feature_panel(prices=prices, benchmarks=benchmarks, config=config)
        row = panel.iloc[-25]
        next_row = panel.iloc[-24]

        self.assertAlmostEqual(float(row["entry_price_next_open"]), float(next_row["open"]))
        self.assertAlmostEqual(
            float(row["forward_excess_return_20d"]),
            float(row["forward_return_20d"] - row["benchmark_forward_return_20d"]),
        )
        self.assertTrue(bool(panel.iloc[-1]["history_ready"]))

    def test_build_feature_panel_attaches_flow_features(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("AAA",), benchmark_symbols=("VNINDEX",))
        dates = pd.bdate_range("2022-01-03", periods=60)
        prices = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": np.linspace(10, 15, len(dates)),
                "high": np.linspace(10.5, 15.5, len(dates)),
                "low": np.linspace(9.5, 14.5, len(dates)),
                "close": np.linspace(10.1, 15.1, len(dates)),
                "volume": np.linspace(1000, 3000, len(dates)),
                "source": "TEST",
            }
        )
        benchmarks = pd.DataFrame(
            {
                "symbol": "VNINDEX",
                "date": dates,
                "open": np.linspace(100, 110, len(dates)),
                "high": np.linspace(100.5, 110.5, len(dates)),
                "low": np.linspace(99.5, 109.5, len(dates)),
                "close": np.linspace(100.2, 110.2, len(dates)),
                "volume": np.linspace(10000, 20000, len(dates)),
                "source": "TEST",
            }
        )
        flow_history = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "foreign_buy_qty": np.linspace(100, 300, len(dates)),
                "foreign_sell_qty": np.linspace(50, 100, len(dates)),
                "foreign_buy_value": np.linspace(1_000_000, 3_000_000, len(dates)),
                "foreign_sell_value": np.linspace(400_000, 1_200_000, len(dates)),
                "foreign_room": np.linspace(1_000_000, 900_000, len(dates)),
                "buy_order_count": np.linspace(20, 60, len(dates)),
                "sell_order_count": np.linspace(10, 40, len(dates)),
                "buy_order_qty": np.linspace(300, 600, len(dates)),
                "sell_order_qty": np.linspace(100, 500, len(dates)),
                "deal_volume": np.linspace(1000, 3000, len(dates)),
                "putthrough_volume": np.linspace(0, 300, len(dates)),
                "putthrough_value": np.linspace(0, 300_000, len(dates)),
                "market_cap": np.linspace(10_000_000, 20_000_000, len(dates)),
                "pe_daily": np.linspace(8, 12, len(dates)),
                "pb_daily": np.linspace(1.2, 2.0, len(dates)),
                "ps_daily": np.linspace(1.0, 1.5, len(dates)),
                "shares_outstanding": np.linspace(1_000_000, 1_000_000, len(dates)),
                "source": "FIREANT",
            }
        )

        panel = build_feature_panel(prices=prices, benchmarks=benchmarks, flow_history=flow_history, config=config)
        row = panel.iloc[-1]
        self.assertGreater(float(row["foreign_net_qty"]), 0.0)
        self.assertGreater(float(row["foreign_flow_score"]), 0.0)
        self.assertTrue(pd.notna(row["order_pressure_score"]))

    def test_build_feature_panel_attaches_event_features(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("AAA",), benchmark_symbols=("VNINDEX",))
        dates = pd.bdate_range("2022-01-03", periods=60)
        prices = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": np.linspace(10, 15, len(dates)),
                "high": np.linspace(10.5, 15.5, len(dates)),
                "low": np.linspace(9.5, 14.5, len(dates)),
                "close": np.linspace(10.1, 15.1, len(dates)),
                "volume": np.linspace(1000, 3000, len(dates)),
                "source": "TEST",
            }
        )
        benchmarks = pd.DataFrame(
            {
                "symbol": "VNINDEX",
                "date": dates,
                "open": np.linspace(100, 110, len(dates)),
                "high": np.linspace(100.5, 110.5, len(dates)),
                "low": np.linspace(99.5, 109.5, len(dates)),
                "close": np.linspace(100.2, 110.2, len(dates)),
                "volume": np.linspace(10000, 20000, len(dates)),
                "source": "TEST",
            }
        )
        event_history = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "available_date": pd.Timestamp("2022-02-10"),
                    "issue_date": pd.Timestamp("2022-02-11"),
                    "record_date": pd.Timestamp("2022-02-18"),
                    "exright_date": pd.Timestamp("2022-02-17"),
                    "event_code": "DIV",
                    "event_name": "Cash dividend",
                    "ratio": np.nan,
                    "value": 1000.0,
                    "source": "VCI",
                }
            ]
        )

        panel = build_feature_panel(prices=prices, benchmarks=benchmarks, event_history=event_history, config=config)
        row = panel.loc[panel["date"] == pd.Timestamp("2022-02-15")].iloc[0]
        self.assertGreater(float(row["recent_event_count_20d"]), 0.0)
        self.assertGreater(float(row["latest_dividend_value"]), 0.0)
        self.assertGreater(float(row["upcoming_record_days"]), 0.0)

    def test_build_feature_panel_creates_anchor_conditioned_regime_participation_target(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1], symbols=("AAA",), benchmark_symbols=("VNINDEX",))
        dates = pd.bdate_range("2022-01-03", periods=280)
        prices = pd.DataFrame(
            {
                "symbol": "AAA",
                "date": dates,
                "open": np.linspace(10, 40, len(dates)),
                "high": np.linspace(10.4, 40.4, len(dates)),
                "low": np.linspace(9.8, 39.8, len(dates)),
                "close": np.linspace(10.2, 40.2, len(dates)),
                "volume": np.linspace(1000, 2000, len(dates)),
                "source": "TEST",
            }
        )
        benchmarks = pd.DataFrame(
            {
                "symbol": "VNINDEX",
                "date": dates,
                "open": np.linspace(100, 180, len(dates)),
                "high": np.linspace(100.3, 180.3, len(dates)),
                "low": np.linspace(99.8, 179.8, len(dates)),
                "close": np.linspace(100.1, 180.1, len(dates)),
                "volume": np.linspace(10000, 20000, len(dates)),
                "source": "TEST",
            }
        )

        panel = build_feature_panel(prices=prices, benchmarks=benchmarks, config=config)
        participation_rows = panel.loc[panel["target_regime_participation"].notna()].copy()

        self.assertIn("target_regime_participation", panel.columns)
        self.assertFalse(participation_rows.empty)
        self.assertTrue((participation_rows["target_regime"] == 1.0).all())
        self.assertTrue((participation_rows["target_regime_participation"] == 1.0).any())


if __name__ == "__main__":
    unittest.main()
