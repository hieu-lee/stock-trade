from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.portfolio.transactions import replay_transactions


class TransactionReplayTests(unittest.TestCase):
    def test_replay_transactions_respects_settlement_and_fees(self) -> None:
        config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            symbols=("AAA",),
            benchmark_symbols=("VNINDEX",),
        )
        trading_calendar = pd.bdate_range("2026-03-10", periods=10).tolist()
        transactions = pd.DataFrame(
            [
                {"date": "2026-03-10", "symbol": "AAA", "action": "BUY", "quantity": 1000, "price": 10.0},
                {"date": "2026-03-12", "symbol": "AAA", "action": "SELL", "quantity": 1000, "price": 11.0},
                {"date": "2026-03-16", "symbol": "AAA", "action": "SELL", "quantity": 1000, "price": 11.0},
            ]
        )

        replay = replay_transactions(
            transactions=transactions,
            starting_cash=100_000.0,
            config=config,
            as_of_date=pd.Timestamp("2026-03-17"),
            trading_calendar=trading_calendar,
        )

        processed = replay.processed_transactions
        self.assertEqual(list(processed["status"]), ["EXECUTED", "REJECTED", "EXECUTED"])
        self.assertIn("settled shares were sellable", replay.notes[0])
        self.assertAlmostEqual(replay.available_cash, 100_000.0 - 10_000.0 - 3.0, places=6)
        self.assertAlmostEqual(replay.pending_cash_total, 11_000.0 - 14.3, places=6)
        self.assertEqual(int(replay.holdings["sellable_quantity"].sum()), 0)
        self.assertEqual(int(replay.holdings["quantity"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
