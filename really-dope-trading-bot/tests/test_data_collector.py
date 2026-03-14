from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.data.collector import MarketDataCollector


class DataCollectorTests(unittest.TestCase):
    def test_manual_history_without_symbol_column_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            manual_dir = config.manual_import_dir / "prices"
            manual_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"date": "2006-01-02", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
                    {"date": "2006-01-03", "open": 10.5, "high": 11.2, "low": 10.2, "close": 11.0, "volume": 1100},
                ]
            ).to_csv(manual_dir / "AAA.csv", index=False)

            collector = MarketDataCollector(config=config)
            frame = collector._load_manual_history(symbol="AAA", kind="prices")

            self.assertEqual(list(frame["symbol"].unique()), ["AAA"])
            self.assertEqual(frame["source"].iloc[0], "MANUAL")
            self.assertEqual(len(frame), 2)

    def test_incremental_refresh_starts_after_latest_cached_day(self) -> None:
        class RecordingAdapter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, str]] = []

            def fetch_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
                self.calls.append((symbol, start, end))
                return pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "date": start,
                            "open": 12.0,
                            "high": 13.0,
                            "low": 11.0,
                            "close": 12.5,
                            "volume": 1200,
                            "source": "TEST",
                        }
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(
                project_dir=project_dir,
                symbols=("AAA",),
                benchmark_symbols=("VNINDEX",),
                end_date="2024-01-31",
            )
            adapter = RecordingAdapter()
            collector = MarketDataCollector(config=config, adapter=adapter)  # type: ignore[arg-type]
            cached = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2024-01-10",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000,
                        "source": "TEST",
                    }
                ]
            )

            collector._refresh_history(symbol="AAA", cached=cached)

            self.assertEqual(adapter.calls[0], ("AAA", "2024-01-11", "2024-01-31"))


if __name__ == "__main__":
    unittest.main()
