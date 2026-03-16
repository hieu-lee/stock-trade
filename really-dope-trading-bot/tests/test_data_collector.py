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

    def test_normalize_frame_drops_nonpositive_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            collector = MarketDataCollector(config=config)
            frame = pd.DataFrame(
                [
                    {"date": "2018-07-20", "open": 3.63, "high": 3.69, "low": 3.55, "close": 3.69, "volume": 1000},
                    {"date": "2018-07-23", "open": 3.73, "high": 3.85, "low": 3.73, "close": 0.0, "volume": 900},
                ]
            )

            normalized = collector._normalize_frame(frame, symbol="AAA")

            self.assertEqual(len(normalized), 1)
            self.assertEqual(pd.Timestamp(normalized.iloc[0]["date"]), pd.Timestamp("2018-07-20"))

    def test_normalize_frame_repairs_internal_thousand_fold_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            collector = MarketDataCollector(config=config)
            frame = pd.DataFrame(
                [
                    {"date": "2026-02-21", "open": 23.50, "high": 24.00, "low": 23.10, "close": 23.85, "volume": 1000},
                    {"date": "2026-02-23", "open": 24000.0, "high": 24100.0, "low": 23900.0, "close": 24050.0, "volume": 1200},
                    {"date": "2026-02-24", "open": 24100.0, "high": 24300.0, "low": 24000.0, "close": 24250.0, "volume": 1300},
                ]
            )

            normalized = collector._normalize_frame(frame, symbol="AAA")

            self.assertEqual(len(normalized), 3)
            self.assertAlmostEqual(float(normalized.iloc[0]["close"]), 23850.0)
            self.assertAlmostEqual(float(normalized.iloc[1]["close"]), 24050.0)
            self.assertAlmostEqual(float(normalized.iloc[2]["close"]), 24250.0)

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

    def test_collect_external_markets_refreshes_only_missing_days(self) -> None:
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
                            "open": 101.0,
                            "high": 102.0,
                            "low": 100.0,
                            "close": 101.5,
                            "volume": 1000,
                            "source": "YF",
                        }
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(
                project_dir=project_dir,
                symbols=("AAA",),
                benchmark_symbols=("VNINDEX",),
                external_symbols=("SPY",),
                end_date="2024-01-31",
            )
            adapter = RecordingAdapter()
            collector = MarketDataCollector(config=config, external_adapter=adapter)  # type: ignore[arg-type]
            cached = pd.DataFrame(
                [
                    {
                        "symbol": "SPY",
                        "date": "2024-01-10",
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 900,
                        "source": "YF",
                    }
                ]
            )
            cached.to_parquet(config.external_dir / "SPY.parquet", index=False)

            refreshed = collector.collect_external_markets(
                refresh=True,
                progress_start=0.0,
                progress_end=1.0,
                incremental=True,
            )

            self.assertEqual(adapter.calls[0], ("SPY", "2024-01-11", "2024-01-31"))
            self.assertEqual(len(refreshed), 2)
            self.assertAlmostEqual(float(refreshed.iloc[-1]["close"]), 101.5)

    def test_collect_flow_history_refreshes_only_missing_days(self) -> None:
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
                            "open": 145000.0,
                            "high": 146000.0,
                            "low": 144500.0,
                            "close": 145500.0,
                            "volume": 1200,
                            "source": "FIREANT",
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
            collector = MarketDataCollector(config=config, flow_adapter=adapter)  # type: ignore[arg-type]
            cached = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2024-01-10",
                        "open": 144000.0,
                        "high": 145000.0,
                        "low": 143000.0,
                        "close": 144500.0,
                        "volume": 1000,
                        "source": "FIREANT",
                    }
                ]
            )
            cached.to_parquet(config.flow_dir / "AAA.parquet", index=False)

            refreshed = collector.collect_flow_history(
                refresh=True,
                progress_start=0.0,
                progress_end=1.0,
                incremental=True,
            )

            self.assertEqual(adapter.calls[0], ("AAA", "2024-01-11", "2024-01-31"))
            self.assertEqual(len(refreshed), 2)
            self.assertAlmostEqual(float(refreshed.iloc[-1]["close"]), 145500.0)

    def test_collect_symbol_set_uses_fireant_for_prices_and_vnstock_for_benchmarks(self) -> None:
        class RecordingAdapter:
            def __init__(self, source: str, close: float) -> None:
                self.source = source
                self.close = close
                self.calls: list[tuple[str, str, str]] = []

            def fetch_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
                self.calls.append((symbol, start, end))
                return pd.DataFrame(
                    [
                        {
                            "symbol": symbol,
                            "date": start,
                            "open": self.close,
                            "high": self.close,
                            "low": self.close,
                            "close": self.close,
                            "volume": 1000,
                            "source": self.source,
                        }
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            price_adapter = RecordingAdapter(source="FIREANT", close=145500.0)
            benchmark_adapter = RecordingAdapter(source="VCI", close=1652.79)
            collector = MarketDataCollector(
                config=config,
                adapter=price_adapter,  # type: ignore[arg-type]
                benchmark_adapter=benchmark_adapter,  # type: ignore[arg-type]
            )

            prices = collector.collect_symbol_set(
                symbols=config.symbols,
                target_dir=config.price_dir,
                fallback_kind="prices",
                use_benchmark_adapter=False,
                refresh=False,
                progress_start=0.0,
                progress_end=0.5,
            )
            benchmarks = collector.collect_symbol_set(
                symbols=config.benchmark_symbols,
                target_dir=config.benchmark_dir,
                fallback_kind="benchmarks",
                use_benchmark_adapter=True,
                refresh=False,
                progress_start=0.5,
                progress_end=1.0,
            )

            self.assertEqual(price_adapter.calls[0], ("AAA", config.start_date, config.end_date))
            self.assertEqual(benchmark_adapter.calls[0], ("VNINDEX", config.start_date, config.end_date))
            self.assertAlmostEqual(float(prices.iloc[0]["close"]), 145500.0)
            self.assertAlmostEqual(float(benchmarks.iloc[0]["close"]), 1652.79)

    def test_merge_histories_rescales_thousand_fold_price_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            collector = MarketDataCollector(config=config)
            cached = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2024-01-10",
                        "open": 144.0,
                        "high": 145.0,
                        "low": 143.0,
                        "close": 144.0,
                        "volume": 1000,
                        "source": "CACHED",
                    }
                ]
            )
            recent = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2024-01-11",
                        "open": 145000.0,
                        "high": 146000.0,
                        "low": 144000.0,
                        "close": 145500.0,
                        "volume": 1200,
                        "source": "FLOW",
                    }
                ]
            )

            merged = collector._merge_histories([cached, recent])

            self.assertEqual(len(merged), 2)
            self.assertAlmostEqual(float(merged.iloc[0]["open"]), 144000.0)
            self.assertAlmostEqual(float(merged.iloc[-1]["open"]), 145000.0)
            self.assertAlmostEqual(float(merged.iloc[-1]["close"]), 145500.0)

    def test_refresh_prices_from_flow_history_keeps_fireant_full_vnd_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            config = TradingBotConfig(project_dir=project_dir, symbols=("AAA",), benchmark_symbols=("VNINDEX",))
            collector = MarketDataCollector(config=config)
            cached = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2026-03-12",
                        "open": 144.0,
                        "high": 145.0,
                        "low": 143.0,
                        "close": 144.5,
                        "volume": 1000,
                        "source": "VCI",
                    }
                ]
            )
            cached.to_parquet(config.prices_path, index=False)
            flow_history = pd.DataFrame(
                [
                    {
                        "symbol": "AAA",
                        "date": "2026-03-13",
                        "open": 145000.0,
                        "high": 146000.0,
                        "low": 144500.0,
                        "close": 145500.0,
                        "volume": 1200,
                        "source": "FIREANT",
                    }
                ]
            )

            refreshed = collector._refresh_prices_from_flow_history(flow_history)
            symbol_cache = pd.read_parquet(config.price_dir / "AAA.parquet")

            self.assertEqual(len(refreshed), 2)
            self.assertAlmostEqual(float(refreshed.iloc[0]["open"]), 144000.0)
            self.assertAlmostEqual(float(refreshed.iloc[-1]["open"]), 145000.0)
            self.assertAlmostEqual(float(refreshed.iloc[-1]["close"]), 145500.0)
            self.assertEqual(len(symbol_cache), 2)
            self.assertAlmostEqual(float(symbol_cache.iloc[-1]["close"]), 145500.0)


if __name__ == "__main__":
    unittest.main()
