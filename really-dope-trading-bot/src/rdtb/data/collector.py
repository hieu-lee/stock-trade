from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.data.adapters.fireant_history_adapter import FireAntHistoryAdapter
from rdtb.data.adapters.vnstock_adapter import VnstockDailyAdapter
from rdtb.data.adapters.vnstock_fundamentals_adapter import VnstockFundamentalsAdapter
from rdtb.data.adapters.yfinance_adapter import YFinanceDailyAdapter
from rdtb.data.store import DuckDBMarketStore
from rdtb.utils import ProgressCallback, report_progress

PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(slots=True)
class DataBundle:
    universe: pd.DataFrame
    prices: pd.DataFrame
    benchmarks: pd.DataFrame
    listings: pd.DataFrame
    external_markets: pd.DataFrame
    fundamentals: pd.DataFrame
    company_metadata: pd.DataFrame
    flow_history: pd.DataFrame
    event_history: pd.DataFrame


class MarketDataCollector:
    def __init__(
        self,
        config: TradingBotConfig,
        progress_callback: ProgressCallback | None = None,
        adapter: FireAntHistoryAdapter | None = None,
        benchmark_adapter: VnstockDailyAdapter | None = None,
        fundamentals_adapter: VnstockFundamentalsAdapter | None = None,
        external_adapter: YFinanceDailyAdapter | None = None,
        flow_adapter: FireAntHistoryAdapter | None = None,
        store: DuckDBMarketStore | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.adapter = adapter or FireAntHistoryAdapter()
        self.benchmark_adapter = benchmark_adapter or VnstockDailyAdapter(source=config.finance_source)
        self.fundamentals_adapter = fundamentals_adapter or VnstockFundamentalsAdapter(source=config.finance_source)
        self.external_adapter = external_adapter or YFinanceDailyAdapter()
        self.flow_adapter = flow_adapter or FireAntHistoryAdapter()
        self.store = store or DuckDBMarketStore(config)
        self.store.ensure_layout()

    def collect_all(self, refresh: bool = False) -> DataBundle:
        report_progress(self.progress_callback, "Preparing VN60 universe...", 0.0)
        universe = self.collect_universe()
        benchmarks = self.collect_symbol_set(
            symbols=self.config.benchmark_symbols,
            target_dir=self.config.benchmark_dir,
            fallback_kind="benchmarks",
            use_benchmark_adapter=True,
            refresh=refresh,
            progress_start=0.0,
            progress_end=0.08,
        )
        external_markets = self.collect_external_markets(
            refresh=refresh,
            progress_start=0.08,
            progress_end=0.18,
            incremental=refresh,
        )
        fundamentals = self.collect_fundamentals(refresh=refresh, progress_start=0.18, progress_end=0.30)
        company_metadata = self.collect_company_metadata(refresh=refresh, progress_start=0.30, progress_end=0.36)
        flow_history = self.collect_flow_history(
            refresh=refresh,
            progress_start=0.36,
            progress_end=0.44,
            incremental=refresh,
        )
        event_history = self.collect_event_history(refresh=refresh, progress_start=0.44, progress_end=0.50)
        prices = self.collect_symbol_set(
            symbols=self.config.symbols,
            target_dir=self.config.price_dir,
            fallback_kind="prices",
            use_benchmark_adapter=False,
            refresh=refresh,
            progress_start=0.50,
            progress_end=1.00,
        )
        listings = self.build_listing_frame(prices)
        self.store.write_bundle(
            universe=universe,
            prices=prices,
            benchmarks=benchmarks,
            listings=listings,
            external_markets=external_markets,
            fundamentals=fundamentals,
            company_metadata=company_metadata,
            flow_history=flow_history,
            event_history=event_history,
        )
        report_progress(self.progress_callback, "Finished market data refresh.", 1.0)
        return DataBundle(
            universe=universe,
            prices=prices,
            benchmarks=benchmarks,
            listings=listings,
            external_markets=external_markets,
            fundamentals=fundamentals,
            company_metadata=company_metadata,
            flow_history=flow_history,
            event_history=event_history,
        )

    def collect_for_decision(self, refresh: bool = False) -> DataBundle:
        report_progress(self.progress_callback, "Preparing fast market refresh...", 0.0)
        universe = self.collect_universe()
        benchmarks = self.collect_symbol_set(
            symbols=self.config.benchmark_symbols,
            target_dir=self.config.benchmark_dir,
            fallback_kind="benchmarks",
            use_benchmark_adapter=True,
            refresh=refresh,
            progress_start=0.0,
            progress_end=0.10,
        )
        external_markets = self.collect_external_markets(
            refresh=refresh,
            progress_start=0.10,
            progress_end=0.18,
            incremental=refresh,
        )
        fundamentals = self._load_or_collect_fundamentals(progress_start=0.18, progress_end=0.26)
        company_metadata = self._load_or_collect_company_metadata(progress_start=0.26, progress_end=0.30)
        flow_history = self.collect_flow_history(
            refresh=refresh,
            progress_start=0.30,
            progress_end=0.44,
            incremental=refresh,
        )
        event_history = self._load_or_collect_event_history(progress_start=0.44, progress_end=0.48)
        if refresh:
            prices = self._refresh_prices_from_flow_history(flow_history)
            if prices.empty and self.config.prices_path.exists():
                report_progress(
                    self.progress_callback,
                    "Using cached price history because fast incremental price update is unavailable.",
                    0.48,
                )
                prices = pd.read_parquet(self.config.prices_path)
            elif prices.empty:
                prices = self.collect_symbol_set(
                    symbols=self.config.symbols,
                    target_dir=self.config.price_dir,
                    fallback_kind="prices",
                    use_benchmark_adapter=False,
                    refresh=True,
                    progress_start=0.48,
                    progress_end=1.00,
                )
        elif self.config.prices_path.exists():
            prices = pd.read_parquet(self.config.prices_path)
        else:
            prices = self.collect_symbol_set(
                symbols=self.config.symbols,
                target_dir=self.config.price_dir,
                fallback_kind="prices",
                use_benchmark_adapter=False,
                refresh=False,
                progress_start=0.48,
                progress_end=1.00,
            )
        listings = self.build_listing_frame(prices)
        self.store.write_bundle(
            universe=universe,
            prices=prices,
            benchmarks=benchmarks,
            listings=listings,
            external_markets=external_markets,
            fundamentals=fundamentals,
            company_metadata=company_metadata,
            flow_history=flow_history,
            event_history=event_history,
        )
        report_progress(self.progress_callback, "Finished fast market refresh.", 1.0)
        return DataBundle(
            universe=universe,
            prices=prices,
            benchmarks=benchmarks,
            listings=listings,
            external_markets=external_markets,
            fundamentals=fundamentals,
            company_metadata=company_metadata,
            flow_history=flow_history,
            event_history=event_history,
        )

    def collect_universe(self) -> pd.DataFrame:
        frame = pd.DataFrame({"symbol": list(self.config.symbols), "display_name": list(self.config.symbols)})
        frame["benchmark"] = False
        benchmark_frame = pd.DataFrame(
            {"symbol": list(self.config.benchmark_symbols), "display_name": list(self.config.benchmark_symbols), "benchmark": True}
        )
        universe = pd.concat([frame, benchmark_frame], ignore_index=True)
        universe = universe.drop_duplicates(subset=["symbol"]).sort_values(["benchmark", "symbol"]).reset_index(drop=True)
        universe.to_parquet(self.config.universe_path, index=False)
        return universe

    def collect_symbol_set(
        self,
        symbols: tuple[str, ...],
        target_dir: Path,
        fallback_kind: str,
        use_benchmark_adapter: bool,
        refresh: bool,
        progress_start: float,
        progress_end: float,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        total = max(len(symbols), 1)
        for index, symbol in enumerate(symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting {fallback_kind} for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = target_dir / f"{symbol}.parquet"
            manual = self._load_manual_history(symbol=symbol, kind=fallback_kind)
            seed = self._load_seed_history(symbol=symbol, kind=fallback_kind)
            if path.exists() and not refresh:
                remote = pd.read_parquet(path)
            elif path.exists() and refresh:
                cached = pd.read_parquet(path)
                remote = self._refresh_history(symbol=symbol, cached=cached, use_benchmark_adapter=use_benchmark_adapter)
            else:
                remote = self._fetch_full_history(symbol=symbol, use_benchmark_adapter=use_benchmark_adapter)
            remote = self._normalize_frame(
                remote,
                symbol=symbol,
                default_source=self._default_history_source(use_benchmark_adapter),
            )
            merged = self._merge_histories([seed, manual, remote])
            if merged.empty:
                continue
            merged.to_parquet(path, index=False)
            frames.append(merged)
        if not frames:
            raise RuntimeError(f"No {fallback_kind} histories could be collected.")
        result = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        return result

    def build_listing_frame(self, prices: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            prices.groupby("symbol", as_index=False)
            .agg(
                listing_date=("date", "min"),
                last_date=("date", "max"),
                observations=("date", "size"),
            )
            .sort_values("symbol")
            .reset_index(drop=True)
        )
        grouped["listing_year"] = pd.to_datetime(grouped["listing_date"]).dt.year
        return grouped

    def _fetch_full_history(self, symbol: str, use_benchmark_adapter: bool = False) -> pd.DataFrame:
        return self._history_adapter(use_benchmark_adapter).fetch_history(
            symbol=symbol,
            start=self.config.start_date,
            end=self.config.end_date,
        )

    def collect_external_markets(
        self,
        refresh: bool,
        progress_start: float,
        progress_end: float,
        incremental: bool = False,
    ) -> pd.DataFrame:
        if not self.config.external_symbols:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])
        frames: list[pd.DataFrame] = []
        total = max(len(self.config.external_symbols), 1)
        for index, symbol in enumerate(self.config.external_symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting external market data for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = self.config.external_dir / f"{symbol}.parquet"
            if path.exists() and not refresh:
                frame = pd.read_parquet(path)
            elif path.exists() and refresh and incremental:
                cached = pd.read_parquet(path)
                frame = self._refresh_external_history(symbol=symbol, cached=cached)
                frame.to_parquet(path, index=False)
            else:
                frame = self.external_adapter.fetch_history(symbol=symbol, start=self.config.start_date, end=self.config.end_date)
                frame.to_parquet(path, index=False)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("No external market histories could be collected.")
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    def collect_fundamentals(
        self,
        refresh: bool,
        progress_start: float,
        progress_end: float,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        total = max(len(self.config.symbols), 1)
        for index, symbol in enumerate(self.config.symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting quarterly fundamentals for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = self.config.fundamentals_dir / f"{symbol}.parquet"
            if path.exists() and not refresh:
                frame = pd.read_parquet(path)
            else:
                frame = self.fundamentals_adapter.fetch_quarterly_fundamentals(
                    symbol=symbol,
                    quarterly_lag_days=self.config.quarterly_report_lag_days,
                    annual_lag_days=self.config.annual_report_lag_days,
                )
                frame.to_parquet(path, index=False)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("No quarterly fundamentals could be collected.")
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "available_date"]).reset_index(drop=True)

    def collect_company_metadata(
        self,
        refresh: bool,
        progress_start: float,
        progress_end: float,
    ) -> pd.DataFrame:
        if not refresh and not any(self.config.company_dir.glob("*.parquet")):
            return pd.DataFrame(columns=["symbol", "industry_level2", "industry_level3", "issue_share", "charter_capital"])
        frames: list[pd.DataFrame] = []
        total = max(len(self.config.symbols), 1)
        for index, symbol in enumerate(self.config.symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting company metadata for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = self.config.company_dir / f"{symbol}.parquet"
            if path.exists() and not refresh:
                frame = pd.read_parquet(path)
            else:
                frame = self.fundamentals_adapter.fetch_company_overview(symbol=symbol)
                frame.to_parquet(path, index=False)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["symbol", "industry_level2", "industry_level3", "issue_share", "charter_capital"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    def collect_flow_history(
        self,
        refresh: bool,
        progress_start: float,
        progress_end: float,
        incremental: bool = False,
    ) -> pd.DataFrame:
        if not self.config.use_fireant_flow_features:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "foreign_buy_qty",
                    "foreign_sell_qty",
                    "foreign_buy_value",
                    "foreign_sell_value",
                    "foreign_room",
                    "buy_order_count",
                    "sell_order_count",
                    "buy_order_qty",
                    "sell_order_qty",
                    "deal_volume",
                    "putthrough_volume",
                    "putthrough_value",
                    "market_cap",
                    "pe_daily",
                    "pb_daily",
                    "ps_daily",
                    "shares_outstanding",
                    "source",
                ]
            )
        frames: list[pd.DataFrame] = []
        total = max(len(self.config.symbols), 1)
        for index, symbol in enumerate(self.config.symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting FireAnt flow history for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = self.config.flow_dir / f"{symbol}.parquet"
            if path.exists() and not refresh:
                frame = pd.read_parquet(path)
            elif path.exists() and refresh and incremental:
                cached = pd.read_parquet(path)
                frame = self._refresh_flow_history(symbol=symbol, cached=cached)
                frame.to_parquet(path, index=False)
            else:
                frame = self.flow_adapter.fetch_history(symbol=symbol, start=self.config.start_date, end=self.config.end_date)
                frame.to_parquet(path, index=False)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("No FireAnt flow histories could be collected.")
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    def collect_event_history(
        self,
        refresh: bool,
        progress_start: float,
        progress_end: float,
    ) -> pd.DataFrame:
        if not self.config.use_event_features:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "available_date",
                    "issue_date",
                    "record_date",
                    "exright_date",
                    "event_code",
                    "event_name",
                    "ratio",
                    "value",
                    "source",
                ]
            )
        frames: list[pd.DataFrame] = []
        total = max(len(self.config.symbols), 1)
        for index, symbol in enumerate(self.config.symbols, start=1):
            report_progress(
                self.progress_callback,
                f"Collecting filing/event history for {symbol} ({index}/{total})...",
                progress_start + (progress_end - progress_start) * ((index - 1) / total),
            )
            path = self.config.events_dir / f"{symbol}.parquet"
            if path.exists() and not refresh:
                frame = pd.read_parquet(path)
            else:
                frame = self.fundamentals_adapter.fetch_company_events(symbol=symbol)
                frame.to_parquet(path, index=False)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("No filing/event histories could be collected.")
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "available_date"]).reset_index(drop=True)

    def _load_or_collect_fundamentals(self, progress_start: float, progress_end: float) -> pd.DataFrame:
        if self.config.fundamentals_path.exists():
            report_progress(self.progress_callback, "Using cached quarterly fundamentals...", progress_start)
            return pd.read_parquet(self.config.fundamentals_path)
        return self.collect_fundamentals(refresh=False, progress_start=progress_start, progress_end=progress_end)

    def _load_or_collect_company_metadata(self, progress_start: float, progress_end: float) -> pd.DataFrame:
        if self.config.company_metadata_path.exists():
            report_progress(self.progress_callback, "Using cached company metadata...", progress_start)
            return pd.read_parquet(self.config.company_metadata_path)
        return self.collect_company_metadata(refresh=True, progress_start=progress_start, progress_end=progress_end)

    def _load_or_collect_event_history(self, progress_start: float, progress_end: float) -> pd.DataFrame:
        if self.config.events_path.exists():
            report_progress(self.progress_callback, "Using cached filing/event history...", progress_start)
            return pd.read_parquet(self.config.events_path)
        return self.collect_event_history(refresh=True, progress_start=progress_start, progress_end=progress_end)

    def _history_adapter(self, use_benchmark_adapter: bool):
        return self.benchmark_adapter if use_benchmark_adapter else self.adapter

    def _default_history_source(self, use_benchmark_adapter: bool) -> str:
        if use_benchmark_adapter:
            source = getattr(self.benchmark_adapter, "source", self.config.finance_source)
        else:
            source = getattr(self.adapter, "source", self.config.primary_price_source)
        return str(source).upper()

    def _refresh_history(self, symbol: str, cached: pd.DataFrame, use_benchmark_adapter: bool = False) -> pd.DataFrame:
        default_source = self._default_history_source(use_benchmark_adapter)
        if cached.empty:
            return self._normalize_frame(
                self._fetch_full_history(symbol, use_benchmark_adapter=use_benchmark_adapter),
                symbol=symbol,
                default_source=default_source,
            )
        normalized = self._normalize_frame(cached, symbol=symbol, default_source=default_source)
        last_date = pd.to_datetime(normalized["date"]).max()
        target_end = pd.Timestamp(self.config.end_date).normalize()
        if last_date.normalize() >= target_end:
            return normalized
        refresh_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        recent = self._normalize_frame(
            self._history_adapter(use_benchmark_adapter).fetch_history(
                symbol=symbol,
                start=refresh_start,
                end=self.config.end_date,
            ),
            symbol=symbol,
            default_source=default_source,
        )
        return self._merge_histories([normalized, recent])

    def _refresh_external_history(self, symbol: str, cached: pd.DataFrame) -> pd.DataFrame:
        cached = cached.copy()
        if cached.empty:
            return self.external_adapter.fetch_history(symbol=symbol, start=self.config.start_date, end=self.config.end_date)
        cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None)
        last_date = pd.to_datetime(cached["date"]).max()
        target_end = pd.Timestamp(self.config.end_date).normalize()
        if last_date.normalize() >= target_end:
            return cached.sort_values(["symbol", "date"]).reset_index(drop=True)
        refresh_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        recent = self.external_adapter.fetch_history(symbol=symbol, start=refresh_start, end=self.config.end_date)
        return self._merge_symbol_date_frames([cached, recent])

    def _refresh_flow_history(self, symbol: str, cached: pd.DataFrame) -> pd.DataFrame:
        cached = cached.copy()
        if cached.empty:
            return self.flow_adapter.fetch_history(symbol=symbol, start=self.config.start_date, end=self.config.end_date)
        required_columns = {"open", "high", "low", "close", "volume"}
        cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None)
        last_date = pd.to_datetime(cached["date"]).max()
        target_end = pd.Timestamp(self.config.end_date).normalize()
        if last_date.normalize() >= target_end and required_columns.issubset(cached.columns):
            return cached.sort_values(["symbol", "date"]).reset_index(drop=True)
        refresh_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        recent = self.flow_adapter.fetch_history(symbol=symbol, start=refresh_start, end=self.config.end_date)
        return self._merge_symbol_date_frames([cached, recent])

    def _refresh_prices_from_flow_history(self, flow_history: pd.DataFrame) -> pd.DataFrame:
        flow_columns = {"symbol", "date", "open", "high", "low", "close", "volume"}
        if not flow_columns.issubset(flow_history.columns):
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        selected_columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
        if "source" in flow_history.columns:
            selected_columns.append("source")
        flow_prices = flow_history[selected_columns].copy()
        flow_prices = self._normalize_frame(flow_prices, default_source="FIREANT")
        if flow_prices.empty:
            return flow_prices
        if self.config.prices_path.exists():
            cached_prices = self._normalize_frame(
                pd.read_parquet(self.config.prices_path),
                default_source=self._default_history_source(use_benchmark_adapter=False),
            )
            prices = self._merge_histories([cached_prices, flow_prices])
        else:
            prices = flow_prices
        prices.to_parquet(self.config.prices_path, index=False)
        self._write_symbol_histories(prices, self.config.price_dir)
        return prices

    def _load_manual_history(self, symbol: str, kind: str) -> pd.DataFrame:
        candidates = [
            self.config.manual_import_dir / kind / f"{symbol}.parquet",
            self.config.manual_import_dir / kind / f"{symbol}.csv",
            self.config.manual_import_dir / f"{symbol}_{kind}.parquet",
            self.config.manual_import_dir / f"{symbol}_{kind}.csv",
        ]
        for path in candidates:
            if path.exists():
                frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                normalized = self._normalize_frame(frame, symbol=symbol)
                if not normalized.empty:
                    normalized["source"] = "MANUAL"
                    return normalized
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])

    def _load_seed_history(self, symbol: str, kind: str) -> pd.DataFrame:
        root = self.config.project_dir.parent
        per_symbol = root / "trade60_app" / "data" / "raw" / kind / f"{symbol}.parquet"
        aggregate = root / "trade60_app" / "data" / "raw" / f"{kind}.parquet"
        fallback = root / "data" / "all60" / "raw" / f"{kind}.parquet"
        candidates = [per_symbol, aggregate, fallback]
        for path in candidates:
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if "symbol" in frame.columns:
                frame = frame.loc[frame["symbol"] == symbol]
            normalized = self._normalize_frame(frame, symbol=symbol)
            if not normalized.empty:
                normalized["source"] = "SEED"
                return normalized
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])

    def _normalize_frame(
        self,
        frame: pd.DataFrame,
        symbol: str | None = None,
        default_source: str | None = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])
        normalized = frame.rename(columns={"time": "date"}).copy()
        if "symbol" not in normalized.columns:
            if symbol is None:
                raise ValueError("Historical data frame is missing a `symbol` column.")
            normalized["symbol"] = symbol
        if "source" not in normalized.columns:
            normalized["source"] = (default_source or self.config.primary_price_source).upper()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in normalized.columns:
                normalized[column] = pd.NA
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized[["symbol", "date", "open", "high", "low", "close", "volume", "source"]]
        normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
        normalized = normalized.loc[(normalized[list(PRICE_COLUMNS)] > 0).all(axis=1)].copy()
        normalized = self._promote_legacy_thousand_unit_quotes(normalized)
        normalized = self._repair_internal_price_scale_jumps(normalized)
        return normalized.drop_duplicates(subset=["symbol", "date"], keep="last").sort_values("date").reset_index(drop=True)

    def _merge_histories(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])
        combined = pd.concat(usable, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        return self._repair_internal_price_scale_jumps(combined)

    @staticmethod
    def _merge_symbol_date_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return pd.DataFrame()
        combined = pd.concat(usable, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        return combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    @staticmethod
    def _write_symbol_histories(frame: pd.DataFrame, target_dir: Path) -> None:
        if frame.empty or "symbol" not in frame.columns:
            return
        for symbol, part in frame.groupby("symbol", sort=False):
            if not symbol:
                continue
            part.sort_values("date").reset_index(drop=True).to_parquet(target_dir / f"{symbol}.parquet", index=False)

    @staticmethod
    def _promote_legacy_thousand_unit_quotes(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", *PRICE_COLUMNS}
        if frame.empty or not required.issubset(frame.columns):
            return frame
        promoted_parts: list[pd.DataFrame] = []
        for _, part in frame.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
            current = part.copy()
            closes = current["close"].dropna()
            if closes.empty:
                promoted_parts.append(current)
                continue
            # Legacy vnstock equity files are quoted in thousands of VND; FireAnt uses full VND.
            if float(closes.max()) < 1000.0:
                current.loc[:, list(PRICE_COLUMNS)] = current.loc[:, list(PRICE_COLUMNS)].astype(float) * 1000.0
            promoted_parts.append(current)
        if not promoted_parts:
            return frame
        promoted = pd.concat(promoted_parts, ignore_index=True)
        return promoted.sort_values(["symbol", "date"]).reset_index(drop=True)

    @classmethod
    def _repair_internal_price_scale_jumps(cls, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "date", *PRICE_COLUMNS}
        if frame.empty or not required.issubset(frame.columns):
            return frame
        repaired_parts: list[pd.DataFrame] = []
        for _, part in frame.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
            current = part.copy().reset_index(drop=True)
            prices = current[list(PRICE_COLUMNS)].to_numpy(dtype=float, copy=True)
            multiplier = 1.0
            next_close: float | None = None
            for idx in range(len(current) - 1, -1, -1):
                adjusted_close = float(prices[idx, 3]) * multiplier
                if next_close is not None and next_close > 0:
                    step_multiplier = cls._scale_multiplier_from_ratio(adjusted_close / next_close)
                    if step_multiplier != 1.0:
                        multiplier *= step_multiplier
                        adjusted_close = float(prices[idx, 3]) * multiplier
                prices[idx, :] = prices[idx, :] * multiplier
                next_close = adjusted_close
            current.loc[:, list(PRICE_COLUMNS)] = prices
            repaired_parts.append(current)
        if not repaired_parts:
            return frame
        repaired = pd.concat(repaired_parts, ignore_index=True)
        return repaired.sort_values(["symbol", "date"]).reset_index(drop=True)

    @staticmethod
    def _scale_multiplier_from_ratio(ratio: float) -> float:
        if not pd.notna(ratio) or ratio <= 0:
            return 1.0
        if 100.0 <= ratio <= 10000.0:
            return 0.001
        if 0.0001 <= ratio <= 0.01:
            return 1000.0
        return 1.0


def collect_market_data(
    config: TradingBotConfig,
    refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
    mode: str = "full",
) -> DataBundle:
    collector = MarketDataCollector(config=config, progress_callback=progress_callback)
    if mode == "decision":
        return collector.collect_for_decision(refresh=refresh)
    return collector.collect_all(refresh=refresh)
