from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.utils import ensure_directories


@dataclass(slots=True)
class DuckDBMarketStore:
    config: TradingBotConfig

    def ensure_layout(self) -> None:
        ensure_directories(
            [
                self.config.raw_dir,
                self.config.processed_dir,
                self.config.manual_import_dir,
                self.config.price_dir,
                self.config.benchmark_dir,
                self.config.external_dir,
                self.config.fundamentals_dir,
                self.config.company_dir,
                self.config.flow_dir,
                self.config.events_dir,
                self.config.artifacts_dir,
                self.config.models_dir,
                self.config.reports_dir,
                self.config.recommendations_dir,
            ]
        )

    def write_bundle(
        self,
        universe: pd.DataFrame,
        prices: pd.DataFrame,
        benchmarks: pd.DataFrame,
        listings: pd.DataFrame,
        external_markets: pd.DataFrame,
        fundamentals: pd.DataFrame,
        company_metadata: pd.DataFrame,
        flow_history: pd.DataFrame,
        event_history: pd.DataFrame,
    ) -> None:
        self.ensure_layout()
        universe.to_parquet(self.config.universe_path, index=False)
        prices.to_parquet(self.config.prices_path, index=False)
        benchmarks.to_parquet(self.config.benchmarks_path, index=False)
        listings.to_parquet(self.config.listings_path, index=False)
        external_markets.to_parquet(self.config.external_markets_path, index=False)
        fundamentals.to_parquet(self.config.fundamentals_path, index=False)
        company_metadata.to_parquet(self.config.company_metadata_path, index=False)
        flow_history.to_parquet(self.config.flow_path, index=False)
        event_history.to_parquet(self.config.events_path, index=False)
        self._refresh_duckdb_views(
            universe_path=self.config.universe_path,
            prices_path=self.config.prices_path,
            benchmarks_path=self.config.benchmarks_path,
            listings_path=self.config.listings_path,
            external_markets_path=self.config.external_markets_path,
            fundamentals_path=self.config.fundamentals_path,
            company_metadata_path=self.config.company_metadata_path,
            flow_path=self.config.flow_path,
            event_path=self.config.events_path,
        )

    def _refresh_duckdb_views(
        self,
        universe_path,
        prices_path,
        benchmarks_path,
        listings_path,
        external_markets_path,
        fundamentals_path,
        company_metadata_path,
        flow_path,
        event_path,
    ) -> None:
        try:
            import duckdb
        except Exception:  # pragma: no cover - optional runtime dependency
            return
        connection = duckdb.connect(str(self.config.duckdb_path))
        try:
            connection.execute(
                f"""
                create or replace view universe as
                select * from read_parquet({self._sql_string(universe_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view prices as
                select * from read_parquet({self._sql_string(prices_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view benchmarks as
                select * from read_parquet({self._sql_string(benchmarks_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view listings as
                select * from read_parquet({self._sql_string(listings_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view external_markets as
                select * from read_parquet({self._sql_string(external_markets_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view fundamentals as
                select * from read_parquet({self._sql_string(fundamentals_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view company_metadata as
                select * from read_parquet({self._sql_string(company_metadata_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view flow_history as
                select * from read_parquet({self._sql_string(flow_path)})
                """
            )
            connection.execute(
                f"""
                create or replace view event_history as
                select * from read_parquet({self._sql_string(event_path)})
                """
            )
        finally:
            connection.close()

    @staticmethod
    def _sql_string(path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    def read_prices(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.prices_path)

    def read_benchmarks(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.benchmarks_path)

    def read_universe(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.universe_path)

    def read_listings(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.listings_path)

    def read_external_markets(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.external_markets_path)

    def read_fundamentals(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.fundamentals_path)

    def read_company_metadata(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.company_metadata_path)

    def read_flow_history(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.flow_path)

    def read_event_history(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.events_path)
