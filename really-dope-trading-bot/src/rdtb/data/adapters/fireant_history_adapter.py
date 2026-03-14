from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests


@dataclass(slots=True)
class FireAntHistoryAdapter:
    timeout_seconds: int = 60

    def fetch_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        url = (
            "https://www.fireant.vn/api/Data/Companies/HistoricalQuotes"
            f"?symbol={symbol}&startDate={start}&endDate={end}"
        )
        response = requests.get(
            url,
            timeout=self.timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected FireAnt payload for {symbol}: {payload}")
        return self.normalize(payload, symbol=symbol)

    @staticmethod
    def normalize(payload: list[dict], symbol: str) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adj_close",
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
        frame = pd.DataFrame(payload)
        renamed = frame.rename(
            columns={
                "Date": "date",
                "Symbol": "symbol",
                "PriceOpen": "open",
                "PriceHigh": "high",
                "PriceLow": "low",
                "PriceClose": "close",
                "AdjClose": "adj_close",
                "Volume": "volume",
                "BuyForeignQuantity": "foreign_buy_qty",
                "SellForeignQuantity": "foreign_sell_qty",
                "BuyForeignValue": "foreign_buy_value",
                "SellForeignValue": "foreign_sell_value",
                "CurrentForeignRoom": "foreign_room",
                "BuyCount": "buy_order_count",
                "SellCount": "sell_order_count",
                "BuyQuantity": "buy_order_qty",
                "SellQuantity": "sell_order_qty",
                "DealVolume": "deal_volume",
                "PutthroughVolume": "putthrough_volume",
                "PutthroughValue": "putthrough_value",
                "MarketCap": "market_cap",
                "PE": "pe_daily",
                "PB": "pb_daily",
                "PS": "ps_daily",
                "Shares": "shares_outstanding",
            }
        ).copy()
        renamed["symbol"] = symbol
        renamed["date"] = pd.to_datetime(renamed["date"]).dt.tz_localize(None)
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
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
        ]
        for column in numeric_columns:
            renamed[column] = pd.to_numeric(renamed.get(column), errors="coerce")
        renamed["source"] = "FIREANT"
        columns = ["symbol", "date", *numeric_columns, "source"]
        return renamed[columns].drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
