# FireAnt Daily Stock Data Guide

This repo already uses FireAnt as the default source for VN equity daily price history. This note explains the exact pattern so you can reuse it in a future project without reverse-engineering the code again.

## How `really-dope-trading-bot` uses FireAnt

- `FireAntHistoryAdapter` is the primary adapter for VN equity price history.
- The same adapter is also reused for `flow_history`, because the FireAnt payload includes price, foreign-flow, order-flow, and daily valuation fields together.
- Benchmarks and fundamentals are still fetched from the configured `vnstock` / `VCI` adapters.
- The config defaults already reflect this:
  - `primary_price_source = "FIREANT"`
  - `use_fireant_flow_features = True`

## Endpoint

The adapter calls:

```text
GET https://www.fireant.vn/api/Data/Companies/HistoricalQuotes?symbol={SYMBOL}&startDate={YYYY-MM-DD}&endDate={YYYY-MM-DD}
```

Example:

```text
https://www.fireant.vn/api/Data/Companies/HistoricalQuotes?symbol=FPT&startDate=2024-01-01&endDate=2024-12-31
```

The current implementation sends a browser-like `User-Agent` header:

```http
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36
```

That is worth keeping in future projects because this endpoint behaves more reliably when the request looks like a normal browser request.

## What comes back

The code expects the response body to be a JSON list. Each row can contain:

- OHLCV fields
- adjusted close
- foreign buy/sell quantity and value
- foreign room
- buy/sell order count and quantity
- deal / putthrough volume and value
- market cap
- daily `PE`, `PB`, `PS`
- shares outstanding

If the payload is not a JSON list, the adapter raises an error instead of silently accepting a schema change.

## Normalized columns

This repo maps FireAnt fields into this normalized schema:

| FireAnt field | Normalized column |
| --- | --- |
| `Date` | `date` |
| `Symbol` | `symbol` |
| `PriceOpen` | `open` |
| `PriceHigh` | `high` |
| `PriceLow` | `low` |
| `PriceClose` | `close` |
| `AdjClose` | `adj_close` |
| `Volume` | `volume` |
| `BuyForeignQuantity` | `foreign_buy_qty` |
| `SellForeignQuantity` | `foreign_sell_qty` |
| `BuyForeignValue` | `foreign_buy_value` |
| `SellForeignValue` | `foreign_sell_value` |
| `CurrentForeignRoom` | `foreign_room` |
| `BuyCount` | `buy_order_count` |
| `SellCount` | `sell_order_count` |
| `BuyQuantity` | `buy_order_qty` |
| `SellQuantity` | `sell_order_qty` |
| `DealVolume` | `deal_volume` |
| `PutthroughVolume` | `putthrough_volume` |
| `PutthroughValue` | `putthrough_value` |
| `MarketCap` | `market_cap` |
| `PE` | `pe_daily` |
| `PB` | `pb_daily` |
| `PS` | `ps_daily` |
| `Shares` | `shares_outstanding` |

The adapter then:

- forces `symbol`
- parses `date` with `pandas.to_datetime(...).dt.tz_localize(None)`
- coerces all numeric columns with `pd.to_numeric(..., errors="coerce")`
- adds `source = "FIREANT"`
- drops duplicate rows by `["symbol", "date"]`
- sorts by `["symbol", "date"]`

## Important price-scale note

FireAnt prices are in full VND units. This repo contains extra repair logic because some older `vnstock` equity files were stored in thousands of VND.

For a future project, the safe default is:

- treat FireAnt equity prices as full VND
- do not divide them by `1000`
- if you merge FireAnt with older cached data from another source, check for thousand-fold scale mismatches

## Minimal reusable client

If you want the same behavior in another project, this is the smallest practical version:

```python
from dataclasses import dataclass

import pandas as pd
import requests


COLUMN_MAP = {
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

NUMERIC_COLUMNS = [
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


@dataclass(slots=True)
class FireAntDailyClient:
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

        if not payload:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])

        frame = pd.DataFrame(payload).rename(columns=COLUMN_MAP).copy()
        frame["symbol"] = symbol
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)

        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")

        frame["source"] = "FIREANT"
        columns = ["symbol", "date", *NUMERIC_COLUMNS, "source"]
        return (
            frame[columns]
            .drop_duplicates(subset=["symbol", "date"])
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
```

## Daily refresh pattern

The repo uses a simple cache-first approach that is easy to copy:

1. On the first run, fetch the full range from `start_date` to `end_date`.
2. Save one file per symbol.
3. On the daily job, read the cached file and find the latest `date`.
4. If cache is already up to date, return it.
5. Otherwise call FireAnt again with:
   - `startDate = last_cached_date + 1 day`
   - `endDate = target_date`
6. Merge cached and new rows on `["symbol", "date"]`, keeping the newest row.
7. Rewrite the per-symbol cache and any aggregate dataset you maintain.

In this repo:

- price cache lives under `data/raw/prices/{symbol}.parquet`
- flow cache lives under `data/raw/flow/{symbol}.parquet`
- aggregate outputs are also written to:
  - `data/raw/prices.parquet`
  - `data/raw/flow.parquet`

## Why the bot reuses FireAnt flow history for daily price refresh

`collect_for_decision(refresh=True)` first refreshes FireAnt flow history, then extracts:

- `symbol`
- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

from that same FireAnt dataset and uses it to refresh the main prices cache.

That means one FireAnt pull can support both:

- daily prices
- flow / valuation features

For a future project, this is a good pattern if you want fewer network calls and a single source of truth for end-of-day VN equity data.

## Recommended setup for a new project

- Add `requests`, `pandas`, and `pyarrow` explicitly to your dependencies.
- Keep one normalized cache file per symbol.
- Use FireAnt as the source for VN equity daily history.
- Store prices in full VND units.
- Deduplicate by `symbol` and `date`.
- Fail loudly if the payload is no longer a JSON list.
- Add retry / backoff around `requests.get(...)` if the new project will run unattended on a scheduler.

## Practical daily job example

For a simple cron job or scheduler:

```python
from pathlib import Path

import pandas as pd


def refresh_symbol(symbol: str, cache_dir: Path, end_date: str) -> pd.DataFrame:
    client = FireAntDailyClient()
    cache_path = cache_dir / f"{symbol}.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        last_date = pd.to_datetime(cached["date"]).max().strftime("%Y-%m-%d")
        if last_date >= end_date:
            return cached.sort_values(["symbol", "date"]).reset_index(drop=True)
        start_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        recent = client.fetch_history(symbol=symbol, start=start_date, end=end_date)
        refreshed = (
            pd.concat([cached, recent], ignore_index=True)
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
    else:
        refreshed = client.fetch_history(symbol=symbol, start="2006-01-01", end=end_date)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    refreshed.to_parquet(cache_path, index=False)
    return refreshed
```

## Source files in this repo

If you want the exact implementation details later, start here:

- `src/rdtb/data/adapters/fireant_history_adapter.py`
- `src/rdtb/data/collector.py`
- `src/rdtb/config.py`

