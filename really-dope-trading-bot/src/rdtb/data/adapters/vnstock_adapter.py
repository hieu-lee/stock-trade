from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(slots=True)
class VnstockDailyAdapter:
    source: str = "VCI"
    min_request_spacing_seconds: float = 6.5
    retry_cooldown_seconds: float = 75.0
    max_retries: int = 4
    _last_request_at: float = field(init=False, default=0.0)

    def fetch_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        Quote = self._resolve_quote_class()
        quote = Quote(symbol=symbol, source=self.source, show_log=False)
        frame = self._invoke(
            lambda: quote.history(start=start, end=end, interval="1D").copy(),
            context=f"{symbol}:{start}:{end}",
        )
        return self.normalize(frame, symbol=symbol, source=self.source)

    def _resolve_quote_class(self):
        try:
            from vnstock import Quote  # type: ignore
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "The `vnstock` package is required for live data collection. "
                "Install project dependencies or provide manual import files in `data/manual_imports/`."
            ) from exc
        return Quote

    def _invoke(self, fn: Callable[[], pd.DataFrame], context: str) -> pd.DataFrame:
        for _ in range(self.max_retries):
            self._respect_rate_limit()
            try:
                result = fn()
                self._last_request_at = time.time()
                return result
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # pragma: no cover - network dependent
                message = str(exc)
                if "Rate limit exceeded" in message or "GIỚI HẠN API" in message or "RetryError" in message:
                    time.sleep(self.retry_cooldown_seconds)
                    self._last_request_at = time.time()
                    continue
                raise RuntimeError(f"Failed to fetch {context}") from exc
        raise RuntimeError(f"Repeated rate-limit failures while fetching {context}.")

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_spacing_seconds:
            time.sleep(self.min_request_spacing_seconds - elapsed)

    @staticmethod
    def normalize(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"]
            )
        normalized = frame.rename(columns={"time": "date"}).copy()
        normalized["symbol"] = symbol
        normalized["source"] = source.upper()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized[["symbol", "date", "open", "high", "low", "close", "volume", "source"]]
        normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
        return normalized.drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)
