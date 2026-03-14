from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class YFinanceDailyAdapter:
    auto_adjust: bool = True

    def fetch_history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `yfinance` package is required for external market features.") from exc
        ticker = yf.Ticker(symbol)
        frame = ticker.history(start=start, end=end, interval="1d", auto_adjust=self.auto_adjust)
        return self.normalize(frame, symbol)

    @staticmethod
    def normalize(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "source"])
        normalized = frame.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        normalized["symbol"] = symbol
        normalized["source"] = "YFINANCE"
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
        for column in ["open", "high", "low", "close", "volume"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized[["symbol", "date", "open", "high", "low", "close", "volume", "source"]]
        return normalized.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
