from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator

from vn30_strategy.data.adjustments import build_monthly_snapshot
from vn30_strategy.utils import safe_divide


def build_technical_features(prices: pd.DataFrame, benchmarks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_daily = _prepare_benchmark_features(benchmarks)

    feature_frames: list[pd.DataFrame] = []
    breadth_frames: list[pd.DataFrame] = []

    for symbol, symbol_df in prices.groupby("symbol", sort=False):
        df = symbol_df.sort_values("date").reset_index(drop=True).copy()
        df["ret_5d"] = df["adj_close"].pct_change(5, fill_method=None)
        df["ret_20d"] = df["adj_close"].pct_change(20, fill_method=None)
        df["ret_60d"] = df["adj_close"].pct_change(60, fill_method=None)
        df["ret_120d"] = df["adj_close"].pct_change(120, fill_method=None)
        df["ret_252d"] = df["adj_close"].pct_change(252, fill_method=None)
        df["volatility_20d"] = df["total_return"].rolling(20).std() * np.sqrt(20)
        df["volatility_60d"] = df["total_return"].rolling(60).std() * np.sqrt(20)
        df["avg_volume_20d"] = df["volume"].rolling(20).mean()
        df["avg_turnover_20d"] = (df["close"] * df["volume"]).rolling(20).mean()
        df["volume_zscore_20d"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
        df["distance_high_60d"] = safe_divide(df["adj_close"], df["adj_close"].rolling(60).max()) - 1.0
        df["distance_high_252d"] = safe_divide(df["adj_close"], df["adj_close"].rolling(252).max()) - 1.0
        df["drawdown_252d"] = safe_divide(df["adj_close"], df["adj_close"].rolling(252).max()) - 1.0

        df["sma_20"] = SMAIndicator(df["adj_close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(df["adj_close"], window=50).sma_indicator()
        df["sma_200"] = SMAIndicator(df["adj_close"], window=200).sma_indicator()
        df["ema_20"] = EMAIndicator(df["adj_close"], window=20).ema_indicator()
        df["distance_ma20"] = safe_divide(df["adj_close"], df["sma_20"]) - 1.0
        df["distance_ma50"] = safe_divide(df["adj_close"], df["sma_50"]) - 1.0
        df["distance_ma200"] = safe_divide(df["adj_close"], df["sma_200"]) - 1.0

        df["rsi_14"] = RSIIndicator(df["adj_close"], window=14).rsi()
        macd = MACD(df["adj_close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = safe_divide(df["atr"], df["close"])
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()
        stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
        df["stoch"] = stoch.stoch()
        df["mfi"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
        boll = BollingerBands(df["adj_close"], window=20, window_dev=2)
        df["bb_pos"] = safe_divide(df["adj_close"] - boll.bollinger_lband(), boll.bollinger_hband() - boll.bollinger_lband())

        merged = df.merge(benchmark_daily, on="date", how="left").reset_index(drop=True)
        for column in [
            "benchmark_return",
            "benchmark_ret_20d",
            "benchmark_ret_60d",
            "benchmark_vol_20d",
            "benchmark_drawdown_252d",
            "benchmark_distance_ma200",
        ]:
            df[column] = merged[column].to_numpy()
        df["relative_strength_20d"] = df["ret_20d"] - df["benchmark_ret_20d"]
        df["relative_strength_60d"] = df["ret_60d"] - df["benchmark_ret_60d"]
        covariance = df["total_return"].rolling(60).cov(df["benchmark_return"])
        variance = df["benchmark_return"].rolling(60).var()
        df["beta_60d"] = safe_divide(covariance, variance)
        df["idiosyncratic_strength"] = df["relative_strength_20d"] - df["beta_60d"] * df["benchmark_ret_20d"]

        breadth_frames.append(
            df[["date", "symbol", "distance_ma200", "ret_20d", "ret_60d", "avg_turnover_20d"]].assign(
                above_ma200=(df["distance_ma200"] > 0).astype(float),
                positive_20d=(df["ret_20d"] > 0).astype(float),
            )
        )
        feature_frames.append(df)

    daily = pd.concat(feature_frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    breadth = pd.concat(breadth_frames, ignore_index=True)

    breadth_daily = breadth.groupby("date", as_index=False).agg(
        breadth_above_ma200=("above_ma200", "mean"),
        breadth_positive_20d=("positive_20d", "mean"),
        breadth_ret_20d=("ret_20d", "median"),
        breadth_ret_60d=("ret_60d", "median"),
        breadth_turnover_20d=("avg_turnover_20d", "median"),
    )
    breadth_monthly = build_monthly_snapshot(breadth_daily.assign(symbol="BREADTH")).drop(columns=["symbol"])
    monthly = build_monthly_snapshot(daily)
    monthly = monthly.merge(breadth_monthly, on=["date", "month"], how="left")
    return daily, monthly


def _prepare_benchmark_features(benchmarks: pd.DataFrame) -> pd.DataFrame:
    vnindex = benchmarks.loc[benchmarks["symbol"] == "VNINDEX"].sort_values("date").copy()
    if vnindex.empty:
        raise RuntimeError("VNINDEX benchmark history is required for regime features.")
    vnindex["benchmark_return"] = vnindex["adj_close"].pct_change(fill_method=None).fillna(0.0)
    vnindex["benchmark_ret_20d"] = vnindex["adj_close"].pct_change(20, fill_method=None)
    vnindex["benchmark_ret_60d"] = vnindex["adj_close"].pct_change(60, fill_method=None)
    vnindex["benchmark_vol_20d"] = vnindex["benchmark_return"].rolling(20).std() * np.sqrt(20)
    vnindex["benchmark_drawdown_252d"] = safe_divide(vnindex["adj_close"], vnindex["adj_close"].rolling(252).max()) - 1.0
    vnindex["benchmark_sma_200"] = SMAIndicator(vnindex["adj_close"], window=200).sma_indicator()
    vnindex["benchmark_distance_ma200"] = safe_divide(vnindex["adj_close"], vnindex["benchmark_sma_200"]) - 1.0
    return vnindex[
        [
            "date",
            "benchmark_return",
            "benchmark_ret_20d",
            "benchmark_ret_60d",
            "benchmark_vol_20d",
            "benchmark_drawdown_252d",
            "benchmark_distance_ma200",
        ]
    ]
