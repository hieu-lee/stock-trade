from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange

from trade60_app.config import Trade60Config
from trade60_app.data.collector import DataBundle
from trade60_app.utils import ProgressCallback, report_progress, safe_divide

LIVE_FEATURE_LOOKBACK_ROWS = 320


def build_daily_feature_panel(
    bundle: DataBundle,
    config: Trade60Config,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    benchmark_daily = _prepare_benchmark_features(bundle.benchmarks)

    feature_frames: list[pd.DataFrame] = []
    breadth_frames: list[pd.DataFrame] = []
    total_symbols = max(bundle.prices["symbol"].nunique(), 1)

    for index, (symbol, symbol_df) in enumerate(bundle.prices.groupby("symbol", sort=False), start=1):
        report_progress(
            progress_callback,
            f"Đang xây dựng đặc trưng cho {symbol} ({index}/{total_symbols})...",
            (index - 1) / total_symbols,
        )
        df = symbol_df.sort_values("date").reset_index(drop=True).copy()
        df["symbol"] = symbol
        df["ret_1d"] = df["close"].pct_change(fill_method=None)
        df["ret_5d"] = df["close"].pct_change(5, fill_method=None)
        df["ret_10d"] = df["close"].pct_change(10, fill_method=None)
        df["ret_20d"] = df["close"].pct_change(20, fill_method=None)
        df["ret_40d"] = df["close"].pct_change(40, fill_method=None)
        df["gap_open"] = safe_divide(df["open"], df["close"].shift(1)) - 1.0
        df["intraday_return"] = safe_divide(df["close"], df["open"]) - 1.0
        df["range_pct"] = safe_divide(df["high"] - df["low"], df["close"])
        df["volatility_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)
        df["volatility_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        df["avg_volume_20d"] = df["volume"].rolling(20).mean()
        df["avg_turnover_20d"] = (df["close"] * df["volume"]).rolling(20).mean()
        df["volume_zscore_20d"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

        df["sma_10"] = SMAIndicator(df["close"], window=10).sma_indicator()
        df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
        df["sma_200"] = SMAIndicator(df["close"], window=200).sma_indicator()
        df["distance_ma10"] = safe_divide(df["close"], df["sma_10"]) - 1.0
        df["distance_ma20"] = safe_divide(df["close"], df["sma_20"]) - 1.0
        df["distance_ma50"] = safe_divide(df["close"], df["sma_50"]) - 1.0
        df["distance_ma200"] = safe_divide(df["close"], df["sma_200"]) - 1.0

        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        df["atr"] = atr.average_true_range()
        df["atr_pct"] = safe_divide(df["atr"], df["close"])
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()

        merged = df.merge(benchmark_daily, on="date", how="left").reset_index(drop=True)
        for column in [
            "benchmark_ret_1d",
            "benchmark_ret_5d",
            "benchmark_ret_10d",
            "benchmark_ret_20d",
            "benchmark_vol_20d",
            "benchmark_drawdown_252d",
            "benchmark_distance_ma200",
            "benchmark_gap_open",
            "benchmark_intraday_return",
        ]:
            df[column] = merged[column].to_numpy()
        df["relative_strength_10d"] = df["ret_10d"] - df["benchmark_ret_10d"]
        df["relative_strength_20d"] = df["ret_20d"] - df["benchmark_ret_20d"]
        covariance = df["ret_1d"].rolling(60).cov(df["benchmark_ret_1d"])
        variance = df["benchmark_ret_1d"].rolling(60).var()
        df["beta_60d"] = safe_divide(covariance, variance)

        breadth_frames.append(
            df[["date", "symbol", "distance_ma50", "distance_ma200", "ret_10d", "ret_20d", "avg_turnover_20d"]].assign(
                above_ma50=(df["distance_ma50"] > 0).astype(float),
                above_ma200=(df["distance_ma200"] > 0).astype(float),
                positive_10d=(df["ret_10d"] > 0).astype(float),
            )
        )
        feature_frames.append(df)
        report_progress(progress_callback, f"Đã xong đặc trưng cho {symbol} ({index}/{total_symbols})", index / total_symbols)

    panel = pd.concat(feature_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    breadth = pd.concat(breadth_frames, ignore_index=True)
    breadth_daily = breadth.groupby("date", as_index=False).agg(
        breadth_above_ma50=("above_ma50", "mean"),
        breadth_above_ma200=("above_ma200", "mean"),
        breadth_positive_10d=("positive_10d", "mean"),
        breadth_ret_20d=("ret_20d", "median"),
        breadth_turnover_20d=("avg_turnover_20d", "median"),
    )
    panel = panel.merge(breadth_daily, on="date", how="left")
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    labeled_frames = [_attach_symbol_targets(symbol_df, config) for _, symbol_df in panel.groupby("symbol", sort=False)]
    labeled = pd.concat(labeled_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    labeled = labeled.merge(_prepare_regime_targets(bundle.benchmarks), on="date", how="left")
    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    labeled["is_trainable"] = (
        labeled["target_long"].notna()
        & labeled["target_regime"].notna()
        & labeled["entry_price_next_open"].notna()
        & labeled["open_next"].notna()
    )
    report_progress(progress_callback, "Đã hoàn tất xây dựng đặc trưng.", 1.0)
    return labeled


def build_live_feature_snapshot(
    bundle: DataBundle,
    config: Trade60Config,
    progress_callback: ProgressCallback | None = None,
    lookback_rows: int = LIVE_FEATURE_LOOKBACK_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    report_progress(progress_callback, "Đang chuẩn bị cửa sổ dữ liệu gần nhất cho khuyến nghị live...", 0.0)
    recent_prices = (
        bundle.prices.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False).tail(lookback_rows).reset_index(drop=True)
    )
    recent_benchmarks = (
        bundle.benchmarks.sort_values(["symbol", "date"]).groupby("symbol", group_keys=False).tail(lookback_rows).reset_index(drop=True)
    )
    recent_bundle = DataBundle(
        universe=bundle.universe,
        prices=recent_prices,
        benchmarks=recent_benchmarks,
    )
    panel = build_daily_feature_panel(recent_bundle, config, progress_callback=progress_callback)
    latest_date = pd.to_datetime(panel["date"]).max()
    latest_panel = panel.loc[pd.to_datetime(panel["date"]) == latest_date].copy()
    trading_calendar = bundle.prices["date"].drop_duplicates().sort_values().reset_index(drop=True)
    report_progress(progress_callback, f"Đã sẵn sàng snapshot live cho ngày {latest_date.date()}.", 1.0)
    return latest_panel, trading_calendar


def _prepare_benchmark_features(benchmarks: pd.DataFrame) -> pd.DataFrame:
    vnindex = benchmarks.loc[benchmarks["symbol"] == "VNINDEX"].sort_values("date").copy()
    if vnindex.empty:
        raise RuntimeError("VNINDEX benchmark history is required for regime features.")
    vnindex["benchmark_ret_1d"] = vnindex["close"].pct_change(fill_method=None).fillna(0.0)
    vnindex["benchmark_ret_5d"] = vnindex["close"].pct_change(5, fill_method=None)
    vnindex["benchmark_ret_10d"] = vnindex["close"].pct_change(10, fill_method=None)
    vnindex["benchmark_ret_20d"] = vnindex["close"].pct_change(20, fill_method=None)
    vnindex["benchmark_vol_20d"] = vnindex["benchmark_ret_1d"].rolling(20).std() * np.sqrt(252)
    vnindex["benchmark_gap_open"] = safe_divide(vnindex["open"], vnindex["close"].shift(1)) - 1.0
    vnindex["benchmark_intraday_return"] = safe_divide(vnindex["close"], vnindex["open"]) - 1.0
    vnindex["benchmark_drawdown_252d"] = safe_divide(vnindex["close"], vnindex["close"].rolling(252).max()) - 1.0
    benchmark_sma_200 = SMAIndicator(vnindex["close"], window=200).sma_indicator()
    vnindex["benchmark_distance_ma200"] = safe_divide(vnindex["close"], benchmark_sma_200) - 1.0
    return vnindex[
        [
            "date",
            "benchmark_ret_1d",
            "benchmark_ret_5d",
            "benchmark_ret_10d",
            "benchmark_ret_20d",
            "benchmark_vol_20d",
            "benchmark_drawdown_252d",
            "benchmark_distance_ma200",
            "benchmark_gap_open",
            "benchmark_intraday_return",
        ]
    ]


def _prepare_regime_targets(benchmarks: pd.DataFrame) -> pd.DataFrame:
    vnindex = benchmarks.loc[benchmarks["symbol"] == "VNINDEX"].sort_values("date").copy()
    if vnindex.empty:
        raise RuntimeError("VNINDEX benchmark history is required for regime targets.")
    vnindex["benchmark_entry_next_open"] = vnindex["open"].shift(-1)
    future_close_10d = vnindex["close"].shift(-10)
    future_min_low_10d = _future_window_stat(vnindex["low"], window=10, stat="min")
    vnindex["benchmark_forward_return_10d"] = safe_divide(future_close_10d, vnindex["benchmark_entry_next_open"]) - 1.0
    vnindex["benchmark_forward_min_10d"] = safe_divide(future_min_low_10d, vnindex["benchmark_entry_next_open"]) - 1.0
    vnindex["target_regime"] = (
        (vnindex["benchmark_forward_return_10d"] > 0.01)
        & (vnindex["benchmark_forward_min_10d"] > -0.05)
    ).astype(float)
    vnindex.loc[vnindex["benchmark_forward_return_10d"].isna(), "target_regime"] = np.nan
    return vnindex[["date", "target_regime"]]


def _attach_symbol_targets(frame: pd.DataFrame, config: Trade60Config) -> pd.DataFrame:
    df = frame.sort_values("date").reset_index(drop=True).copy()
    df["open_next"] = df["open"].shift(-1)
    df["entry_price_next_open"] = df["open_next"]
    future_close_5d = df["close"].shift(-5)
    future_close_10d = df["close"].shift(-10)
    future_close_20d = df["close"].shift(-20)
    future_min_low_10d = _future_window_stat(df["low"], window=10, stat="min")
    future_max_high_20d = _future_window_stat(df["high"], window=20, stat="max")
    df["forward_return_5d"] = safe_divide(future_close_5d, df["entry_price_next_open"]) - 1.0
    df["forward_return_10d"] = safe_divide(future_close_10d, df["entry_price_next_open"]) - 1.0
    df["forward_return_20d"] = safe_divide(future_close_20d, df["entry_price_next_open"]) - 1.0
    df["forward_min_return_10d"] = safe_divide(future_min_low_10d, df["entry_price_next_open"]) - 1.0
    df["forward_max_return_20d"] = safe_divide(future_max_high_20d, df["entry_price_next_open"]) - 1.0
    df["target_long"] = (
        (df["forward_return_10d"] > 0.02)
        & (df["forward_return_20d"] > 0.04)
        & (df["forward_min_return_10d"] > -max(config.stop_loss_pct, 0.05))
    ).astype(float)
    df.loc[df["forward_return_20d"].isna(), "target_long"] = np.nan
    return df


def _future_window_stat(series: pd.Series, window: int, stat: str) -> pd.Series:
    shifted = series.shift(-1)
    reversed_series = shifted.iloc[::-1]
    if stat == "min":
        result = reversed_series.rolling(window, min_periods=window).min()
    elif stat == "max":
        result = reversed_series.rolling(window, min_periods=window).max()
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return result.iloc[::-1]
