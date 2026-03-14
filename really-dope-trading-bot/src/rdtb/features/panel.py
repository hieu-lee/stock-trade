from __future__ import annotations

import numpy as np
import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.utils import safe_divide


def build_feature_panel(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: TradingBotConfig,
    external_markets: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    company_metadata: pd.DataFrame | None = None,
    flow_history: pd.DataFrame | None = None,
    event_history: pd.DataFrame | None = None,
    persist_path=None,
) -> pd.DataFrame:
    symbol_panel = _build_symbol_features(prices, config)
    benchmark_panel = _build_benchmark_features(benchmarks, config)
    merged = symbol_panel.merge(benchmark_panel, on="date", how="left", suffixes=("", "_benchmark"))
    merged = _attach_external_features(merged, external_markets)
    merged = _attach_fundamental_features(merged, fundamentals, company_metadata)
    merged = _attach_flow_features(merged, flow_history)
    merged = _attach_event_features(merged, event_history)
    merged = _attach_breadth_features(merged)
    merged = _attach_cross_sectional_features(merged)
    merged = _attach_targets(merged, benchmark_panel, config)
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
    if persist_path is not None:
        merged.to_parquet(persist_path, index=False)
    return merged


def _build_symbol_features(prices: pd.DataFrame, config: TradingBotConfig) -> pd.DataFrame:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", group_keys=False)

    frame["prev_close"] = grouped["close"].shift(1)
    frame["ret_1d"] = grouped["close"].pct_change(1)
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["ret_10d"] = grouped["close"].pct_change(10)
    frame["ret_20d"] = grouped["close"].pct_change(20)
    frame["ret_40d"] = grouped["close"].pct_change(40)
    frame["gap_open"] = safe_divide(frame["open"], frame["prev_close"]) - 1.0
    frame["intraday_return"] = safe_divide(frame["close"], frame["open"]) - 1.0
    frame["range_pct"] = safe_divide(frame["high"], frame["low"]) - 1.0
    frame["turnover"] = frame["close"] * frame["volume"]
    frame["avg_volume_20d"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["avg_turnover_20d"] = grouped["turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    volume_std = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0))
    frame["volume_zscore_20d"] = safe_divide(frame["volume"] - frame["avg_volume_20d"], volume_std)
    for window in (10, 20, 50, 200):
        ma = grouped["close"].transform(lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
        frame[f"ma{window}"] = ma
        frame[f"distance_ma{window}"] = safe_divide(frame["close"], ma) - 1.0
    frame["volatility_10d"] = grouped["ret_1d"].transform(lambda s: s.rolling(10, min_periods=5).std(ddof=0))
    frame["volatility_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0))
    frame["rsi_14"] = grouped["close"].transform(_compute_rsi)
    fast_ema = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    slow_ema = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    frame["macd"] = fast_ema - slow_ema
    frame["macd_signal"] = frame.groupby("symbol")["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    frame["macd_diff"] = frame["macd"] - frame["macd_signal"]
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - frame["prev_close"]).abs(),
            (frame["low"] - frame["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["true_range"] = true_range
    atr = grouped["true_range"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    frame["atr_pct"] = safe_divide(atr, frame["close"])
    frame["days_since_listing"] = grouped.cumcount()
    frame["history_ready"] = frame["days_since_listing"] >= config.min_history_days
    return frame


def _build_benchmark_features(benchmarks: pd.DataFrame, config: TradingBotConfig) -> pd.DataFrame:
    benchmark_symbol = config.benchmark_symbols[0]
    frame = benchmarks.loc[benchmarks["symbol"] == benchmark_symbol].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["benchmark_ret_1d"] = frame["close"].pct_change(1)
    frame["benchmark_ret_5d"] = frame["close"].pct_change(5)
    frame["benchmark_ret_10d"] = frame["close"].pct_change(10)
    frame["benchmark_ret_20d"] = frame["close"].pct_change(20)
    frame["benchmark_gap_open"] = safe_divide(frame["open"], frame["close"].shift(1)) - 1.0
    frame["benchmark_intraday_return"] = safe_divide(frame["close"], frame["open"]) - 1.0
    frame["benchmark_vol_20d"] = frame["benchmark_ret_1d"].rolling(20, min_periods=5).std(ddof=0)
    benchmark_ma200 = frame["close"].rolling(200, min_periods=50).mean()
    frame["benchmark_distance_ma200"] = safe_divide(frame["close"], benchmark_ma200) - 1.0
    frame["benchmark_drawdown_252d"] = frame["close"] / frame["close"].rolling(252, min_periods=20).max() - 1.0
    frame["benchmark_forward_return_10d"] = safe_divide(frame["close"].shift(-10), frame["open"].shift(-1)) - 1.0
    frame["benchmark_forward_return_20d"] = safe_divide(frame["close"].shift(-20), frame["open"].shift(-1)) - 1.0
    frame["benchmark_forward_min_10d"] = safe_divide(_future_window_stat(frame["low"], 10, "min"), frame["open"].shift(-1)) - 1.0
    frame["target_regime"] = (
        (frame["benchmark_forward_return_20d"] > 0.03) & (frame["benchmark_forward_min_10d"] > -0.06)
    ).astype(float)
    frame.loc[frame["benchmark_forward_return_20d"].isna(), "target_regime"] = np.nan
    return frame[
        [
            "date",
            "benchmark_ret_1d",
            "benchmark_ret_5d",
            "benchmark_ret_10d",
            "benchmark_ret_20d",
            "benchmark_gap_open",
            "benchmark_intraday_return",
            "benchmark_vol_20d",
            "benchmark_distance_ma200",
            "benchmark_drawdown_252d",
            "benchmark_forward_return_10d",
            "benchmark_forward_return_20d",
            "benchmark_forward_min_10d",
            "target_regime",
        ]
    ]


def _attach_breadth_features(frame: pd.DataFrame) -> pd.DataFrame:
    breadth = (
        frame.groupby("date", as_index=False)
        .agg(
            breadth_above_ma50=("distance_ma50", lambda s: float((s > 0).mean())),
            breadth_above_ma200=("distance_ma200", lambda s: float((s > 0).mean())),
            breadth_positive_10d=("ret_10d", lambda s: float((s > 0).mean())),
            breadth_ret_20d=("ret_20d", "median"),
            breadth_turnover_20d=("avg_turnover_20d", "median"),
            quality_breadth=("fundamental_quality_score", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
            growth_breadth=("fundamental_growth_score", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
            value_breadth=("fundamental_value_score", "median"),
            event_breadth_20d=("recent_event_count_20d", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
            dividend_breadth_252d=("recent_dividend_event_count_252d", lambda s: float((s > 0).mean()) if s.notna().any() else np.nan),
        )
        .sort_values("date")
    )
    return frame.merge(breadth, on="date", how="left")


def _attach_cross_sectional_features(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.copy()
    panel["relative_strength_10d"] = panel["ret_10d"] - panel["benchmark_ret_10d"]
    panel["relative_strength_20d"] = panel["ret_20d"] - panel["benchmark_ret_20d"]
    for column in [
        "ret_20d",
        "relative_strength_20d",
        "distance_ma50",
        "volume_zscore_20d",
        "volatility_20d",
        "foreign_flow_score",
        "order_pressure_score",
        "event_score",
    ]:
        panel[f"{column}_rank"] = panel.groupby("date")[column].rank(pct=True, na_option="keep")
        panel[f"{column}_zscore"] = panel.groupby("date")[column].transform(_zscore)
    beta_parts: list[pd.Series] = []
    for _, part in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        covariance = part["ret_1d"].rolling(60, min_periods=20).cov(part["benchmark_ret_1d"])
        benchmark_variance = part["benchmark_ret_1d"].rolling(60, min_periods=20).var(ddof=0)
        beta_parts.append(covariance.div(benchmark_variance))
    panel["beta_60d"] = pd.concat(beta_parts).sort_index() if beta_parts else np.nan
    return panel


def _attach_flow_features(frame: pd.DataFrame, flow_history: pd.DataFrame | None) -> pd.DataFrame:
    panel = frame.copy()
    if flow_history is None or flow_history.empty:
        for column in [
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
            "foreign_net_qty",
            "foreign_net_value",
            "foreign_buy_ratio",
            "foreign_sell_ratio",
            "foreign_net_ratio",
            "foreign_value_net_ratio",
            "foreign_room_ratio",
            "order_quantity_imbalance",
            "order_count_imbalance",
            "putthrough_ratio",
            "deal_ratio",
            "market_cap_turnover_ratio",
            "foreign_flow_score",
            "order_pressure_score",
            "foreign_flow_5d",
            "foreign_flow_20d",
            "room_change_5d",
        ]:
            panel[column] = np.nan
        return panel

    flow = flow_history.copy().drop(columns=["open", "high", "low", "close", "volume", "adj_close"], errors="ignore")
    flow["date"] = pd.to_datetime(flow["date"])
    flow = flow.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel = panel.merge(flow, on=["symbol", "date"], how="left")
    panel["foreign_net_qty"] = panel["foreign_buy_qty"] - panel["foreign_sell_qty"]
    panel["foreign_net_value"] = panel["foreign_buy_value"] - panel["foreign_sell_value"]
    volume_safe = panel["volume"].replace(0, np.nan)
    total_value_safe = (panel["close"] * panel["volume"]).replace(0, np.nan)
    shares_safe = panel["shares_outstanding"].replace(0, np.nan)
    panel["foreign_buy_ratio"] = panel["foreign_buy_qty"] / volume_safe
    panel["foreign_sell_ratio"] = panel["foreign_sell_qty"] / volume_safe
    panel["foreign_net_ratio"] = panel["foreign_net_qty"] / volume_safe
    panel["foreign_value_net_ratio"] = panel["foreign_net_value"] / total_value_safe
    panel["foreign_room_ratio"] = panel["foreign_room"] / shares_safe
    panel["order_quantity_imbalance"] = safe_divide(panel["buy_order_qty"] - panel["sell_order_qty"], panel["buy_order_qty"] + panel["sell_order_qty"])
    panel["order_count_imbalance"] = safe_divide(panel["buy_order_count"] - panel["sell_order_count"], panel["buy_order_count"] + panel["sell_order_count"])
    panel["putthrough_ratio"] = panel["putthrough_volume"] / volume_safe
    panel["deal_ratio"] = panel["deal_volume"] / volume_safe
    panel["market_cap_turnover_ratio"] = total_value_safe / panel["market_cap"].replace(0, np.nan)

    grouped = panel.groupby("symbol", group_keys=False)
    panel["foreign_flow_5d"] = grouped["foreign_net_ratio"].transform(lambda s: s.rolling(5, min_periods=2).sum())
    panel["foreign_flow_20d"] = grouped["foreign_net_ratio"].transform(lambda s: s.rolling(20, min_periods=5).sum())
    panel["room_change_5d"] = grouped["foreign_room_ratio"].transform(lambda s: s.diff(5))
    panel["foreign_flow_score"] = (
        panel["foreign_net_ratio"].fillna(0.0) * 0.35
        + panel["foreign_value_net_ratio"].fillna(0.0) * 0.35
        + panel["foreign_flow_5d"].fillna(0.0) * 0.20
        - panel["putthrough_ratio"].fillna(0.0) * 0.10
    )
    panel["order_pressure_score"] = (
        panel["order_quantity_imbalance"].fillna(0.0) * 0.6
        + panel["order_count_imbalance"].fillna(0.0) * 0.2
        + panel["deal_ratio"].fillna(0.0) * 0.2
    )
    return panel


def _attach_event_features(frame: pd.DataFrame, event_history: pd.DataFrame | None) -> pd.DataFrame:
    panel = frame.copy()
    if event_history is None or event_history.empty:
        for column in [
            "recent_event_count_20d",
            "recent_event_count_60d",
            "recent_dividend_event_count_252d",
            "recent_issue_event_count_252d",
            "recent_listing_event_count_252d",
            "days_since_last_event",
            "days_since_last_dividend_event",
            "days_since_last_issue_event",
            "latest_dividend_value",
            "latest_issue_ratio",
            "upcoming_record_days",
            "upcoming_exright_days",
            "event_score",
        ]:
            panel[column] = np.nan
        return panel

    events = event_history.copy()
    events["symbol"] = events["symbol"].astype(str)
    events["available_date"] = pd.to_datetime(events["available_date"])
    for column in ["issue_date", "record_date", "exright_date"]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    events["is_dividend"] = events["event_code"].astype(str).eq("DIV").astype(float)
    events["is_issue"] = events["event_code"].astype(str).eq("ISS").astype(float)
    events["is_listing"] = events["event_code"].astype(str).isin(["AIS", "NLIS"]).astype(float)

    merged_parts: list[pd.DataFrame] = []
    for symbol, left_part in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        right_part = events.loc[events["symbol"] == symbol].sort_values("available_date").reset_index(drop=True)
        symbol_panel = left_part.copy().sort_values("date").reset_index(drop=True)
        if right_part.empty:
            merged_parts.append(_ensure_event_columns(symbol_panel))
            continue

        daily_events = (
            right_part.groupby("available_date", as_index=False)
            .agg(
                event_count=("event_code", "size"),
                dividend_event_count=("is_dividend", "sum"),
                issue_event_count=("is_issue", "sum"),
                listing_event_count=("is_listing", "sum"),
                latest_dividend_value=("value", lambda s: s[right_part.loc[s.index, "is_dividend"] > 0].max() if not s.empty else np.nan),
                latest_issue_ratio=("ratio", lambda s: s[right_part.loc[s.index, "is_issue"] > 0].max() if not s.empty else np.nan),
            )
            .rename(columns={"available_date": "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        symbol_panel = symbol_panel.merge(daily_events, on="date", how="left")
        symbol_panel[["event_count", "dividend_event_count", "issue_event_count", "listing_event_count"]] = (
            symbol_panel[["event_count", "dividend_event_count", "issue_event_count", "listing_event_count"]].fillna(0.0)
        )
        symbol_panel["latest_dividend_value"] = symbol_panel["latest_dividend_value"].ffill()
        symbol_panel["latest_issue_ratio"] = symbol_panel["latest_issue_ratio"].ffill()

        symbol_panel["recent_event_count_20d"] = symbol_panel["event_count"].rolling(20, min_periods=1).sum()
        symbol_panel["recent_event_count_60d"] = symbol_panel["event_count"].rolling(60, min_periods=1).sum()
        symbol_panel["recent_dividend_event_count_252d"] = symbol_panel["dividend_event_count"].rolling(252, min_periods=1).sum()
        symbol_panel["recent_issue_event_count_252d"] = symbol_panel["issue_event_count"].rolling(252, min_periods=1).sum()
        symbol_panel["recent_listing_event_count_252d"] = symbol_panel["listing_event_count"].rolling(252, min_periods=1).sum()

        last_event_date = symbol_panel["date"].where(symbol_panel["event_count"] > 0).ffill()
        last_dividend_date = symbol_panel["date"].where(symbol_panel["dividend_event_count"] > 0).ffill()
        last_issue_date = symbol_panel["date"].where(symbol_panel["issue_event_count"] > 0).ffill()
        symbol_panel["days_since_last_event"] = (symbol_panel["date"] - last_event_date).dt.days
        symbol_panel["days_since_last_dividend_event"] = (symbol_panel["date"] - last_dividend_date).dt.days
        symbol_panel["days_since_last_issue_event"] = (symbol_panel["date"] - last_issue_date).dt.days

        symbol_panel["upcoming_record_days"] = _days_until_announced_target(
            symbol_panel["date"],
            right_part["available_date"],
            right_part["record_date"],
        )
        symbol_panel["upcoming_exright_days"] = _days_until_announced_target(
            symbol_panel["date"],
            right_part["available_date"],
            right_part["exright_date"],
        )
        symbol_panel["event_score"] = (
            symbol_panel["recent_dividend_event_count_252d"].fillna(0.0) * 0.20
            - symbol_panel["recent_issue_event_count_252d"].fillna(0.0) * 0.20
            - symbol_panel["recent_listing_event_count_252d"].fillna(0.0) * 0.10
            + symbol_panel["latest_dividend_value"].fillna(0.0) / 10000.0
        )
        merged_parts.append(_ensure_event_columns(symbol_panel))

    return pd.concat(merged_parts, ignore_index=True)


def _attach_fundamental_features(
    frame: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    company_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    panel = frame.copy()
    if company_metadata is not None and not company_metadata.empty:
        metadata = company_metadata.copy()
        metadata["sector_name"] = metadata.get("industry_level3", pd.Series(index=metadata.index, dtype=object)).fillna(
            metadata.get("industry_level2", pd.Series(index=metadata.index, dtype=object))
        )
        panel = panel.merge(metadata[["symbol", "sector_name"]], on="symbol", how="left")
    else:
        panel["sector_name"] = np.nan

    if fundamentals is None or fundamentals.empty:
        return _attach_sector_features(_ensure_fundamental_columns(panel))

    fundamentals_frame = fundamentals.copy()
    fundamentals_frame["symbol"] = fundamentals_frame["symbol"].astype(str)
    fundamentals_frame["available_date"] = pd.to_datetime(fundamentals_frame["available_date"])
    fundamentals_frame["report_period_end"] = pd.to_datetime(fundamentals_frame["report_period_end"])
    fundamentals_frame = fundamentals_frame.sort_values(["symbol", "available_date"]).reset_index(drop=True)
    merged_parts: list[pd.DataFrame] = []
    additional_columns = [column for column in fundamentals_frame.columns if column not in {"symbol", "available_date"}]
    for symbol, left_part in panel.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        right_part = fundamentals_frame.loc[fundamentals_frame["symbol"] == symbol].drop(columns=["symbol"]).sort_values("available_date")
        if right_part.empty:
            merged_part = left_part.copy()
            for column in additional_columns:
                merged_part[column] = np.nan
        else:
            merged_part = pd.merge_asof(
                left_part.sort_values("date"),
                right_part,
                left_on="date",
                right_on="available_date",
                direction="backward",
            )
        merged_parts.append(merged_part)
    panel = pd.concat(merged_parts, ignore_index=True).drop(columns=["available_date"], errors="ignore")
    panel = _ensure_fundamental_columns(panel)
    panel["fundamental_freshness_days"] = (pd.to_datetime(panel["date"]) - pd.to_datetime(panel["report_period_end"])).dt.days
    panel["log_revenue_bn_vnd"] = np.log1p(panel["revenue_bn_vnd"].clip(lower=0))
    panel["log_market_cap_bn_vnd"] = np.log1p(panel["market_cap_bn_vnd"].clip(lower=0))
    panel["fundamental_growth_score"] = (
        panel.groupby("date")["revenue_yoy_pct"].transform(_zscore).fillna(0.0) * 0.4
        + panel.groupby("date")["parent_profit_yoy_pct"].transform(_zscore).fillna(0.0) * 0.4
        + panel.groupby("date")["roe_pct"].transform(_zscore).fillna(0.0) * 0.2
    )
    panel["fundamental_quality_score"] = (
        panel.groupby("date")["roe_pct"].transform(_zscore).fillna(0.0) * 0.25
        + panel.groupby("date")["roa_pct"].transform(_zscore).fillna(0.0) * 0.20
        + panel.groupby("date")["gross_profit_margin_pct"].transform(_zscore).fillna(0.0) * 0.20
        + panel.groupby("date")["current_ratio"].transform(_zscore).fillna(0.0) * 0.15
        - panel.groupby("date")["debt_to_equity"].transform(_zscore).fillna(0.0) * 0.20
    )
    panel["fundamental_value_score"] = (
        _inverse_rank_by_date(panel, "pe_ratio") * 0.4
        + _inverse_rank_by_date(panel, "pb_ratio") * 0.3
        + _inverse_rank_by_date(panel, "ps_ratio") * 0.3
    )
    return _attach_sector_features(panel)


def _attach_sector_features(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.copy()
    if "sector_name" not in panel.columns or panel["sector_name"].isna().all():
        panel["sector_ret_20d"] = np.nan
        panel["sector_breadth_above_ma50"] = np.nan
        panel["sector_turnover_20d"] = np.nan
        panel["sector_ret_20d_rank"] = np.nan
        return panel
    sector_daily = (
        panel.groupby(["date", "sector_name"], as_index=False)
        .agg(
            sector_ret_20d=("ret_20d", "median"),
            sector_breadth_above_ma50=("distance_ma50", lambda s: float((s > 0).mean())),
            sector_turnover_20d=("avg_turnover_20d", "median"),
        )
        .sort_values(["date", "sector_name"])
    )
    sector_daily["sector_ret_20d_rank"] = sector_daily.groupby("date")["sector_ret_20d"].rank(pct=True, na_option="keep")
    return panel.merge(sector_daily, on=["date", "sector_name"], how="left")


def _ensure_fundamental_columns(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.copy()
    required_columns = [
        "revenue_yoy_pct",
        "revenue_bn_vnd",
        "parent_profit_bn_vnd",
        "parent_profit_yoy_pct",
        "operating_profit_vnd",
        "profit_before_tax_vnd",
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "interest_coverage",
        "financial_leverage",
        "asset_turnover",
        "inventory_turnover",
        "ebit_margin_pct",
        "gross_profit_margin_pct",
        "net_profit_margin_pct",
        "roe_pct",
        "roic_pct",
        "roa_pct",
        "dividend_yield_pct",
        "market_cap_bn_vnd",
        "outstanding_share_mil",
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "pcf_ratio",
        "eps_vnd",
        "bvps_vnd",
        "operating_cash_flow_vnd",
        "capex_vnd",
        "investing_cash_flow_vnd",
        "financing_cash_flow_vnd",
        "report_period_end",
        "fundamental_growth_score",
        "fundamental_quality_score",
        "fundamental_value_score",
        "fundamental_freshness_days",
        "log_revenue_bn_vnd",
        "log_market_cap_bn_vnd",
    ]
    for column in required_columns:
        if column not in panel.columns:
            panel[column] = np.nan
    return panel


def _ensure_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.copy()
    required_columns = [
        "recent_event_count_20d",
        "recent_event_count_60d",
        "recent_dividend_event_count_252d",
        "recent_issue_event_count_252d",
        "recent_listing_event_count_252d",
        "days_since_last_event",
        "days_since_last_dividend_event",
        "days_since_last_issue_event",
        "latest_dividend_value",
        "latest_issue_ratio",
        "upcoming_record_days",
        "upcoming_exright_days",
        "event_score",
    ]
    for column in required_columns:
        if column not in panel.columns:
            panel[column] = np.nan
    return panel


def _days_until_announced_target(dates: pd.Series, available_dates: pd.Series, target_dates: pd.Series) -> pd.Series:
    known_available = pd.to_datetime(available_dates).to_list()
    targets = pd.to_datetime(target_dates).to_list()
    values: list[float] = []
    for current_date in pd.to_datetime(dates):
        best_days: float | None = None
        for available_date, target_date in zip(known_available, targets):
            if pd.isna(available_date) or pd.isna(target_date):
                continue
            if available_date <= current_date < target_date:
                delta = float((target_date - current_date).days)
                best_days = delta if best_days is None else min(best_days, delta)
        values.append(np.nan if best_days is None else best_days)
    return pd.Series(values, index=dates.index, dtype=float)


def _inverse_rank_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    ranks = frame.groupby("date")[column].rank(pct=True, na_option="keep", ascending=True)
    return 1.0 - ranks


def _attach_external_features(frame: pd.DataFrame, external_markets: pd.DataFrame | None) -> pd.DataFrame:
    if external_markets is None or external_markets.empty:
        return frame
    external_frame = _build_external_feature_frame(external_markets)
    merged = pd.merge_asof(
        frame.sort_values("date"),
        external_frame.drop(columns=["date"], errors="ignore").sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    return merged.drop(columns=["available_date"], errors="ignore")


def _build_external_feature_frame(external_markets: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "SPY": "spy",
        "QQQ": "qqq",
        "EEM": "eem",
        "FXI": "fxi",
        "TLT": "tlt",
        "GLD": "gld",
        "UUP": "uup",
    }
    merged: pd.DataFrame | None = None
    for symbol, alias in aliases.items():
        part = external_markets.loc[external_markets["symbol"] == symbol].copy()
        if part.empty:
            continue
        part = part.sort_values("date").reset_index(drop=True)
        close = part["close"]
        ma50 = close.rolling(50, min_periods=10).mean()
        feature_frame = part[["date"]].copy()
        feature_frame[f"{alias}_ret_1d"] = close.pct_change(1)
        feature_frame[f"{alias}_ret_5d"] = close.pct_change(5)
        feature_frame[f"{alias}_ret_20d"] = close.pct_change(20)
        feature_frame[f"{alias}_distance_ma50"] = safe_divide(close, ma50) - 1.0
        merged = feature_frame if merged is None else merged.merge(feature_frame, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["date", "available_date"])
    equity_cols_5d = [column for column in ["spy_ret_5d", "qqq_ret_5d", "eem_ret_5d", "fxi_ret_5d"] if column in merged.columns]
    equity_cols_20d = [column for column in ["spy_ret_20d", "qqq_ret_20d", "eem_ret_20d", "fxi_ret_20d"] if column in merged.columns]
    defensive_cols_20d = [column for column in ["tlt_ret_20d", "gld_ret_20d", "uup_ret_20d"] if column in merged.columns]
    trend_cols = [column for column in ["spy_distance_ma50", "qqq_distance_ma50", "eem_distance_ma50", "fxi_distance_ma50"] if column in merged.columns]
    merged["global_equity_momentum_5d"] = merged[equity_cols_5d].mean(axis=1) if equity_cols_5d else np.nan
    merged["global_equity_momentum_20d"] = merged[equity_cols_20d].mean(axis=1) if equity_cols_20d else np.nan
    merged["global_defensive_momentum_20d"] = merged[defensive_cols_20d].mean(axis=1) if defensive_cols_20d else np.nan
    merged["global_equity_trend_ma50"] = merged[trend_cols].mean(axis=1) if trend_cols else np.nan
    merged["global_risk_on_score"] = merged["global_equity_momentum_20d"] - merged["global_defensive_momentum_20d"]
    merged["available_date"] = pd.to_datetime(merged["date"]) + pd.Timedelta(days=1)
    return merged.sort_values("available_date").reset_index(drop=True)


def _attach_targets(frame: pd.DataFrame, benchmark_panel: pd.DataFrame, config: TradingBotConfig) -> pd.DataFrame:
    panel = frame.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    grouped = panel.groupby("symbol", group_keys=False)
    panel["open_next"] = grouped["open"].shift(-1)
    panel["entry_price_next_open"] = panel["open_next"]
    panel["forward_return_5d"] = safe_divide(grouped["close"].shift(-5), panel["entry_price_next_open"]) - 1.0
    panel["forward_return_10d"] = safe_divide(grouped["close"].shift(-10), panel["entry_price_next_open"]) - 1.0
    panel["forward_return_20d"] = safe_divide(grouped["close"].shift(-20), panel["entry_price_next_open"]) - 1.0
    panel["forward_min_return_10d"] = safe_divide(
        grouped["low"].transform(lambda s: _future_window_stat(s, 10, "min")),
        panel["entry_price_next_open"],
    ) - 1.0
    panel["forward_max_return_20d"] = safe_divide(
        grouped["high"].transform(lambda s: _future_window_stat(s, 20, "max")),
        panel["entry_price_next_open"],
    ) - 1.0
    if "benchmark_forward_return_10d" not in panel.columns:
        benchmark_targets = benchmark_panel[
            ["date", "benchmark_forward_return_10d", "benchmark_forward_return_20d", "benchmark_forward_min_10d", "target_regime"]
        ]
        panel = panel.merge(benchmark_targets, on="date", how="left")
    panel["forward_excess_return_10d"] = panel["forward_return_10d"] - panel["benchmark_forward_return_10d"]
    panel["forward_excess_return_20d"] = panel["forward_return_20d"] - panel["benchmark_forward_return_20d"]
    panel["target_alpha_rank_10d"] = panel.groupby("date")["forward_excess_return_10d"].rank(pct=True, na_option="keep")
    panel["target_alpha_rank_20d"] = panel.groupby("date")["forward_excess_return_20d"].rank(pct=True, na_option="keep")
    panel["target_alpha_blend"] = panel["target_alpha_rank_10d"] * 0.35 + panel["target_alpha_rank_20d"] * 0.65
    panel["target_alpha_class"] = (panel["target_alpha_blend"] >= config.alpha_target_quantile).astype(float)
    panel.loc[panel["target_alpha_blend"].isna(), "target_alpha_class"] = np.nan
    panel["target_downside"] = (panel["forward_min_return_10d"] <= -config.stop_loss_pct).astype(float)
    panel.loc[panel["forward_min_return_10d"].isna(), "target_downside"] = np.nan
    panel["is_trainable"] = (
        panel["history_ready"]
        & panel["target_alpha_class"].notna()
        & panel["target_downside"].notna()
        & panel["target_regime"].notna()
    )
    return panel


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _future_window_stat(series: pd.Series, window: int, stat: str) -> pd.Series:
    shifted = series.shift(-1)
    reversed_series = shifted.iloc[::-1]
    if stat == "min":
        result = reversed_series.rolling(window, min_periods=window).min()
    elif stat == "max":
        result = reversed_series.rolling(window, min_periods=window).max()
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported future statistic `{stat}`.")
    return result.iloc[::-1]


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std
