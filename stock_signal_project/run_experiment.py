from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VNSTOCK_REPO = PROJECT_ROOT / "vnstock"
OUTPUT_DIR = PROJECT_ROOT / "stock_signal_project" / "outputs"
CACHE_DIR = OUTPUT_DIR / "history_cache"
TODAY = date.today()
START_DATE = (TODAY - timedelta(days=365 * 15 + 4)).isoformat()
TODAY_STR = TODAY.isoformat()
REQUESTED_SYMBOLS = {
    "ACB": "Ngân hàng TMCP Á Châu",
    "BCM": "Tổng Công ty Đầu tư & Phát triển Công nghiệp Becamex",
    "BID": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam",
    "BVH": "Tập đoàn Bảo Việt",
    "CTG": "Ngân hàng TMCP Công Thương Việt Nam",
    "FPT": "CTCP FPT",
    "GAS": "Tổng công ty Khí Việt Nam - CTCP",
    "GVR": "Tập đoàn Công nghiệp Cao su Việt Nam",
    "HDB": "Ngân hàng TMCP Phát triển TP. HCM",
    "HPG": "CTCP Tập đoàn Hòa Phát",
    "LPB": "Ngân hàng TMCP Lộc Phát",
    "MBB": "Ngân hàng TMCP Quân đội",
    "MSN": "Tập đoàn Masan",
    "MWG": "CTCP Đầu tư Thế Giới Di Động",
    "PLX": "Tập đoàn Xăng dầu Việt Nam",
    "SAB": "Tổng CTCP Bia - Rượu - Nước giải khát Sài Gòn",
    "SHB": "Ngân hàng TMCP Sài Gòn - Hà Nội",
    "SSB": "Ngân hàng TMCP Đông Nam Á",
    "SSI": "CTCP Chứng khoán SSI",
    "STB": "Ngân hàng TMCP Sài Gòn Thương Tín",
    "TCB": "Ngân hàng TMCP Kỹ thương Việt Nam",
    "TPB": "Ngân hàng TMCP Tiên Phong",
    "VCB": "Ngân hàng TMCP Ngoại thương Việt Nam",
    "VHM": "CTCP Vinhomes",
    "VIB": "Ngân hàng TMCP Quốc tế Việt Nam",
    "VIC": "Tập đoàn Vingroup - CTCP",
    "VJC": "CTCP Hàng không VietJet",
    "VNM": "CTCP Sữa Việt Nam (Vinamilk)",
    "VPB": "Ngân hàng TMCP Việt Nam Thịnh Vượng",
    "VRE": "CTCP Vincom Retail",
}
REQUESTED_SYMBOL_LIST = list(REQUESTED_SYMBOLS)
UNIVERSE_LABEL = "Requested 30-symbol subset"
UNIVERSE_SOURCE = "Custom symbol list"
HORIZON_DAYS = 20
DEFAULT_TARGET_RETURN = 0.06
MIN_LABELED_ROWS_PER_SYMBOL = 252
MAX_TUNING_ROWS = 200_000
REQUEST_DELAY_SECONDS = 3.5
MA_WINDOWS = [10, 25, 50, 100]
TARGET_SWEEP = [0.06]
BASE_FEATURE_COLUMNS = [
    "ma_gap_10",
    "ma_gap_25",
    "ma_gap_50",
    "ma_gap_100",
    "resistance_gap",
    "support_gap",
    "price_change_1d",
    "volume_ratio_10",
    "volume_ratio_25",
    "volume_ratio_50",
    "volume_ratio_100",
    "bollinger_percent_b",
    "rsi_14",
]
EXTENDED_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_10d",
    "volatility_20d",
    "volatility_60d",
    "atr_ratio_14",
    "intraday_range_ratio",
    "volume_zscore_20",
    "breakout_gap_high_20",
    "breakout_gap_high_60",
    "breakout_gap_high_120",
    "breakout_gap_low_20",
    "breakout_gap_low_60",
    "breakout_gap_low_120",
    "drawdown_20",
    "drawdown_60",
]
TOP_K_VALUES = [3, 5, 10]
TRAIN_RATIO = 0.65
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.20

if str(VNSTOCK_REPO) not in sys.path:
    sys.path.insert(0, str(VNSTOCK_REPO))

from vnstock import Listing, Quote  # noqa: E402

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class ExperimentResult:
    scenario_name: str
    feature_set_name: str
    target_return: float
    model_name: str
    best_params: Dict[str, object]
    best_cv_score: float
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    cv_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    holdout_metrics: Dict[str, float]
    holdout_predictions: pd.DataFrame
    exchange_holdout_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    latest_signals: pd.DataFrame
    threshold: float
    policy_metrics: pd.DataFrame


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    target_return: float
    feature_columns: List[str]
    feature_set_name: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], object]
    param_grid: Dict[str, Iterable[object]]


class FenwickTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=int)

    def add(self, index: int, value: int = 1) -> None:
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        result = 0
        while index > 0:
            result += int(self.tree[index])
            index -= index & -index
        return result

    def total(self) -> int:
        return self.prefix_sum(self.size)

    def find_by_order(self, order: int) -> int:
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and self.tree[candidate] < order:
                order -= int(self.tree[candidate])
                index = candidate
            bit >>= 1
        return index + 1


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator, denominator, fill_value=np.nan):
    if isinstance(numerator, pd.Series):
        numerator = numerator.astype(float)
    if isinstance(denominator, pd.Series):
        denominator = denominator.astype(float)

    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_arr, fill_value, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            numerator_arr,
            denominator_arr,
            out=result,
            where=np.isfinite(denominator_arr) & (denominator_arr != 0),
        )

    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


def get_universe() -> pd.DataFrame:
    listing = Listing(source="vci", show_log=False)
    exchange_df = listing.symbols_by_exchange()
    universe = pd.DataFrame({"symbol": REQUESTED_SYMBOL_LIST})
    universe = universe.merge(exchange_df[["symbol", "exchange", "organ_name", "icb_code2"]], on="symbol", how="left")
    universe["organ_name"] = universe["symbol"].map(REQUESTED_SYMBOLS).fillna(universe["organ_name"])
    universe = universe.sort_values(["exchange", "symbol"]).reset_index(drop=True)
    return universe


def fetch_history(symbol: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{symbol}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df["symbol"] = symbol
        return df

    quote = Quote(symbol=symbol, source="vci", show_log=False)
    df = quote.history(start=START_DATE, end=TODAY_STR, interval="1D").copy()
    if df.empty:
        raise ValueError("No history returned")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    df["symbol"] = symbol
    return df


def fetch_all_histories(universe: pd.DataFrame) -> tuple[List[pd.DataFrame], pd.DataFrame]:
    histories: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []
    symbols = universe["symbol"].tolist()

    for completed, symbol in enumerate(symbols, start=1):
        try:
            cached = (CACHE_DIR / f"{symbol}.csv").exists()
            histories.append(fetch_history(symbol))
            if not cached:
                time.sleep(REQUEST_DELAY_SECONDS)
        except Exception as exc:  # pragma: no cover
            failures.append({"symbol": symbol, "error": str(exc)})
        if completed % 25 == 0 or completed == len(symbols):
            print(f"Fetched {completed}/{len(symbols)} symbols", flush=True)

    failures_df = pd.DataFrame(failures)
    return histories, failures_df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi = rsi.where(avg_loss.ne(0), 100)
    rsi = rsi.where(avg_gain.ne(0), 0)
    return rsi


def compute_support_resistance_gaps(df: pd.DataFrame) -> pd.DataFrame:
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    high_levels = sorted(np.unique(highs))
    low_levels = sorted(np.unique(lows))
    high_tree = FenwickTree(len(high_levels))
    low_tree = FenwickTree(len(low_levels))

    resistance_gaps: List[float] = []
    support_gaps: List[float] = []

    for current_high, current_low, current_close in zip(highs, lows, closes):
        high_rank = bisect_left(high_levels, current_high) + 1
        low_rank = bisect_left(low_levels, current_low) + 1
        high_tree.add(high_rank)
        low_tree.add(low_rank)

        position_right = bisect_right(high_levels, current_close)
        seen_less_equal = high_tree.prefix_sum(position_right)
        if seen_less_equal == high_tree.total():
            resistance_gaps.append(0.0)
        else:
            next_rank = high_tree.find_by_order(seen_less_equal + 1)
            resistance = high_levels[next_rank - 1]
            if resistance <= 0:
                resistance_gaps.append(0.0)
            else:
                resistance_gaps.append(abs(resistance - current_close) / resistance)

        position_left = bisect_left(low_levels, current_close)
        seen_less = low_tree.prefix_sum(position_left)
        if seen_less == 0:
            support_gaps.append(0.0)
        else:
            prev_rank = low_tree.find_by_order(seen_less)
            support = low_levels[prev_rank - 1]
            if support <= 0:
                support_gaps.append(0.0)
            else:
                support_gaps.append(abs(support - current_close) / support)

    df["resistance_gap"] = resistance_gaps
    df["support_gap"] = support_gaps
    return df


def compute_future_peak(high: pd.Series, horizon: int) -> pd.Series:
    values = high.to_numpy(dtype=float)
    future_peak = np.full(len(values), np.nan)
    for idx in range(len(values) - horizon):
        future_peak[idx] = values[idx + 1 : idx + 1 + horizon].max()
    return pd.Series(future_peak, index=high.index)


def make_target_column(target_return: float) -> str:
    return f"target_profit_4w_{int(round(target_return * 100)):02d}pct"


def engineer_features(history: pd.DataFrame, target_return: float) -> pd.DataFrame:
    df = history.copy()
    previous_close = df["close"].shift(1)

    for window in MA_WINDOWS:
        ma = df["close"].rolling(window).mean()
        avg_volume = df["volume"].rolling(window).mean()
        df[f"ma_gap_{window}"] = safe_divide(ma - df["close"], ma)
        df[f"volume_ratio_{window}"] = safe_divide(df["volume"], avg_volume)

    df = compute_support_resistance_gaps(df)
    df["price_change_1d"] = safe_divide(df["close"] - previous_close, previous_close)

    rolling_mean_20 = df["close"].rolling(20).mean()
    rolling_std_20 = df["close"].rolling(20).std(ddof=0)
    bollinger_upper = rolling_mean_20 + 2 * rolling_std_20
    bollinger_lower = rolling_mean_20 - 2 * rolling_std_20
    band_width = bollinger_upper - bollinger_lower
    df["bollinger_percent_b"] = safe_divide(df["close"] - bollinger_lower, band_width)

    for window in [5, 10, 20, 60]:
        df[f"return_{window}d"] = df["close"].pct_change(window)

    close_return = df["close"].pct_change()
    for window in [10, 20, 60]:
        df[f"volatility_{window}d"] = close_return.rolling(window).std(ddof=0)

    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    df["atr_ratio_14"] = safe_divide(atr_14, df["close"])
    df["intraday_range_ratio"] = safe_divide(df["high"] - df["low"], df["close"])

    volume_mean_20 = df["volume"].rolling(20).mean()
    volume_std_20 = df["volume"].rolling(20).std(ddof=0)
    df["volume_zscore_20"] = safe_divide(df["volume"] - volume_mean_20, volume_std_20)

    for window in [20, 60, 120]:
        prior_high = df["high"].shift(1).rolling(window).max()
        prior_low = df["low"].shift(1).rolling(window).min()
        df[f"breakout_gap_high_{window}"] = safe_divide(prior_high - df["close"], prior_high)
        df[f"breakout_gap_low_{window}"] = safe_divide(df["close"] - prior_low, df["close"])

    for window in [20, 60]:
        rolling_peak = df["close"].shift(1).rolling(window).max()
        df[f"drawdown_{window}"] = safe_divide(df["close"] - rolling_peak, rolling_peak)

    df["rsi_14"] = compute_rsi(df["close"], window=14)
    df["future_peak_price"] = compute_future_peak(df["high"], HORIZON_DAYS)
    df["future_peak_return"] = safe_divide(df["future_peak_price"] - df["close"], df["close"])
    target_column = make_target_column(target_return)
    df[target_column] = (df["future_peak_return"] >= target_return).astype(float)
    df[EXTENDED_FEATURE_COLUMNS + ["future_peak_price", "future_peak_return"]] = df[
        EXTENDED_FEATURE_COLUMNS + ["future_peak_price", "future_peak_return"]
    ].replace([np.inf, -np.inf], np.nan)
    for feature in EXTENDED_FEATURE_COLUMNS:
        df[feature] = df[feature].clip(-10.0, 10.0)
    return df


def build_dataset(
    universe: pd.DataFrame,
    target_return: float,
    feature_columns: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    histories, failures_df = fetch_all_histories(universe)
    all_histories: List[pd.DataFrame] = []
    all_features: List[pd.DataFrame] = []
    target_column = make_target_column(target_return)

    for history in histories:
        enriched = history.merge(universe, on="symbol", how="left")
        all_histories.append(enriched)
        all_features.append(engineer_features(enriched, target_return=target_return))

    history_df = pd.concat(all_histories, ignore_index=True)
    features_df = pd.concat(all_features, ignore_index=True)
    labeled = features_df.dropna(subset=feature_columns + ["future_peak_price"]).copy()
    labeled[target_column] = labeled[target_column].astype(int)

    labeled_counts = labeled.groupby("symbol").size().rename("labeled_rows").reset_index()
    eligible_symbols = labeled_counts.loc[labeled_counts["labeled_rows"] >= MIN_LABELED_ROWS_PER_SYMBOL, "symbol"]
    labeled = labeled.loc[labeled["symbol"].isin(eligible_symbols)].copy()

    latest_rows = []
    for symbol in sorted(eligible_symbols):
        symbol_frame = features_df.loc[features_df["symbol"] == symbol].dropna(subset=feature_columns).copy()
        if not symbol_frame.empty:
            latest_rows.append(symbol_frame.iloc[-1])
    latest_df = pd.DataFrame(latest_rows).reset_index(drop=True)

    return history_df, labeled, latest_df, failures_df


def build_time_splits(dataset: pd.DataFrame, n_splits: int = 5) -> List[tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(dataset["time"].dt.normalize().unique()))
    fold_sizes = np.full(n_splits + 1, len(unique_dates) // (n_splits + 1), dtype=int)
    fold_sizes[: len(unique_dates) % (n_splits + 1)] += 1
    boundaries = np.cumsum(fold_sizes)

    splits: List[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        train_end = boundaries[fold_idx]
        test_end = boundaries[fold_idx + 1]
        train_dates = unique_dates[:train_end]
        test_dates = unique_dates[train_end:test_end]
        train_mask = dataset["time"].dt.normalize().isin(train_dates)
        test_mask = dataset["time"].dt.normalize().isin(test_dates)
        splits.append((dataset.index[train_mask].to_numpy(), dataset.index[test_mask].to_numpy()))
    return splits


def score_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = math.nan
        metrics["average_precision"] = math.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def classify_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float) >= float(threshold)).astype(int)


def split_dataset_by_date(
    dataset: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(dataset["time"].dt.normalize().unique()))
    train_cutoff = max(1, int(len(unique_dates) * train_ratio))
    validation_cutoff = max(train_cutoff + 1, int(len(unique_dates) * (train_ratio + validation_ratio)))
    validation_cutoff = min(validation_cutoff, len(unique_dates) - 1)

    train_dates = unique_dates[:train_cutoff]
    validation_dates = unique_dates[train_cutoff:validation_cutoff]
    test_dates = unique_dates[validation_cutoff:]
    train_df = dataset.loc[dataset["time"].dt.normalize().isin(train_dates)].copy()
    validation_df = dataset.loc[dataset["time"].dt.normalize().isin(validation_dates)].copy()
    test_df = dataset.loc[dataset["time"].dt.normalize().isin(test_dates)].copy()
    return train_df, validation_df, test_df


def downsample_for_tuning(train_df: pd.DataFrame, max_rows: int = MAX_TUNING_ROWS) -> pd.DataFrame:
    if len(train_df) <= max_rows:
        return train_df.copy()
    stride = math.ceil(len(train_df) / max_rows)
    return train_df.iloc[::stride].copy().reset_index(drop=True)


def fit_probability_calibrator(y_true: pd.Series, probabilities: np.ndarray) -> LogisticRegression | None:
    y_values = pd.Series(y_true).astype(int)
    if y_values.nunique() < 2:
        return None
    calibrator = LogisticRegression(random_state=42, solver="lbfgs")
    calibrator.fit(np.asarray(probabilities, dtype=float).reshape(-1, 1), y_values.to_numpy())
    return calibrator


def apply_probability_calibration(probabilities: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if calibrator is None:
        return raw
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def choose_precision_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, Dict[str, float]]:
    candidate_thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.30, 0.85, 23),
                    np.quantile(probabilities, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]),
                    np.array([0.50]),
                ]
            ),
            0.0,
            1.0,
        )
    )

    minimum_positive_predictions = max(5, int(len(y_true) * 0.02))
    best_threshold = 0.50
    best_metrics: Dict[str, float] | None = None
    best_rank: tuple[float, ...] | None = None

    for threshold in candidate_thresholds:
        predictions = classify_with_threshold(probabilities, float(threshold))
        metrics = score_predictions(y_true, predictions, probabilities)
        metrics["f0_5"] = fbeta_score(y_true, predictions, beta=0.5, zero_division=0)
        metrics["threshold"] = float(threshold)
        predicted_positives = int(predictions.sum())
        meets_minimum_activity = predicted_positives >= minimum_positive_predictions and metrics["recall"] >= 0.05
        rank = (
            1.0 if meets_minimum_activity else 0.0,
            metrics["precision"],
            metrics["f0_5"],
            metrics["average_precision"],
            metrics["recall"],
            -abs(float(threshold) - 0.5),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def score_topk(frame: pd.DataFrame, k_values: List[int] | None = None) -> pd.DataFrame:
    if k_values is None:
        k_values = TOP_K_VALUES
    rows = []
    ordered = frame.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    for k in k_values:
        subset = ordered.head(min(k, len(ordered))).copy()
        if subset.empty:
            continue
        precision_at_k = float(subset["target"].mean())
        rows.append(
            {
                "k": int(k),
                "rows": int(len(subset)),
                "precision_at_k": precision_at_k,
                "hit_count": int(subset["target"].sum()),
                "average_probability": float(subset["predicted_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_latest_policies(
    latest_signals: pd.DataFrame,
    threshold: float,
    holdout_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    top_k_metrics: pd.DataFrame,
) -> pd.DataFrame:
    calibrated_threshold_signals = int((latest_signals["buy_probability"] >= threshold).sum())
    hard_gate_signals = int((latest_signals["buy_probability"] >= max(threshold + 0.05, 0.60)).sum())
    top3_precision = math.nan
    if not top_k_metrics.empty and (top_k_metrics["k"] == 3).any():
        top3_precision = float(top_k_metrics.loc[top_k_metrics["k"] == 3, "precision_at_k"].iloc[0])
    quality_floor_pass = (
        holdout_metrics.get("average_precision", 0.0) >= 0.40
        and validation_metrics.get("precision", 0.0) >= 0.40
        and (math.isnan(top3_precision) or top3_precision >= 0.66)
    )
    top_k_signals = min(3, len(latest_signals)) if quality_floor_pass else 0
    return pd.DataFrame(
        [
            {"policy": "current_hard_gate", "signal_count": hard_gate_signals, "holdout_precision_at_3": top3_precision},
            {"policy": "calibrated_threshold", "signal_count": calibrated_threshold_signals, "holdout_precision_at_3": top3_precision},
            {"policy": "top_3_with_quality_floor", "signal_count": top_k_signals, "holdout_precision_at_3": top3_precision},
        ]
    )


def score_by_exchange(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for exchange, group in frame.groupby("exchange"):
        metrics = score_predictions(group[target_column], group["predicted_label"], group["predicted_probability"])
        metrics["exchange"] = exchange
        metrics["rows"] = int(len(group))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def evaluate_model(
    scenario: ScenarioConfig,
    model_spec: ModelSpec,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_dataset: pd.DataFrame,
    latest_df: pd.DataFrame,
) -> ExperimentResult:
    target_column = make_target_column(scenario.target_return)
    tuning_df = downsample_for_tuning(train_df)
    cv_splits = build_time_splits(tuning_df, n_splits=5)

    x_tune = tuning_df[scenario.feature_columns]
    y_tune = tuning_df[target_column]
    x_train = train_df[scenario.feature_columns]
    y_train = train_df[target_column]
    x_val = validation_df[scenario.feature_columns]
    y_val = validation_df[target_column]
    x_test = test_df[scenario.feature_columns]
    y_test = test_df[target_column]

    search = GridSearchCV(
        estimator=model_spec.builder(),
        param_grid=model_spec.param_grid,
        scoring={
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        refit="average_precision",
        cv=cv_splits,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_tune, y_tune)
    best_estimator = clone(search.best_estimator_)
    best_estimator.fit(x_train, y_train)

    train_prob = best_estimator.predict_proba(x_train)[:, 1]
    val_prob = best_estimator.predict_proba(x_val)[:, 1]
    calibrator = fit_probability_calibrator(y_val, val_prob)
    train_prob_calibrated = apply_probability_calibration(train_prob, calibrator)
    val_prob_calibrated = apply_probability_calibration(val_prob, calibrator)
    threshold, validation_metrics = choose_precision_threshold(y_val, val_prob_calibrated)
    validation_metrics["threshold"] = threshold
    train_pred = classify_with_threshold(train_prob_calibrated, threshold)
    train_metrics = score_predictions(y_train, train_pred, train_prob_calibrated)
    train_metrics["threshold"] = threshold

    test_prob = best_estimator.predict_proba(x_test)[:, 1]
    test_prob_calibrated = apply_probability_calibration(test_prob, calibrator)
    test_pred = classify_with_threshold(test_prob_calibrated, threshold)
    holdout_metrics = score_predictions(y_test, test_pred, test_prob_calibrated)
    holdout_metrics["threshold"] = threshold
    if holdout_metrics["positive_rate"] > 0:
        holdout_metrics["precision_lift_vs_base_rate"] = (
            holdout_metrics["precision"] / holdout_metrics["positive_rate"]
        )
    else:
        holdout_metrics["precision_lift_vs_base_rate"] = math.nan

    holdout_predictions = test_df[
        ["time", "symbol", "exchange", "organ_name", "close", "future_peak_return", target_column]
    ].copy()
    holdout_predictions["predicted_label"] = test_pred
    holdout_predictions["predicted_probability"] = test_prob_calibrated
    holdout_predictions["raw_probability"] = test_prob
    holdout_predictions["target"] = holdout_predictions[target_column]
    exchange_holdout_metrics = score_by_exchange(holdout_predictions, target_column=target_column)
    top_k_metrics = score_topk(holdout_predictions)

    fold_rows = []
    for fold_no, (train_idx, val_idx) in enumerate(cv_splits, start=1):
        fold_estimator = clone(search.best_estimator_)
        fold_estimator.fit(x_tune.iloc[train_idx], y_tune.iloc[train_idx])
        fold_prob = fold_estimator.predict_proba(x_tune.iloc[val_idx])[:, 1]
        fold_threshold, _ = choose_precision_threshold(y_tune.iloc[val_idx], fold_prob)
        fold_pred = classify_with_threshold(fold_prob, fold_threshold)
        fold_metrics = score_predictions(y_tune.iloc[val_idx], fold_pred, fold_prob)
        fold_metrics["threshold"] = fold_threshold
        fold_metrics["fold"] = fold_no
        fold_metrics["rows"] = int(len(val_idx))
        fold_rows.append(fold_metrics)
    fold_metrics_df = pd.DataFrame(fold_rows)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_summary = cv_results[
        [
            "rank_test_average_precision",
            "mean_test_average_precision",
            "mean_test_roc_auc",
            "mean_test_precision",
            "mean_test_recall",
            "mean_test_f1",
            "params",
        ]
    ].sort_values(["rank_test_average_precision", "mean_test_roc_auc"], ascending=[True, False])

    full_estimator = clone(search.best_estimator_)
    full_estimator.fit(full_dataset[scenario.feature_columns], full_dataset[target_column])
    latest_prob = full_estimator.predict_proba(latest_df[scenario.feature_columns])[:, 1]
    latest_prob_calibrated = apply_probability_calibration(latest_prob, calibrator)
    latest_pred = classify_with_threshold(latest_prob_calibrated, threshold)
    latest_signals = latest_df[["time", "symbol", "exchange", "organ_name", "close"]].copy()
    latest_signals["buy_signal"] = latest_pred
    latest_signals["buy_probability"] = latest_prob_calibrated
    latest_signals["raw_probability"] = latest_prob
    latest_signals = latest_signals.sort_values("buy_probability", ascending=False).reset_index(drop=True)

    feature_importance = pd.DataFrame({"feature": scenario.feature_columns, "importance": 0.0})
    estimator_for_importance = full_estimator
    if isinstance(full_estimator, Pipeline):
        estimator_for_importance = full_estimator.steps[-1][1]
    if hasattr(estimator_for_importance, "feature_importances_"):
        feature_importance["importance"] = estimator_for_importance.feature_importances_
    elif hasattr(estimator_for_importance, "coef_"):
        coefficients = np.asarray(estimator_for_importance.coef_).reshape(-1)
        feature_importance["importance"] = np.abs(coefficients)
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    policy_metrics = top_k_metrics.copy()
    latest_policy_metrics = evaluate_latest_policies(
        latest_signals=latest_signals,
        threshold=threshold,
        holdout_metrics=holdout_metrics,
        validation_metrics=validation_metrics,
        top_k_metrics=top_k_metrics,
    )
    if not latest_policy_metrics.empty:
        latest_policy_metrics["rows"] = np.nan
        latest_policy_metrics["precision_at_k"] = np.nan
        latest_policy_metrics["hit_count"] = np.nan
        latest_policy_metrics["average_probability"] = np.nan
        policy_metrics = pd.concat([policy_metrics, latest_policy_metrics], ignore_index=True, sort=False)

    return ExperimentResult(
        scenario_name=scenario.name,
        feature_set_name=scenario.feature_set_name,
        target_return=scenario.target_return,
        model_name=model_spec.name,
        best_params=search.best_params_,
        best_cv_score=float(search.best_score_),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        cv_summary=cv_summary,
        fold_metrics=fold_metrics_df,
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        exchange_holdout_metrics=exchange_holdout_metrics,
        feature_importance=feature_importance,
        latest_signals=latest_signals,
        threshold=threshold,
        policy_metrics=policy_metrics,
    )


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: f"{value:.{float_digits}f}" if pd.notna(value) else "nan"
            )
    headers = "| " + " | ".join(display.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([headers, divider, *rows])


def save_dataframe(frame: pd.DataFrame, file_name: str) -> None:
    output_path = OUTPUT_DIR / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_report(
    universe_df: pd.DataFrame,
    scenario_summaries: List[dict],
    final_result: ExperimentResult,
    final_history_df: pd.DataFrame,
    final_labeled_df: pd.DataFrame,
    final_latest_df: pd.DataFrame,
    final_failures_df: pd.DataFrame,
    final_train_df: pd.DataFrame,
    final_validation_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
) -> None:
    requested_symbol_text = ", ".join(
        f"{symbol} ({REQUESTED_SYMBOLS[symbol]})" for symbol in REQUESTED_SYMBOL_LIST
    )
    swept_targets = list(dict.fromkeys(item["target_return"] for item in scenario_summaries))
    swept_targets_text = ", ".join(f"`{int(round(value * 100))}%`" for value in swept_targets)
    target_column = make_target_column(final_result.target_return)
    coverage_summary = final_labeled_df.groupby(["exchange"]).agg(
        symbols=("symbol", "nunique"),
        labeled_rows=("symbol", "size"),
        positive_rate=(target_column, "mean"),
    ).reset_index()

    symbol_summary = final_labeled_df.groupby(["symbol", "exchange", "organ_name"]).agg(
        labeled_rows=("symbol", "size"),
        first_date=("time", "min"),
        last_date=("time", "max"),
        positive_rate=(target_column, "mean"),
    ).reset_index()
    symbol_summary["first_date"] = symbol_summary["first_date"].dt.date.astype(str)
    symbol_summary["last_date"] = symbol_summary["last_date"].dt.date.astype(str)
    symbol_summary = symbol_summary.sort_values(["positive_rate", "labeled_rows"], ascending=[False, False])
    scenario_summary_df = pd.DataFrame(scenario_summaries).sort_values(
        ["holdout_precision_at_3", "holdout_average_precision", "holdout_f1"],
        ascending=[False, False, False],
    )
    top_3_policy_signal_count = 0
    if (final_result.policy_metrics["policy"] == "top_3_with_quality_floor").any():
        top_3_policy_signal_count = int(
            final_result.policy_metrics.loc[
                final_result.policy_metrics["policy"] == "top_3_with_quality_floor", "signal_count"
            ].iloc[0]
        )
    top_policy_symbols = ", ".join(final_result.latest_signals.head(top_3_policy_signal_count)["symbol"])

    report_lines = [
        "# Vietnam stock signal experiment report",
        "",
        "## Scope",
        "",
        f"- Universe used for this run: `{UNIVERSE_LABEL}`",
        f"- Universe source: {UNIVERSE_SOURCE}",
        f"- Requested symbols ({len(REQUESTED_SYMBOL_LIST)}): {requested_symbol_text}",
        f"- Requested history window: {START_DATE} to {TODAY_STR}",
        f"- Horizon: next {HORIZON_DAYS} trading days",
        "- This report summarizes the full model-quality experiment matrix and then drills into the selected final setup.",
        "",
        "## Experiment Matrix Summary",
        "",
        f"- Targets swept: {swept_targets_text}",
        "- Feature sets compared: `baseline_technical`, `extended_breakout`",
        "- Model families compared: Decision Tree, Random Forest, Logistic Regression, HistGradientBoosting, and XGBoost when available",
        "- Selection goal: maximize precision while still producing some actionable latest signals",
        "",
        markdown_table(
            scenario_summary_df[
                [
                    "scenario",
                    "target_return",
                    "feature_set",
                    "model",
                    "holdout_precision_at_3",
                    "holdout_precision",
                    "holdout_average_precision",
                    "holdout_f1",
                    "holdout_balanced_accuracy",
                    "latest_signal_count",
                    "top_3_policy_signal_count",
                ]
            ].head(18)
        ),
        "",
        "## Final Selected Setup",
        "",
        f"- Scenario: `{final_result.scenario_name}`",
        f"- Target: future max daily high reaches at least `{final_result.target_return:.0%}` above current close",
        f"- Feature set: `{final_result.feature_set_name}` with `{len(final_result.feature_importance)}` features",
        f"- Model: `{final_result.model_name}`",
        f"- Best CV average precision: `{final_result.best_cv_score:.4f}`",
        f"- Selected decision threshold: `{final_result.threshold:.4f}`",
        f"- Latest calibrated buy signals at the chosen threshold: `{int(final_result.latest_signals['buy_signal'].sum())} / {len(final_result.latest_signals)}`",
        f"- Latest `top_3_with_quality_floor` policy signals: `{top_3_policy_signal_count}`",
        f"- Selected `YES` symbols under that policy: {top_policy_symbols if top_policy_symbols else 'none'}",
        "",
        "## Universe coverage",
        "",
        f"- Symbols in requested universe: {universe_df['symbol'].nunique()}",
        f"- Symbols with fetched history: {final_history_df['symbol'].nunique()}",
        f"- Symbols with enough labeled rows for modeling: {final_labeled_df['symbol'].nunique()}",
        f"- Fetch failures: {len(final_failures_df)}",
        f"- Total raw rows: {len(final_history_df)}",
        f"- Total labeled rows: {len(final_labeled_df)}",
        f"- Combined positive rate: {final_labeled_df[target_column].mean():.4f}",
        "",
        markdown_table(coverage_summary),
        "",
        "## Feature set",
        "",
        "- Baseline technical signals: MA gaps, support/resistance gaps, volume ratios, Bollinger `%B`, RSI, one-day return",
        "- Extended signals: multi-horizon momentum, volatility, ATR ratio, breakout gaps to rolling highs/lows, drawdown, intraday range, and volume z-score",
        "",
        "## Data split",
        "",
        f"- Train rows: {len(final_train_df)}",
        f"- Validation rows: {len(final_validation_df)}",
        f"- Holdout rows: {len(final_test_df)}",
        f"- Tuning cap for CV: {MAX_TUNING_ROWS} rows",
        "",
        "## Latest daily rows used for ranking",
        "",
        markdown_table(final_latest_df[["time", "symbol", "exchange", "close"]].sort_values(["exchange", "symbol"])),
        "",
        f"## {final_result.model_name}",
        "",
        f"- Best params: `{json.dumps(final_result.best_params, ensure_ascii=True)}`",
        f"- Best CV average precision: {final_result.best_cv_score:.4f}",
        "",
        "### Train metrics",
        "",
        markdown_table(pd.DataFrame([final_result.train_metrics])),
        "",
        "### Validation metrics",
        "",
        markdown_table(pd.DataFrame([final_result.validation_metrics])),
        "",
        "### Holdout metrics",
        "",
        markdown_table(pd.DataFrame([final_result.holdout_metrics])),
        "",
        "### Holdout metrics by exchange",
        "",
        markdown_table(final_result.exchange_holdout_metrics),
        "",
        "### Decision Policy Comparison",
        "",
        markdown_table(final_result.policy_metrics),
        "",
        "### Top latest calibrated buy probabilities",
        "",
        markdown_table(final_result.latest_signals.head(25)),
        "",
        "### Top feature importance",
        "",
        markdown_table(final_result.feature_importance.head(15)),
        "",
        "### Top CV configurations",
        "",
        markdown_table(final_result.cv_summary.head(5)),
        "",
        "### Fold-by-fold CV metrics",
        "",
        markdown_table(final_result.fold_metrics),
        "",
        "## Best model takeaway",
        "",
        f"- Best final scenario by precision-first ranking: `{final_result.scenario_name}` using `{final_result.model_name}`",
        f"- Holdout precision: `{final_result.holdout_metrics['precision']:.4f}`",
        f"- Holdout precision@3: `{float(final_result.policy_metrics.loc[final_result.policy_metrics['k'] == 3, 'precision_at_k'].iloc[0]) if (final_result.policy_metrics['k'] == 3).any() else math.nan:.4f}`",
        f"- Holdout average precision: `{final_result.holdout_metrics['average_precision']:.4f}`",
        f"- Holdout F1: `{final_result.holdout_metrics['f1']:.4f}`",
        f"- Best model latest top 10 candidates: {', '.join(final_result.latest_signals.head(10)['symbol'])}",
        "",
        "## Most favorable symbols by historical target frequency",
        "",
        markdown_table(symbol_summary.head(20)),
        "",
        "## Notes",
        "",
        "- Four weeks is approximated as 20 trading days.",
        "- The target is based on the future daily high, not the future close.",
        "- Thresholds are selected on a chronological validation slice after probability calibration.",
        "- The report summarizes the currently configured target sweep and selects the strongest precision-first setup from those runs.",
        "- Current report artifacts and detailed CSV outputs are saved in `stock_signal_project/outputs/`.",
    ]

    if not final_failures_df.empty:
        report_lines.extend(
            [
                "",
                "## Fetch failures",
                "",
                markdown_table(final_failures_df.head(20)),
            ]
        )

    (OUTPUT_DIR / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def build_model_specs() -> List[ModelSpec]:
    specs = [
        ModelSpec(
            name="Decision Tree",
            builder=lambda: DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            param_grid={
                "max_depth": [5, 8, 12],
                "min_samples_leaf": [10, 25, 50],
                "criterion": ["gini", "entropy"],
            },
        ),
        ModelSpec(
            name="Random Forest",
            builder=lambda: RandomForestClassifier(
                random_state=42, class_weight="balanced_subsample", n_jobs=-1
            ),
            param_grid={
                "n_estimators": [200],
                "max_depth": [8, None],
                "min_samples_leaf": [10, 25],
                "max_features": ["sqrt", 0.7],
            },
        ),
        ModelSpec(
            name="Logistic Regression",
            builder=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            random_state=42,
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__C": [0.1, 0.5, 1.0, 2.0],
            },
        ),
        ModelSpec(
            name="HistGradientBoosting",
            builder=lambda: HistGradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                max_iter=250,
                validation_fraction=None,
            ),
            param_grid={
                "max_depth": [3, 5, None],
                "max_leaf_nodes": [15, 31],
                "min_samples_leaf": [20, 50],
            },
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                builder=lambda: XGBClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=4,
                ),
                param_grid={
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 5],
                    "reg_lambda": [1.0, 3.0],
                },
            )
        )
    return specs


def run_scenario(
    scenario: ScenarioConfig,
    universe_df: pd.DataFrame,
    model_specs: List[ModelSpec],
) -> tuple[dict, ExperimentResult, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_df, labeled_df, latest_df, failures_df = build_dataset(
        universe_df,
        target_return=scenario.target_return,
        feature_columns=scenario.feature_columns,
    )
    history_df = history_df.sort_values(["time", "symbol"]).reset_index(drop=True)
    labeled_df = labeled_df.sort_values(["time", "symbol"]).reset_index(drop=True)
    latest_df = latest_df.sort_values(["time", "symbol"]).reset_index(drop=True)
    train_df, validation_df, test_df = split_dataset_by_date(labeled_df)

    scenario_dir = OUTPUT_DIR / "experiments" / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    results: List[ExperimentResult] = []
    summary_rows = []

    for model_spec in model_specs:
        print(f"[{scenario.name}] Training {model_spec.name}...", flush=True)
        result = evaluate_model(
            scenario=scenario,
            model_spec=model_spec,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            full_dataset=labeled_df,
            latest_df=latest_df,
        )
        results.append(result)
        top3_precision = math.nan
        if not result.policy_metrics.empty and "k" in result.policy_metrics.columns:
            top3_rows = result.policy_metrics.loc[result.policy_metrics["k"] == 3, "precision_at_k"]
            if not top3_rows.empty:
                top3_precision = float(top3_rows.iloc[0])
        top3_signal_count = 0
        if not result.policy_metrics.empty and "policy" in result.policy_metrics.columns:
            top3_policy_rows = result.policy_metrics.loc[
                result.policy_metrics["policy"] == "top_3_with_quality_floor", "signal_count"
            ]
            if not top3_policy_rows.empty:
                top3_signal_count = int(top3_policy_rows.iloc[0])
        summary_rows.append(
            {
                "scenario": scenario.name,
                "target_return": scenario.target_return,
                "feature_set": scenario.feature_set_name,
                "model": result.model_name,
                "best_cv_average_precision": result.best_cv_score,
                "holdout_precision": result.holdout_metrics["precision"],
                "holdout_average_precision": result.holdout_metrics["average_precision"],
                "holdout_f1": result.holdout_metrics["f1"],
                "holdout_balanced_accuracy": result.holdout_metrics["balanced_accuracy"],
                "latest_signal_count": int(result.latest_signals["buy_signal"].sum()),
                "holdout_precision_at_3": top3_precision,
                "top_3_policy_signal_count": top3_signal_count,
                "threshold": result.threshold,
            }
        )

        safe_name = result.model_name.lower().replace(" ", "_")
        save_dataframe(result.cv_summary, f"experiments/{scenario.name}/{safe_name}_cv_summary.csv")
        save_dataframe(result.fold_metrics, f"experiments/{scenario.name}/{safe_name}_fold_metrics.csv")
        save_dataframe(result.holdout_predictions, f"experiments/{scenario.name}/{safe_name}_holdout_predictions.csv")
        save_dataframe(
            result.exchange_holdout_metrics,
            f"experiments/{scenario.name}/{safe_name}_exchange_holdout_metrics.csv",
        )
        save_dataframe(result.feature_importance, f"experiments/{scenario.name}/{safe_name}_feature_importance.csv")
        save_dataframe(result.latest_signals, f"experiments/{scenario.name}/{safe_name}_latest_signals.csv")
        save_dataframe(result.policy_metrics, f"experiments/{scenario.name}/{safe_name}_policy_metrics.csv")

    scenario_summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_precision", "holdout_average_precision", "holdout_f1"],
        ascending=[False, False, False],
    )
    save_dataframe(history_df, f"experiments/{scenario.name}/raw_history.csv")
    save_dataframe(labeled_df, f"experiments/{scenario.name}/labeled_dataset.csv")
    save_dataframe(latest_df, f"experiments/{scenario.name}/latest_feature_rows.csv")
    save_dataframe(failures_df, f"experiments/{scenario.name}/fetch_failures.csv")
    save_dataframe(scenario_summary, f"experiments/{scenario.name}/model_summary.csv")

    best_result = max(
        results,
        key=lambda item: (
            float(item.policy_metrics.loc[item.policy_metrics["k"] == 3, "precision_at_k"].iloc[0])
            if (not item.policy_metrics.empty and (item.policy_metrics["k"] == 3).any())
            else -1.0,
            item.holdout_metrics["average_precision"],
            item.holdout_metrics["f1"],
            int(
                item.policy_metrics.loc[
                    item.policy_metrics["policy"] == "top_3_with_quality_floor", "signal_count"
                ].iloc[0]
            )
            if (not item.policy_metrics.empty and (item.policy_metrics["policy"] == "top_3_with_quality_floor").any())
            else 0,
        ),
    )
    best_summary_row = scenario_summary.iloc[0].to_dict()
    best_summary_row["train_rows"] = int(len(train_df))
    best_summary_row["validation_rows"] = int(len(validation_df))
    best_summary_row["test_rows"] = int(len(test_df))
    best_summary_row["modeled_symbols"] = int(labeled_df["symbol"].nunique())
    return best_summary_row, best_result, history_df, labeled_df, latest_df, failures_df, train_df, validation_df, test_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock signal experiment sweeps and generate a summary report.")
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=TARGET_SWEEP,
        help="Target returns to evaluate, for example 0.10 0.07 0.06",
    )
    parser.add_argument(
        "--final-target",
        type=float,
        default=DEFAULT_TARGET_RETURN,
        help="Target return to use for the final extended-feature rerun and report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    universe_df = get_universe()
    model_specs = build_model_specs()
    scenario_summaries: List[dict] = []
    scenario_payloads = []

    for target_return in args.targets:
        scenario = ScenarioConfig(
            name=f"baseline_target_{int(round(target_return * 100)):02d}pct",
            target_return=target_return,
            feature_columns=BASE_FEATURE_COLUMNS,
            feature_set_name="baseline_technical",
        )
        payload = run_scenario(scenario=scenario, universe_df=universe_df, model_specs=model_specs)
        scenario_summaries.append(payload[0])
        scenario_payloads.append(payload)

    extended_target = args.final_target
    extended_scenario = ScenarioConfig(
        name=f"extended_target_{int(round(extended_target * 100)):02d}pct",
        target_return=extended_target,
        feature_columns=EXTENDED_FEATURE_COLUMNS,
        feature_set_name="extended_breakout",
    )
    extended_payload = run_scenario(scenario=extended_scenario, universe_df=universe_df, model_specs=model_specs)
    scenario_summaries.append(extended_payload[0])
    scenario_payloads.append(extended_payload)
    final_summary_df = pd.DataFrame(scenario_summaries).sort_values(
        ["holdout_precision_at_3", "holdout_average_precision", "holdout_f1"],
        ascending=[False, False, False],
    )
    save_dataframe(final_summary_df, "model_summary.csv")
    save_dataframe(universe_df, "universe.csv")

    final_payload = max(
        scenario_payloads,
        key=lambda payload: (
            float(payload[0].get("holdout_precision_at_3", -1.0)),
            payload[1].holdout_metrics["average_precision"],
            payload[1].holdout_metrics["f1"],
            int(payload[0].get("top_3_policy_signal_count", 0)),
        ),
    )
    _, final_result, final_history_df, final_labeled_df, final_latest_df, final_failures_df, final_train_df, final_validation_df, final_test_df = final_payload
    save_dataframe(final_history_df, "raw_history.csv")
    save_dataframe(final_labeled_df, "labeled_dataset.csv")
    save_dataframe(final_latest_df, "latest_feature_rows.csv")
    save_dataframe(final_failures_df, "fetch_failures.csv")
    save_dataframe(final_result.latest_signals, "selected_latest_signals.csv")
    save_dataframe(final_result.holdout_predictions, "selected_holdout_predictions.csv")
    save_dataframe(final_result.feature_importance, "selected_feature_importance.csv")
    save_dataframe(final_result.policy_metrics, "selected_policy_metrics.csv")
    selected_policy_yes_count = 0
    if (final_result.policy_metrics["policy"] == "top_3_with_quality_floor").any():
        selected_policy_yes_count = int(
            final_result.policy_metrics.loc[
                final_result.policy_metrics["policy"] == "top_3_with_quality_floor", "signal_count"
            ].iloc[0]
        )
    selected_policy_yes = final_result.latest_signals.head(selected_policy_yes_count).copy()
    if not selected_policy_yes.empty:
        selected_policy_yes["decision"] = "YES"
    save_dataframe(selected_policy_yes, "selected_policy_yes_symbols.csv")

    metadata = {
        "universe_label": UNIVERSE_LABEL,
        "universe_source": UNIVERSE_SOURCE,
        "requested_symbols": REQUESTED_SYMBOL_LIST,
        "start_date": START_DATE,
        "end_date": TODAY_STR,
        "horizon_days": HORIZON_DAYS,
        "targets_swept": args.targets,
        "final_target_return": final_result.target_return,
        "selected_model": final_result.model_name,
        "selected_scenario": final_result.scenario_name,
        "selected_policy": "top_3_with_quality_floor",
        "selected_policy_yes_count": selected_policy_yes_count,
        "feature_columns": final_result.feature_importance["feature"].tolist(),
        "train_rows": int(len(final_train_df)),
        "validation_rows": int(len(final_validation_df)),
        "test_rows": int(len(final_test_df)),
        "modeled_symbols": int(final_labeled_df["symbol"].nunique()),
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
        "min_labeled_rows_per_symbol": MIN_LABELED_ROWS_PER_SYMBOL,
        "max_tuning_rows": MAX_TUNING_ROWS,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        universe_df=universe_df,
        scenario_summaries=scenario_summaries,
        final_result=final_result,
        final_history_df=final_history_df,
        final_labeled_df=final_labeled_df,
        final_latest_df=final_latest_df,
        final_failures_df=final_failures_df,
        final_train_df=final_train_df,
        final_validation_df=final_validation_df,
        final_test_df=final_test_df,
    )


if __name__ == "__main__":
    main()
