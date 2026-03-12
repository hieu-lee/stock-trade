from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VNSTOCK_REPO = PROJECT_ROOT / "vnstock"
ARTIFACT_ROOT = PROJECT_ROOT / "stock_signal_project" / "artifacts"
MODEL_DIR = ARTIFACT_ROOT / "models"
HISTORY_CACHE_DIR = ARTIFACT_ROOT / "history_cache"
DECISION_DIR = ARTIFACT_ROOT / "decisions"
LEGACY_HISTORY_CACHE_DIR = PROJECT_ROOT / "stock_signal_project" / "outputs" / "history_cache"

if str(VNSTOCK_REPO) not in sys.path:
    sys.path.insert(0, str(VNSTOCK_REPO))

from vnstock import Quote  # noqa: E402

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


ARTIFACT_VERSION = 3
LOOKBACK_YEARS = 15
HORIZON_DAYS = 20
TARGET_RETURN = 0.06
MIN_LABELED_ROWS = 252
RECENT_FETCH_DAYS = 260
ROLLING_WINDOWS = [10, 25, 50, 100]
TARGET_COLUMN = "target_profit_4w_06pct"
MIN_PRODUCTION_PROBABILITY = 0.55
MIN_PROBABILITY_MARGIN = 0.00
MIN_VALIDATION_PRECISION = 0.40
MIN_TEST_PRECISION = 0.45
MIN_TEST_BALANCED_ACCURACY = 0.50
MIN_CV_AVERAGE_PRECISION = 0.40
FEATURE_COLUMNS = [
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


@dataclass
class SymbolModelResult:
    symbol: str
    model_name: str
    estimator: object
    calibrator: object | None
    threshold: float
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    best_params: Dict[str, object]
    latest_history_date: str
    trained_rows: int


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


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DECISION_DIR.mkdir(parents=True, exist_ok=True)


def today_str() -> str:
    return date.today().isoformat()


def full_history_start() -> str:
    return (date.today() - timedelta(days=365 * LOOKBACK_YEARS + 4)).isoformat()


def model_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol.upper()}.joblib"


def metadata_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol.upper()}.json"


def history_cache_path(symbol: str) -> Path:
    return HISTORY_CACHE_DIR / f"{symbol.upper()}.csv"


def legacy_history_cache_path(symbol: str) -> Path:
    return LEGACY_HISTORY_CACHE_DIR / f"{symbol.upper()}.csv"


def safe_divide(numerator, denominator, fill_value=np.nan):
    numerator_is_series = isinstance(numerator, pd.Series)
    denominator_is_series = isinstance(denominator, pd.Series)
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

    if numerator_is_series:
        return pd.Series(result, index=numerator.index)
    if denominator_is_series:
        return pd.Series(result, index=denominator.index)
    return result


def sanitize_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required_columns = {"time", "open", "high", "low", "close", "volume"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["time"] = pd.to_datetime(cleaned["time"])
    for column in ["open", "high", "low", "close", "volume"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    cleaned = cleaned.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    cleaned["symbol"] = symbol.upper()
    return cleaned


def fetch_remote_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    quote = Quote(symbol=symbol.upper(), source="vci", show_log=False)
    raw = quote.history(start=start, end=end, interval="1D").copy()
    if raw.empty:
        raise ValueError(f"No history returned for {symbol.upper()}")
    return sanitize_history(raw, symbol.upper())


def load_cached_history(symbol: str) -> pd.DataFrame | None:
    primary = history_cache_path(symbol)
    legacy = legacy_history_cache_path(symbol)
    chosen = primary if primary.exists() else legacy if legacy.exists() else None
    if chosen is None:
        return None
    cached = pd.read_csv(chosen, parse_dates=["time"])
    return sanitize_history(cached, symbol.upper())


def save_cached_history(symbol: str, history: pd.DataFrame) -> None:
    sanitize_history(history, symbol.upper()).to_csv(history_cache_path(symbol), index=False)


def refresh_symbol_history(symbol: str) -> pd.DataFrame:
    cached = load_cached_history(symbol)
    if cached is None:
        full = fetch_remote_history(symbol, full_history_start(), today_str())
        save_cached_history(symbol, full)
        return full

    latest_cached_date = cached["time"].max().date()
    refresh_start = min(
        latest_cached_date - timedelta(days=RECENT_FETCH_DAYS),
        date.today() - timedelta(days=RECENT_FETCH_DAYS),
    )
    recent = fetch_remote_history(symbol, refresh_start.isoformat(), today_str())
    merged = pd.concat([cached, recent], ignore_index=True)
    merged = sanitize_history(merged, symbol.upper())
    save_cached_history(symbol, merged)
    return merged


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

        right_pos = bisect_right(high_levels, current_close)
        seen_less_equal = high_tree.prefix_sum(right_pos)
        if seen_less_equal == high_tree.total():
            resistance_gaps.append(0.0)
        else:
            next_rank = high_tree.find_by_order(seen_less_equal + 1)
            resistance = high_levels[next_rank - 1]
            resistance_gaps.append(0.0 if resistance <= 0 else abs(resistance - current_close) / resistance)

        left_pos = bisect_left(low_levels, current_close)
        seen_less = low_tree.prefix_sum(left_pos)
        if seen_less == 0:
            support_gaps.append(0.0)
        else:
            prev_rank = low_tree.find_by_order(seen_less)
            support = low_levels[prev_rank - 1]
            support_gaps.append(0.0 if support <= 0 else abs(support - current_close) / support)

    output = df.copy()
    output["resistance_gap"] = resistance_gaps
    output["support_gap"] = support_gaps
    return output


def compute_future_peak(high: pd.Series, horizon: int) -> pd.Series:
    values = high.to_numpy(dtype=float)
    future_peak = np.full(len(values), np.nan)
    for idx in range(len(values) - horizon):
        future_peak[idx] = values[idx + 1 : idx + 1 + horizon].max()
    return pd.Series(future_peak, index=high.index)


def engineer_features(history: pd.DataFrame) -> pd.DataFrame:
    df = sanitize_history(history, history["symbol"].iloc[0]).copy()

    for window in ROLLING_WINDOWS:
        ma = df["close"].rolling(window).mean()
        avg_volume = df["volume"].rolling(window).mean()
        df[f"ma_gap_{window}"] = safe_divide(ma - df["close"], ma)
        df[f"volume_ratio_{window}"] = safe_divide(df["volume"], avg_volume)

    df = compute_support_resistance_gaps(df)
    previous_close = df["close"].shift(1)
    df["price_change_1d"] = safe_divide(df["close"] - previous_close, previous_close)

    rolling_mean_20 = df["close"].rolling(20).mean()
    rolling_std_20 = df["close"].rolling(20).std(ddof=0)
    upper = rolling_mean_20 + 2 * rolling_std_20
    lower = rolling_mean_20 - 2 * rolling_std_20
    df["bollinger_percent_b"] = safe_divide(df["close"] - lower, upper - lower)

    df["rsi_14"] = compute_rsi(df["close"], window=14)
    df["future_peak_price"] = compute_future_peak(df["high"], HORIZON_DAYS)
    df["future_peak_return"] = safe_divide(df["future_peak_price"] - df["close"], df["close"])
    df[TARGET_COLUMN] = (df["future_peak_return"] >= TARGET_RETURN).astype(float)
    df[FEATURE_COLUMNS + ["future_peak_price", "future_peak_return"]] = df[
        FEATURE_COLUMNS + ["future_peak_price", "future_peak_return"]
    ].replace([np.inf, -np.inf], np.nan)
    for feature in FEATURE_COLUMNS:
        df[feature] = df[feature].clip(-10.0, 10.0)
    return df


def build_labeled_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = engineer_features(history)
    labeled = features.dropna(subset=FEATURE_COLUMNS + ["future_peak_price"]).copy()
    labeled[TARGET_COLUMN] = labeled[TARGET_COLUMN].astype(int)
    return features, labeled


def split_by_date(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(dataset["time"].dt.normalize().unique()))
    if len(unique_dates) < 90:
        raise ValueError("Not enough trading days to train a robust symbol model")

    train_end = max(1, int(len(unique_dates) * 0.7))
    val_end = max(train_end + 1, int(len(unique_dates) * 0.85))
    val_end = min(val_end, len(unique_dates) - 1)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = dataset.loc[dataset["time"].dt.normalize().isin(train_dates)].copy().reset_index(drop=True)
    val_df = dataset.loc[dataset["time"].dt.normalize().isin(val_dates)].copy().reset_index(drop=True)
    test_df = dataset.loc[dataset["time"].dt.normalize().isin(test_dates)].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Unable to form non-empty chronological train/validation/test splits")
    return train_df, val_df, test_df


def build_time_splits(dataset: pd.DataFrame, n_splits: int = 4) -> List[tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(dataset["time"].dt.normalize().unique()))
    if len(unique_dates) < n_splits + 2:
        n_splits = max(2, len(unique_dates) - 2)
    fold_sizes = np.full(n_splits + 1, len(unique_dates) // (n_splits + 1), dtype=int)
    fold_sizes[: len(unique_dates) % (n_splits + 1)] += 1
    boundaries = np.cumsum(fold_sizes)

    splits: List[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        train_dates = unique_dates[: boundaries[fold_idx]]
        val_dates = unique_dates[boundaries[fold_idx] : boundaries[fold_idx + 1]]
        train_mask = dataset["time"].dt.normalize().isin(train_dates)
        val_mask = dataset["time"].dt.normalize().isin(val_dates)
        if train_mask.any() and val_mask.any():
            train_positions = np.flatnonzero(train_mask.to_numpy())
            val_positions = np.flatnonzero(val_mask.to_numpy())
            splits.append((train_positions, val_positions))

    if len(splits) < 2:
        raise ValueError("Not enough data for time-aware cross-validation")
    return splits


def classify_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


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


def fit_probability_calibrator(y_true: pd.Series, probabilities: np.ndarray) -> LogisticRegression | None:
    target = pd.Series(y_true).astype(int)
    if target.nunique() < 2:
        return None
    calibrator = LogisticRegression(random_state=42, solver="lbfgs")
    calibrator.fit(np.asarray(probabilities, dtype=float).reshape(-1, 1), target.to_numpy())
    return calibrator


def apply_probability_calibration(probabilities: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if calibrator is None:
        return raw
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, Dict[str, float]]:
    candidate_thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.30, 0.80, 21),
                    np.quantile(probabilities, [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
                    np.array([0.5]),
                ]
            ),
            0.0,
            1.0,
        )
    )

    best_threshold = 0.5
    best_metrics: Dict[str, float] | None = None
    best_rank: tuple[float, ...] | None = None
    minimum_positive_predictions = max(3, int(len(y_true) * 0.02))

    for threshold in candidate_thresholds:
        predictions = classify_with_threshold(probabilities, float(threshold))
        metrics = score_predictions(y_true, predictions, probabilities)
        metrics["f0_5"] = fbeta_score(y_true, predictions, beta=0.5, zero_division=0)
        rank = (
            1.0 if int(predictions.sum()) >= minimum_positive_predictions and metrics["recall"] >= 0.05 else 0.0,
            metrics["f0_5"],
            metrics["precision"],
            metrics["balanced_accuracy"],
            metrics["f1"],
            -abs(float(threshold) - 0.5),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def load_model(symbol: str) -> dict | None:
    path = model_path(symbol)
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        return None
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        return None
    metadata = artifact.get("metadata", {})
    if metadata.get("horizon_days") != HORIZON_DAYS:
        return None
    if float(metadata.get("target_return", float("nan"))) != TARGET_RETURN:
        return None
    return artifact


def evaluate_production_rule(artifact: dict, probability: float) -> tuple[bool, float, List[str], Dict[str, object]]:
    metadata = artifact["metadata"]
    base_threshold = float(artifact["threshold"])
    effective_threshold = max(base_threshold + MIN_PROBABILITY_MARGIN, MIN_PRODUCTION_PROBABILITY)

    cv_average_precision = float(metadata.get("cv_best_average_precision", float("nan")))
    validation_metrics = metadata.get("validation_metrics", {})
    validation_precision = float(validation_metrics.get("precision", float("nan")))
    test_metrics = metadata.get("test_metrics", {})
    test_precision = float(test_metrics.get("precision", float("nan")))
    test_balanced_accuracy = float(test_metrics.get("balanced_accuracy", float("nan")))

    checks = {
        "probability_gate": probability >= effective_threshold,
        "cv_average_precision_gate": cv_average_precision >= MIN_CV_AVERAGE_PRECISION,
        "validation_precision_gate": validation_precision >= MIN_VALIDATION_PRECISION,
        "test_precision_gate": test_precision >= MIN_TEST_PRECISION,
        "test_balanced_accuracy_gate": test_balanced_accuracy >= MIN_TEST_BALANCED_ACCURACY,
    }

    reasons: List[str] = []
    if not checks["probability_gate"]:
        reasons.append(
            f"probability {probability:.4f} is below effective threshold {effective_threshold:.4f}"
        )
    if not checks["cv_average_precision_gate"]:
        reasons.append(
            f"cv average precision {cv_average_precision:.4f} is below minimum {MIN_CV_AVERAGE_PRECISION:.4f}"
        )
    if not checks["validation_precision_gate"]:
        reasons.append(
            f"validation precision {validation_precision:.4f} is below minimum {MIN_VALIDATION_PRECISION:.4f}"
        )
    if not checks["test_precision_gate"]:
        reasons.append(
            f"test precision {test_precision:.4f} is below minimum {MIN_TEST_PRECISION:.4f}"
        )
    if not checks["test_balanced_accuracy_gate"]:
        reasons.append(
            f"test balanced accuracy {test_balanced_accuracy:.4f} is below minimum {MIN_TEST_BALANCED_ACCURACY:.4f}"
        )

    production_policy = {
        "base_threshold": base_threshold,
        "effective_threshold": effective_threshold,
        "min_probability_margin": MIN_PROBABILITY_MARGIN,
        "min_production_probability": MIN_PRODUCTION_PROBABILITY,
        "min_cv_average_precision": MIN_CV_AVERAGE_PRECISION,
        "min_validation_precision": MIN_VALIDATION_PRECISION,
        "min_test_precision": MIN_TEST_PRECISION,
        "min_test_balanced_accuracy": MIN_TEST_BALANCED_ACCURACY,
        "checks": checks,
    }
    return all(checks.values()), effective_threshold, reasons, production_policy


def save_model(symbol: str, artifact: dict) -> None:
    joblib.dump(artifact, model_path(symbol))
    metadata_path(symbol).write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")


def train_symbol_model(symbol: str, history: pd.DataFrame | None = None) -> dict:
    clean_symbol = symbol.upper()
    history = sanitize_history(history, clean_symbol) if history is not None else refresh_symbol_history(clean_symbol)
    features, labeled = build_labeled_dataset(history)

    if len(labeled) < MIN_LABELED_ROWS:
        raise ValueError(f"Not enough labeled rows to train {clean_symbol}: only {len(labeled)} rows")

    train_df, val_df, test_df = split_by_date(labeled)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    candidates = [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [5, 10, 20],
                "criterion": ["gini", "entropy"],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1),
            {
                "n_estimators": [200],
                "max_depth": [5, None],
                "min_samples_leaf": [5, 10],
                "max_features": ["sqrt", 0.7],
            },
        ),
    ]
    if XGBClassifier is not None:
        candidates.append(
            (
                "XGBoost",
                XGBClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=4,
                ),
                {
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 5],
                    "reg_lambda": [1.0, 3.0],
                },
            )
        )

    best_result: SymbolModelResult | None = None
    best_metadata: dict | None = None

    for model_name, estimator, param_grid in candidates:
        cv_splits = build_time_splits(train_df)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring={
                "f1": "f1",
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
            },
            refit="average_precision",
            cv=cv_splits,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(x_train, y_train)

        fitted = clone(search.best_estimator_)
        fitted.fit(x_train, y_train)

        val_prob = fitted.predict_proba(x_val)[:, 1]
        calibrator = fit_probability_calibrator(y_val, val_prob)
        val_prob_calibrated = apply_probability_calibration(val_prob, calibrator)
        threshold, validation_metrics = choose_threshold(y_val, val_prob_calibrated)
        test_prob = fitted.predict_proba(x_test)[:, 1]
        test_prob_calibrated = apply_probability_calibration(test_prob, calibrator)
        test_pred = classify_with_threshold(test_prob_calibrated, threshold)
        test_metrics = score_predictions(y_test, test_pred, test_prob_calibrated)

        result = SymbolModelResult(
            symbol=clean_symbol,
            model_name=model_name,
            estimator=fitted,
            calibrator=calibrator,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_params=search.best_params_,
            latest_history_date=history["time"].max().date().isoformat(),
            trained_rows=len(labeled),
        )

        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "symbol": clean_symbol,
            "model_name": model_name,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
            "best_params": search.best_params_,
            "cv_best_average_precision": float(search.best_score_),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "history_start": history["time"].min().date().isoformat(),
            "history_end": history["time"].max().date().isoformat(),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "labeled_rows": int(len(labeled)),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "lookback_years": LOOKBACK_YEARS,
            "horizon_days": HORIZON_DAYS,
            "target_return": TARGET_RETURN,
            "recent_fetch_days": RECENT_FETCH_DAYS,
        }

        if best_result is None:
            best_result = result
            best_metadata = metadata
            continue

        current_key = (
            result.validation_metrics["f1"],
            result.test_metrics["f1"],
            result.test_metrics["roc_auc"],
        )
        best_key = (
            best_result.validation_metrics["f1"],
            best_result.test_metrics["f1"],
            best_result.test_metrics["roc_auc"],
        )
        if current_key > best_key:
            best_result = result
            best_metadata = metadata

    assert best_result is not None
    assert best_metadata is not None

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_estimator = clone(best_result.estimator)
    final_estimator.fit(train_val_df[FEATURE_COLUMNS], train_val_df[TARGET_COLUMN])

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "symbol": clean_symbol,
        "model_name": best_result.model_name,
        "estimator": final_estimator,
        "calibrator": best_result.calibrator,
        "threshold": best_result.threshold,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": best_metadata,
    }
    save_model(clean_symbol, artifact)
    save_cached_history(clean_symbol, history)
    return artifact


def prepare_latest_features(symbol: str, history: pd.DataFrame) -> pd.Series:
    features = engineer_features(history)
    latest_ready = features.dropna(subset=FEATURE_COLUMNS).copy()
    if latest_ready.empty:
        raise ValueError(f"Unable to compute latest features for {symbol.upper()}")
    return latest_ready.iloc[-1]


def predict_symbol(symbol: str, force_retrain: bool = False) -> dict:
    ensure_directories()
    clean_symbol = symbol.upper()
    artifact = None if force_retrain else load_model(clean_symbol)
    history = refresh_symbol_history(clean_symbol)

    trained_now = False
    if artifact is None:
        artifact = train_symbol_model(clean_symbol, history=history)
        trained_now = True
        cached_history = load_cached_history(clean_symbol)
        if cached_history is not None:
            history = cached_history

    latest_row = prepare_latest_features(clean_symbol, history)
    latest_frame = pd.DataFrame([latest_row[FEATURE_COLUMNS].to_dict()])
    raw_probability = float(artifact["estimator"].predict_proba(latest_frame)[0, 1])
    probability = float(apply_probability_calibration(np.array([raw_probability]), artifact.get("calibrator"))[0])
    should_buy, effective_threshold, reasons, production_policy = evaluate_production_rule(artifact, probability)

    result = {
        "symbol": clean_symbol,
        "decision": "YES" if should_buy else "NO",
        "should_buy": bool(should_buy),
        "probability": probability,
        "threshold": float(artifact["threshold"]),
        "effective_threshold": effective_threshold,
        "model_name": artifact["model_name"],
        "trained_now": trained_now,
        "latest_signal_date": pd.Timestamp(latest_row["time"]).date().isoformat(),
        "latest_close": float(latest_row["close"]),
        "buy_for_next_open": "YES" if should_buy else "NO",
        "decision_reasons": reasons if reasons else ["all production checks passed"],
        "production_policy": production_policy,
        "metadata": artifact["metadata"],
        "raw_probability": raw_probability,
    }

    decision_file = DECISION_DIR / f"{clean_symbol}_latest_decision.json"
    decision_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or reuse a per-symbol stock model and output a YES/NO buy decision."
    )
    parser.add_argument("symbol", help="Vietnam stock symbol, for example FPT or SSI")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore any saved model and retrain it")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()

    result = predict_symbol(args.symbol, force_retrain=args.force_retrain)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(result["decision"])
    print(
        f"{result['symbol']} | model={result['model_name']} | prob={result['probability']:.4f} "
        f"| threshold={result['threshold']:.4f} | effective_threshold={result['effective_threshold']:.4f} "
        f"| close={result['latest_close']:.2f} "
        f"| signal_date={result['latest_signal_date']} | trained_now={result['trained_now']}"
    )


if __name__ == "__main__":
    main()
