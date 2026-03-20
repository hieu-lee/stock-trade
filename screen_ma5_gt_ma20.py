#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

LONGEST_MA_WINDOW = 200
RECENT_CROSSOVER_DAYS = 2
DEFAULT_LOOKBACK_DAYS = 320
DEFAULT_VNSTOCK_REQUESTS_PER_MINUTE = 18.0
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 6.5
DEFAULT_MAX_RETRIES = 8
STRATEGY_TAG = f"ma_stack_crossup_last{RECENT_CROSSOVER_DAYS}d"
FIREANT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}
CHECKPOINT_COLUMNS = [
    "symbol",
    "status",
    "close",
    "ma5",
    "ma20",
    "ma50",
    "ma100",
    "ma200",
    "ma5_vs_ma20_spread_pct",
    "ma5_vs_ma200_spread_pct",
    "crossover_date",
    "error",
]


def import_vnstock():
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from vnstock import Listing, Quote
        return Listing, Quote
    except ImportError:
        # Some local editable installs point at `./vnstock/` but do not expose
        # the real package until that project root is added to sys.path.
        local_checkout = Path(__file__).resolve().parent / "vnstock"
        package_init = local_checkout / "vnstock" / "__init__.py"
        if not package_init.exists():
            raise

        sys.modules.pop("vnstock", None)
        sys.path.insert(0, str(local_checkout))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from vnstock import Listing, Quote
        return Listing, Quote


Listing, Quote = import_vnstock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Return symbols where MA200 < MA100 < MA50 < MA20 < MA5, "
            "with a bullish MA5/MA20 crossover in the last 2 trading days."
        )
    )
    parser.add_argument(
        "--history-provider",
        default="fireant",
        choices=["fireant", "vnstock"],
        help="Provider used for daily price history. Default: fireant",
    )
    parser.add_argument(
        "--source",
        default="VCI",
        help="vnstock quote source to use when --history-provider=vnstock. Default: VCI",
    )
    parser.add_argument(
        "--listing-source",
        default="VCI",
        help="vnstock listing source for auto-discovery. Default: VCI",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=(
            "Calendar-day buffer used to capture enough data for MA200 plus recent crossovers. "
            f"Default: {DEFAULT_LOOKBACK_DAYS}"
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional extra delay between requests. Default: 0",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=0.0,
        help=(
            "Auto-throttle target request rate. Use 0 to apply provider defaults "
            f"(currently {DEFAULT_VNSTOCK_REQUESTS_PER_MINUTE}/min for vnstock, unlimited for fireant)."
        ),
    )
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        help=(
            "Wait time before retrying after a rate-limit response. "
            f"Default: {DEFAULT_RATE_LIMIT_WAIT_SECONDS}"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per symbol after rate limits or transient errors. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbol list. If omitted, the script tries Listing().all_symbols().",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Optional text file with one symbol per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        help="Optional checkpoint CSV path for resume support.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start from scratch.",
    )
    return parser.parse_args()


def load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return normalize_symbols(args.symbols.split(","))

    if args.symbols_file:
        return normalize_symbols(args.symbols_file.read_text(encoding="utf-8").splitlines())

    frame = load_symbols_from_vnstock(args.listing_source)
    if frame is None or frame.empty or "symbol" not in frame.columns:
        frame = load_symbols_from_snapshot()

    if frame is None or frame.empty or "symbol" not in frame.columns:
        raise RuntimeError(
            "Could not load symbols from vnstock live listing or bundled snapshot. "
            "Try again later or pass --symbols / --symbols-file."
        )

    return normalize_symbols(frame["symbol"].astype(str).tolist())


def load_symbols_from_vnstock(source: str) -> pd.DataFrame | None:
    try:
        listing = Listing(source=source.lower(), show_log=False)
        frame = listing.all_symbols()
    except Exception:
        return None

    if frame is None or frame.empty or "symbol" not in frame.columns:
        return None
    return frame


def load_symbols_from_snapshot() -> pd.DataFrame | None:
    root = Path(__file__).resolve().parent
    snapshot_paths = [
        root / "vnstock" / "assets" / "data" / "all_symbols.csv",
        root / "vnstock" / "assets" / "data" / "symbols_by_exchange.csv",
    ]

    for path in snapshot_paths:
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            continue

        if {"exchange", "type"}.issubset(frame.columns):
            frame = frame[
                frame["type"].astype(str).str.upper().eq("STOCK")
                & frame["exchange"].astype(str).str.upper().isin({"HSX", "HOSE", "HNX", "UPCOM"})
            ].copy()

        if not frame.empty:
            return frame

    return None


def normalize_symbols(raw_symbols: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_symbols:
        symbol = str(item).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def resolve_checkpoint_path(args: argparse.Namespace, end: date) -> Path:
    if args.checkpoint_file:
        return args.checkpoint_file

    safe_provider = str(args.history_provider).lower()
    safe_source = str(args.source).lower() if safe_provider == "vnstock" else "default"
    return (
        Path(__file__).resolve().parent
        / "artifacts"
        / f"{STRATEGY_TAG}_{safe_provider}_{safe_source}_{end.isoformat()}.checkpoint.csv"
    )


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)

    frame = pd.read_csv(path)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CHECKPOINT_COLUMNS].copy()


def append_checkpoint_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_frame = pd.DataFrame([{column: row.get(column) for column in CHECKPOINT_COLUMNS}], columns=CHECKPOINT_COLUMNS)
    row_frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def checkpoint_to_results(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    if frame.empty:
        return []

    matched = frame[frame["status"] == "matched"].copy()
    if matched.empty:
        return []

    results: list[dict[str, float | str]] = []
    for _, row in matched.iterrows():
        results.append(
            {
                "symbol": str(row["symbol"]),
                "close": float(row["close"]),
                "ma5": float(row["ma5"]),
                "ma20": float(row["ma20"]),
                "ma50": float(row["ma50"]),
                "ma100": float(row["ma100"]),
                "ma200": float(row["ma200"]),
                "ma5_vs_ma20_spread_pct": float(row["ma5_vs_ma20_spread_pct"]),
                "ma5_vs_ma200_spread_pct": float(row["ma5_vs_ma200_spread_pct"]),
                "crossover_date": str(row["crossover_date"]),
            }
        )
    return results


def extract_error_message(exc: BaseException) -> str:
    return str(exc).strip()


def is_rate_limit_error(exc: BaseException) -> bool:
    message = extract_error_message(exc).lower()
    patterns = [
        "rate limit",
        "giới hạn api",
        "giới hạn tối đa",
        "retry after",
        "too many requests",
    ]
    return any(pattern in message for pattern in patterns)


def request_spacing_seconds(args: argparse.Namespace) -> float:
    if args.requests_per_minute > 0:
        return max(args.sleep_seconds, 60.0 / args.requests_per_minute)
    if args.history_provider == "vnstock":
        return max(args.sleep_seconds, 60.0 / DEFAULT_VNSTOCK_REQUESTS_PER_MINUTE)
    return max(args.sleep_seconds, 0.0)


def wait_for_slot(last_request_started_at: float | None, spacing_seconds: float) -> float:
    if spacing_seconds <= 0:
        return time.monotonic()

    if last_request_started_at is not None:
        elapsed = time.monotonic() - last_request_started_at
        if elapsed < spacing_seconds:
            time.sleep(spacing_seconds - elapsed)
    return time.monotonic()


def fetch_price_history_vnstock(symbol: str, source: str, start: str, end: str) -> pd.DataFrame:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            frame = Quote(symbol=symbol, source=source, show_log=False).history(
                start=start,
                end=end,
                interval="1D",
            )
    except SystemExit as exc:
        # `vnai` may abort the whole process via sys.exit(...) on rate-limit hits.
        # Convert that into a normal exception so the outer retry/checkpoint loop can continue.
        raise RuntimeError(extract_error_message(exc)) from exc
    return frame


def fetch_price_history_fireant(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://www.fireant.vn/api/Data/Companies/HistoricalQuotes"
        f"?symbol={symbol}&startDate={start}&endDate={end}"
    )
    response = requests.get(url, timeout=60, headers=FIREANT_HEADERS)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FireAnt payload for {symbol}: {payload}")
    if not payload:
        return pd.DataFrame(columns=["time", "close"])

    frame = pd.DataFrame(payload).rename(columns={"Date": "time", "PriceClose": "close"}).copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce").dt.tz_localize(None)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame[["time", "close"]]


def fetch_latest_signal(
    symbol: str,
    history_provider: str,
    source: str,
    start: str,
    end: str,
) -> dict[str, float | str] | None:
    if history_provider == "fireant":
        frame = fetch_price_history_fireant(symbol=symbol, start=start, end=end)
    else:
        frame = fetch_price_history_vnstock(symbol=symbol, source=source, start=start, end=end)

    if frame.empty:
        return None

    data = frame.copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
    if len(data) < LONGEST_MA_WINDOW:
        return None

    for window in [5, 20, 50, 100, 200]:
        data[f"ma{window}"] = data["close"].rolling(window).mean()

    valid = data.dropna(subset=["ma5", "ma20", "ma50", "ma100", "ma200"]).copy()
    if valid.empty:
        return None

    valid["cross_up"] = (valid["ma5"].shift(1) < valid["ma20"].shift(1)) & (valid["ma5"] > valid["ma20"])
    latest = valid.iloc[-1]
    stack_ok = latest["ma200"] < latest["ma100"] < latest["ma50"] < latest["ma20"] < latest["ma5"]
    if not stack_ok:
        return None

    recent_crossovers = valid.tail(RECENT_CROSSOVER_DAYS)
    crossover_rows = recent_crossovers[recent_crossovers["cross_up"]]
    if crossover_rows.empty:
        return None

    crossover_date = pd.Timestamp(crossover_rows.iloc[-1]["time"]).date().isoformat()
    ma5 = float(latest["ma5"])
    ma20 = float(latest["ma20"])
    ma50 = float(latest["ma50"])
    ma100 = float(latest["ma100"])
    ma200 = float(latest["ma200"])

    return {
        "symbol": symbol,
        "close": round(float(latest["close"]), 4),
        "ma5": round(ma5, 4),
        "ma20": round(ma20, 4),
        "ma50": round(ma50, 4),
        "ma100": round(ma100, 4),
        "ma200": round(ma200, 4),
        "ma5_vs_ma20_spread_pct": round((ma5 / ma20 - 1.0) * 100.0, 4),
        "ma5_vs_ma200_spread_pct": round((ma5 / ma200 - 1.0) * 100.0, 4),
        "crossover_date": crossover_date,
    }


def main() -> int:
    args = parse_args()
    end = date.today()
    start = end - timedelta(days=args.lookback_days)
    checkpoint_path = resolve_checkpoint_path(args, end)

    try:
        symbols = load_symbols(args)
    except Exception as exc:
        print(f"Failed to load symbols: {exc}", file=sys.stderr)
        return 1

    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    checkpoint_frame = load_checkpoint(checkpoint_path)
    processed_symbols = set(checkpoint_frame["symbol"].astype(str)) if not checkpoint_frame.empty else set()
    results = checkpoint_to_results(checkpoint_frame)
    failures = (
        checkpoint_frame.loc[checkpoint_frame["status"] == "failed", "symbol"].astype(str).tolist()
        if not checkpoint_frame.empty
        else []
    )
    remaining_symbols = [symbol for symbol in symbols if symbol not in processed_symbols]
    spacing_seconds = request_spacing_seconds(args)
    last_request_started_at: float | None = None

    if checkpoint_frame.empty:
        print(f"Starting fresh run for {len(symbols)} symbols.")
    else:
        print(
            f"Resuming from {checkpoint_path}: "
            f"{len(processed_symbols)} processed, {len(remaining_symbols)} remaining."
        )

    for offset, symbol in enumerate(remaining_symbols, start=1):
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                last_request_started_at = wait_for_slot(last_request_started_at, spacing_seconds)
                signal = fetch_latest_signal(
                    symbol=symbol,
                    history_provider=args.history_provider,
                    source=args.source,
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
                if signal:
                    results.append(signal)
                    append_checkpoint_row(
                        checkpoint_path,
                        {
                            "symbol": symbol,
                            "status": "matched",
                            "close": signal["close"],
                            "ma5": signal["ma5"],
                            "ma20": signal["ma20"],
                            "ma50": signal["ma50"],
                            "ma100": signal["ma100"],
                            "ma200": signal["ma200"],
                            "ma5_vs_ma20_spread_pct": signal["ma5_vs_ma20_spread_pct"],
                            "ma5_vs_ma200_spread_pct": signal["ma5_vs_ma200_spread_pct"],
                            "crossover_date": signal["crossover_date"],
                            "error": "",
                        },
                    )
                else:
                    append_checkpoint_row(
                        checkpoint_path,
                        {
                            "symbol": symbol,
                            "status": "filtered",
                            "close": pd.NA,
                            "ma5": pd.NA,
                            "ma20": pd.NA,
                            "ma50": pd.NA,
                            "ma100": pd.NA,
                            "ma200": pd.NA,
                            "ma5_vs_ma20_spread_pct": pd.NA,
                            "ma5_vs_ma200_spread_pct": pd.NA,
                            "crossover_date": pd.NA,
                            "error": "",
                        },
                    )
                break
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise

                last_error = extract_error_message(exc)
                if attempt < args.max_retries and is_rate_limit_error(exc):
                    time.sleep(max(args.rate_limit_wait_seconds, spacing_seconds))
                    continue
                if attempt < args.max_retries and "timed out" in last_error.lower():
                    time.sleep(max(args.rate_limit_wait_seconds, spacing_seconds))
                    continue

                failures.append(symbol)
                append_checkpoint_row(
                    checkpoint_path,
                    {
                        "symbol": symbol,
                        "status": "failed",
                        "close": pd.NA,
                        "ma5": pd.NA,
                        "ma20": pd.NA,
                        "ma50": pd.NA,
                        "ma100": pd.NA,
                        "ma200": pd.NA,
                        "ma5_vs_ma20_spread_pct": pd.NA,
                        "ma5_vs_ma200_spread_pct": pd.NA,
                        "crossover_date": pd.NA,
                        "error": last_error,
                    },
                )
                break

        total_done = len(processed_symbols) + offset
        if total_done % 25 == 0 or total_done == len(symbols):
            print(
                f"Progress: {total_done}/{len(symbols)} processed | "
                f"matches={len(results)} | failed={len(failures)}"
            )

    result_frame = pd.DataFrame(
        results,
        columns=[
            "symbol",
            "close",
            "ma5",
            "ma20",
            "ma50",
            "ma100",
            "ma200",
            "ma5_vs_ma20_spread_pct",
            "ma5_vs_ma200_spread_pct",
            "crossover_date",
        ],
    )
    if not result_frame.empty:
        result_frame = result_frame.sort_values(
            by=["crossover_date", "ma5_vs_ma20_spread_pct", "symbol"],
            ascending=[False, False, True],
        )

    if args.output:
        result_frame.to_csv(args.output, index=False)

    if result_frame.empty:
        print("No symbols found matching the MA stack and recent MA5/MA20 bullish crossover.")
    else:
        print(result_frame.to_string(index=False))

    print(
        f"\nMatched {len(result_frame)} / {len(symbols)} symbols"
        + (f" | failed: {len(failures)}" if failures else "")
    )

    if failures:
        print("Failed symbols:", ", ".join(failures[:20]), file=sys.stderr)
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
