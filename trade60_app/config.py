from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vn30_strategy.config import ALL60_SYMBOLS, SYMBOL_LABELS


@dataclass(slots=True)
class Trade60Config:
    workspace_dir: Path
    start_date: str = "2006-01-01"
    end_date: str = "today"
    price_source: str = "vci"
    benchmark_symbols: tuple[str, ...] = ("VNINDEX",)
    initial_budget: float = 100_000.0
    commission_bps: float = 10.0
    slippage_bps: float = 15.0
    max_positions: int = 5
    max_holding_days: int = 40
    min_holding_days: int = 2
    entry_threshold: float = 0.58
    exit_threshold: float = 0.46
    regime_threshold: float = 0.53
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.18
    hold_alpha_buffer: float = 0.06
    rank_keep_fraction: float = 1.0
    defensive_trim_fraction: float = 0.35
    weak_alpha_trim_fraction: float = 0.5
    profit_trim_fraction: float = 0.5
    target_annual_return: float = 0.08
    validation_days: int = 504
    holdout_days: int = 504
    random_state: int = 42
    symbols: tuple[str, ...] = field(default_factory=lambda: tuple(ALL60_SYMBOLS))

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir).resolve()
        if self.end_date == "today":
            self.end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        self.symbols = tuple(self.symbols)

    @property
    def app_dir(self) -> Path:
        return self.workspace_dir / "trade60_app"

    @property
    def raw_dir(self) -> Path:
        return self.app_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.app_dir / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.app_dir / "artifacts"

    @property
    def price_dir(self) -> Path:
        return self.raw_dir / "prices"

    @property
    def benchmark_dir(self) -> Path:
        return self.raw_dir / "benchmarks"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def recommendations_dir(self) -> Path:
        return self.artifacts_dir / "recommendations"

    @property
    def feature_store_path(self) -> Path:
        return self.processed_dir / "daily_feature_panel.parquet"

    @property
    def universe_path(self) -> Path:
        return self.raw_dir / "universe.parquet"

    @property
    def prices_path(self) -> Path:
        return self.raw_dir / "prices.parquet"

    @property
    def benchmarks_path(self) -> Path:
        return self.raw_dir / "benchmarks.parquet"


def get_default_config(workspace_dir: str | Path | None = None) -> Trade60Config:
    base = Path(workspace_dir) if workspace_dir else Path(__file__).resolve().parents[1]
    return Trade60Config(workspace_dir=base)


def build_symbol_frame(symbols: tuple[str, ...] | None = None) -> pd.DataFrame:
    chosen = tuple(symbols) if symbols else tuple(ALL60_SYMBOLS)
    return pd.DataFrame(
        {
            "symbol": list(chosen),
            "display_name": [SYMBOL_LABELS.get(symbol, symbol) for symbol in chosen],
        }
    )
