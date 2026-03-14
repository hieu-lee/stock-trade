from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rdtb.universe import DEFAULT_BENCHMARKS, VN60_SYMBOLS


@dataclass(slots=True)
class TradingBotConfig:
    project_dir: Path
    start_date: str = "2006-01-01"
    end_date: str = "today"
    symbols: tuple[str, ...] = field(default_factory=lambda: tuple(VN60_SYMBOLS))
    benchmark_symbols: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_BENCHMARKS))
    external_symbols: tuple[str, ...] = ("SPY", "QQQ", "EEM", "FXI", "TLT", "GLD", "UUP")
    primary_price_source: str = "VCI"
    finance_source: str = "VCI"
    use_company_metadata_features: bool = False
    use_fireant_flow_features: bool = True
    use_event_features: bool = True
    commission_bps: float = 10.0
    slippage_bps: float = 15.0
    buy_transaction_fee_bps: float = 3.0
    sell_transaction_fee_bps: float = 13.0
    buy_settlement_days: int = 3
    sell_settlement_days: int = 2
    initial_cash: float = 1_000_000_000.0
    lot_size: int = 100
    max_positions: int = 8
    max_weight: float = 0.18
    min_trade_weight_delta: float = 0.015
    hold_buffer: float = 0.02
    add_buffer: float = 0.05
    trim_buffer: float = 0.04
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.22
    max_holding_days: int = 60
    min_holding_days: int = 2
    min_history_days: int = 220
    alpha_horizon_days: int = 20
    risk_horizon_days: int = 10
    regime_horizon_days: int = 20
    lookahead_days: tuple[int, ...] = (5, 10, 20)
    alpha_target_quantile: float = 0.80
    quarterly_report_lag_days: int = 45
    annual_report_lag_days: int = 60
    optimizer_risk_penalty: float = 0.40
    optimizer_turnover_penalty: float = 0.20
    optimizer_concentration_penalty: float = 0.06
    optimizer_cash_floor: float = 0.08
    optimizer_backend: str = "convex"
    regime_risk_on_threshold: float = 0.55
    regime_defensive_threshold: float = 0.45
    risk_reject_threshold: float = 0.55
    buy_score_threshold: float = 0.58
    add_score_threshold: float = 0.68
    exit_score_threshold: float = 0.44
    trim_score_threshold: float = 0.50
    deployment_min_year_return: float = 0.30
    deployment_max_drawdown: float = -0.15
    development_start_year: int = 2006
    development_end_year: int = 2023
    final_test_years: tuple[int, ...] = (2024, 2025)
    paper_trade_year: int = 2026
    fold_validation_years: int = 1
    fold_step_years: int = 1
    fold_min_train_years: int = 5
    max_train_years: int = 50
    fold_embargo_days: int = 20
    alpha_half_life_years: float = 25.0
    risk_half_life_years: float = 25.0
    regime_half_life_years: float = 20.0
    optuna_trials: int = 8
    optuna_timeout_seconds: int = 180
    random_state: int = 42

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        if self.end_date == "today":
            self.end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        self.symbols = tuple(self.symbols)
        self.benchmark_symbols = tuple(self.benchmark_symbols)

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def manual_import_dir(self) -> Path:
        return self.data_dir / "manual_imports"

    @property
    def price_dir(self) -> Path:
        return self.raw_dir / "prices"

    @property
    def benchmark_dir(self) -> Path:
        return self.raw_dir / "benchmarks"

    @property
    def external_dir(self) -> Path:
        return self.raw_dir / "external"

    @property
    def fundamentals_dir(self) -> Path:
        return self.raw_dir / "fundamentals"

    @property
    def company_dir(self) -> Path:
        return self.raw_dir / "company"

    @property
    def flow_dir(self) -> Path:
        return self.raw_dir / "flow"

    @property
    def events_dir(self) -> Path:
        return self.raw_dir / "events"

    @property
    def listings_path(self) -> Path:
        return self.raw_dir / "listings.parquet"

    @property
    def universe_path(self) -> Path:
        return self.raw_dir / "universe.parquet"

    @property
    def prices_path(self) -> Path:
        return self.raw_dir / "prices.parquet"

    @property
    def benchmarks_path(self) -> Path:
        return self.raw_dir / "benchmarks.parquet"

    @property
    def external_markets_path(self) -> Path:
        return self.raw_dir / "external_markets.parquet"

    @property
    def fundamentals_path(self) -> Path:
        return self.raw_dir / "fundamentals.parquet"

    @property
    def company_metadata_path(self) -> Path:
        return self.raw_dir / "company_metadata.parquet"

    @property
    def flow_path(self) -> Path:
        return self.raw_dir / "flow.parquet"

    @property
    def events_path(self) -> Path:
        return self.raw_dir / "events.parquet"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "market.duckdb"

    @property
    def feature_panel_path(self) -> Path:
        return self.processed_dir / "daily_feature_panel.parquet"

    @property
    def scored_panel_path(self) -> Path:
        return self.processed_dir / "daily_scored_panel.parquet"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_dir / "artifacts"

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
    def training_summary_path(self) -> Path:
        return self.reports_dir / "training_summary.json"

    @property
    def deployment_manifest_path(self) -> Path:
        return self.models_dir / "deployment_manifest.json"

    @property
    def alpha_model_path(self) -> Path:
        return self.models_dir / "alpha_model.joblib"

    @property
    def risk_model_path(self) -> Path:
        return self.models_dir / "risk_model.joblib"

    @property
    def regime_model_path(self) -> Path:
        return self.models_dir / "regime_model.joblib"

    @property
    def latest_decision_path(self) -> Path:
        return self.recommendations_dir / "latest_recommendation.json"


def get_default_config(project_dir: str | Path | None = None) -> TradingBotConfig:
    base = Path(project_dir) if project_dir else Path(__file__).resolve().parents[2]
    return TradingBotConfig(project_dir=base)
