from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


VN30_CORE_SYMBOLS = [
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
]

NEW30_SYMBOLS = [
    "BVS",
    "CAP",
    "CEO",
    "DHT",
    "DP3",
    "DTD",
    "DVM",
    "DXP",
    "HGM",
    "HUT",
    "IDC",
    "IDV",
    "L14",
    "L18",
    "LAS",
    "LHC",
    "MBS",
    "NTP",
    "PLC",
    "PSD",
    "PVB",
    "PVC",
    "PVI",
    "PVS",
    "SHS",
    "SLS",
    "TMB",
    "TNG",
    "VC3",
    "VCS",
]

ALL60_SYMBOLS = VN30_CORE_SYMBOLS + NEW30_SYMBOLS

UNIVERSE_SYMBOLS = {
    "vn30_core": tuple(VN30_CORE_SYMBOLS),
    "new30": tuple(NEW30_SYMBOLS),
    "all60": tuple(ALL60_SYMBOLS),
}

UNIVERSE_TITLES = {
    "vn30_core": "VN30 Core",
    "new30": "New 30",
    "all60": "All 60",
}

SYMBOL_LABELS = {
    "ACB": "Ngân hàng TMCP Á Châu",
    "BCM": "Becamex",
    "BID": "BIDV",
    "BVH": "Bảo Việt",
    "BVS": "BVS",
    "CAP": "CAP",
    "CEO": "CEO",
    "CTG": "VietinBank",
    "DHT": "DHT",
    "DP3": "DP3",
    "DTD": "DTD",
    "DVM": "DVM",
    "DXP": "DXP",
    "FPT": "FPT",
    "GAS": "PV Gas",
    "GVR": "Cao su Việt Nam",
    "HGM": "HGM",
    "HDB": "HDBank",
    "HPG": "Hòa Phát",
    "HUT": "HUT",
    "IDC": "IDC",
    "IDV": "IDV",
    "L14": "L14",
    "L18": "L18",
    "LAS": "LAS",
    "LHC": "LHC",
    "LPB": "LPBank",
    "MBB": "MB",
    "MBS": "MBS",
    "MSN": "Masan",
    "MWG": "Thế Giới Di Động",
    "NTP": "NTP",
    "PLC": "PLC",
    "PLX": "Petrolimex",
    "PSD": "PSD",
    "PVB": "PVB",
    "PVC": "PVC",
    "PVI": "PVI",
    "PVS": "PVS",
    "SAB": "Sabeco",
    "SHB": "SHB",
    "SHS": "SHS",
    "SSB": "SeABank",
    "SLS": "SLS",
    "SSI": "SSI",
    "STB": "Sacombank",
    "TCB": "Techcombank",
    "TMB": "TMB",
    "TNG": "TNG",
    "TPB": "TPBank",
    "VC3": "VC3",
    "VCB": "Vietcombank",
    "VCS": "VCS",
    "VHM": "Vinhomes",
    "VIB": "VIB",
    "VIC": "Vingroup",
    "VJC": "VietJet",
    "VNM": "Vinamilk",
    "VPB": "VPBank",
    "VRE": "Vincom Retail",
}


@dataclass(slots=True)
class StrategyConfig:
    workspace_dir: Path
    profile_name: str = "vn30_core"
    start_date: str = "2006-01-01"
    end_date: str = "today"
    price_source: str = "vci"
    finance_source: str = "VCI"
    company_source: str = "VCI"
    benchmark_symbols: tuple[str, ...] = ("VNINDEX", "VN30")
    target_return: float = 0.10
    target_holding_days: int = 20
    commission_bps: float = 10.0
    slippage_bps: float = 15.0
    max_positions: int = 1
    regime_threshold: float = 0.52
    rank_threshold: float = 0.52
    min_validation_traded_months: int = 4
    target_validation_traded_share: float = 0.35
    optuna_trials: int = 30
    random_state: int = 42
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        if self.profile_name not in UNIVERSE_SYMBOLS:
            raise ValueError(f"Unknown profile_name: {self.profile_name}")
        if self.end_date == "today":
            self.end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        if not self.symbols:
            self.symbols = UNIVERSE_SYMBOLS[self.profile_name]
        else:
            self.symbols = tuple(self.symbols)

    @property
    def package_dir(self) -> Path:
        return self.workspace_dir / "vn30_strategy"

    @property
    def data_dir(self) -> Path:
        return self.workspace_dir / "data" / self.profile_name

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace_dir / "artifacts" / self.profile_name

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
    def price_dir(self) -> Path:
        return self.raw_dir / "prices"

    @property
    def finance_dir(self) -> Path:
        return self.raw_dir / "financials"

    @property
    def company_dir(self) -> Path:
        return self.raw_dir / "company"

    @property
    def dividends_dir(self) -> Path:
        return self.raw_dir / "dividends"

    @property
    def benchmark_dir(self) -> Path:
        return self.raw_dir / "benchmarks"

    @property
    def profile_title(self) -> str:
        return UNIVERSE_TITLES[self.profile_name]


def get_default_config(workspace_dir: str | Path | None = None, profile_name: str = "vn30_core") -> StrategyConfig:
    base = Path(workspace_dir) if workspace_dir else Path(__file__).resolve().parents[1]
    return StrategyConfig(workspace_dir=base, profile_name=profile_name)
