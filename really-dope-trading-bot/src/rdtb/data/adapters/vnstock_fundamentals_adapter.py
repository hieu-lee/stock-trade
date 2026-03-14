from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from calendar import monthrange

import pandas as pd


RATIO_ALIASES = {
    "Debt/Equity": "debt_to_equity",
    "Current Ratio": "current_ratio",
    "Quick Ratio": "quick_ratio",
    "Cash Ratio": "cash_ratio",
    "Interest Coverage": "interest_coverage",
    "Financial Leverage": "financial_leverage",
    "Asset Turnover": "asset_turnover",
    "Inventory Turnover": "inventory_turnover",
    "EBIT Margin (%)": "ebit_margin_pct",
    "Gross Profit Margin (%)": "gross_profit_margin_pct",
    "Net Profit Margin (%)": "net_profit_margin_pct",
    "ROE (%)": "roe_pct",
    "ROIC (%)": "roic_pct",
    "ROA (%)": "roa_pct",
    "Dividend yield (%)": "dividend_yield_pct",
    "Market Capital (Bn. VND)": "market_cap_bn_vnd",
    "Outstanding Share (Mil. Shares)": "outstanding_share_mil",
    "P/E": "pe_ratio",
    "P/B": "pb_ratio",
    "P/S": "ps_ratio",
    "P/Cash Flow": "pcf_ratio",
    "EPS (VND)": "eps_vnd",
    "BVPS (VND)": "bvps_vnd",
}

INCOME_ALIASES = {
    "Revenue YoY (%)": "revenue_yoy_pct",
    "Revenue (Bn. VND)": "revenue_bn_vnd",
    "Attribute to parent company (Bn. VND)": "parent_profit_bn_vnd",
    "Attribute to parent company YoY (%)": "parent_profit_yoy_pct",
    "Operating Profit/Loss": "operating_profit_vnd",
    "Profit before tax": "profit_before_tax_vnd",
}

CASHFLOW_ALIASES = {
    "Net cash inflows/outflows from operating activities": "operating_cash_flow_vnd",
    "Purchase of fixed assets": "capex_vnd",
    "Net Cash Flows from Investing Activities": "investing_cash_flow_vnd",
    "Cash flows from financial activities": "financing_cash_flow_vnd",
}


@dataclass(slots=True)
class VnstockFundamentalsAdapter:
    source: str = "VCI"
    min_request_spacing_seconds: float = 3.4
    retry_cooldown_seconds: float = 75.0
    max_retries: int = 4
    _last_request_at: float = field(init=False, default=0.0)

    def fetch_quarterly_fundamentals(self, symbol: str, quarterly_lag_days: int = 45, annual_lag_days: int = 60) -> pd.DataFrame:
        try:
            from vnstock import Finance
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `vnstock` package is required for quarterly fundamentals.") from exc

        finance = Finance(symbol=symbol, source=self.source)
        ratio = self._normalize_quarter_frame(
            self._invoke(lambda: finance.ratio(period="quarter"), context=f"ratio:{symbol}"),
            symbol=symbol,
            aliases=RATIO_ALIASES,
        )
        income = self._normalize_quarter_frame(
            self._invoke(lambda: finance.income_statement(period="quarter"), context=f"income:{symbol}"),
            symbol=symbol,
            aliases=INCOME_ALIASES,
        )
        merged = ratio.merge(income, on=["symbol", "fiscal_year", "fiscal_quarter", "report_period_end"], how="outer")
        merged["available_date"] = merged.apply(
            lambda row: row["report_period_end"] + pd.Timedelta(days=annual_lag_days if int(row["fiscal_quarter"]) == 4 else quarterly_lag_days),
            axis=1,
        )
        merged["source"] = self.source.upper()
        return merged.sort_values(["symbol", "available_date"]).reset_index(drop=True)

    def fetch_company_overview(self, symbol: str) -> pd.DataFrame:
        try:
            from vnstock import Company
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `vnstock` package is required for company metadata.") from exc

        company = Company(symbol=symbol, source=self.source)
        overview = self._invoke(lambda: company.overview().copy(), context=f"overview:{symbol}")
        if overview.empty:
            return pd.DataFrame(columns=["symbol", "industry_level2", "industry_level3", "issue_share", "charter_capital"])
        renamed = overview.rename(
            columns={
                "symbol": "symbol",
                "icb_name2": "industry_level2",
                "icb_name3": "industry_level3",
                "issue_share": "issue_share",
                "charter_capital": "charter_capital",
            }
        )
        for column in ["issue_share", "charter_capital"]:
            if column in renamed.columns:
                renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        columns = [column for column in ["symbol", "industry_level2", "industry_level3", "issue_share", "charter_capital"] if column in renamed.columns]
        return renamed[columns].drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    def fetch_company_events(self, symbol: str) -> pd.DataFrame:
        try:
            from vnstock import Company
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `vnstock` package is required for company events.") from exc

        company = Company(symbol=symbol, source=self.source)
        events = self._invoke(lambda: company.events().copy(), context=f"events:{symbol}")
        if events.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "available_date",
                    "issue_date",
                    "record_date",
                    "exright_date",
                    "event_code",
                    "event_name",
                    "ratio",
                    "value",
                    "source",
                ]
            )
        normalized = events.rename(
            columns={
                "event_list_code": "event_code",
                "event_list_name": "event_name",
                "public_date": "available_date",
            }
        ).copy()
        normalized["symbol"] = symbol
        for column in ["available_date", "issue_date", "record_date", "exright_date"]:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
            normalized.loc[normalized[column].dt.year <= 1800, column] = pd.NaT
            normalized[column] = normalized[column].dt.tz_localize(None)
        normalized["ratio"] = pd.to_numeric(normalized.get("ratio"), errors="coerce")
        normalized["value"] = pd.to_numeric(normalized.get("value"), errors="coerce")
        normalized["source"] = self.source.upper()
        columns = ["symbol", "available_date", "issue_date", "record_date", "exright_date", "event_code", "event_name", "ratio", "value", "source"]
        normalized = normalized[columns].dropna(subset=["available_date"]).drop_duplicates(
            subset=["symbol", "available_date", "event_code", "event_name"],
            keep="last",
        )
        return normalized.sort_values(["symbol", "available_date"]).reset_index(drop=True)

    def _normalize_quarter_frame(self, frame: pd.DataFrame, symbol: str, aliases: dict[str, str]) -> pd.DataFrame:
        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = [self._flatten_column_name(column) for column in normalized.columns]
        normalized = normalized.rename(columns={"ticker": "symbol", "yearReport": "fiscal_year", "lengthReport": "fiscal_quarter"})
        normalized["symbol"] = symbol
        normalized["fiscal_year"] = pd.to_numeric(normalized["fiscal_year"], errors="coerce")
        normalized["fiscal_quarter"] = pd.to_numeric(normalized["fiscal_quarter"], errors="coerce")
        keep_columns = ["symbol", "fiscal_year", "fiscal_quarter"] + [column for column in aliases if column in normalized.columns]
        trimmed = normalized[keep_columns].copy()
        for source, target in aliases.items():
            if source in trimmed.columns:
                trimmed[target] = pd.to_numeric(trimmed[source], errors="coerce")
        trimmed["report_period_end"] = trimmed.apply(self._quarter_end_timestamp, axis=1)
        final_columns = ["symbol", "fiscal_year", "fiscal_quarter", "report_period_end"] + list(aliases.values())
        available = [column for column in final_columns if column in trimmed.columns]
        return trimmed[available].drop_duplicates(subset=["symbol", "fiscal_year", "fiscal_quarter"]).reset_index(drop=True)

    def _invoke(self, fn, context: str):
        for _ in range(self.max_retries):
            self._respect_rate_limit()
            try:
                result = fn()
                self._last_request_at = time.time()
                return result
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # pragma: no cover - network dependent
                message = str(exc)
                if (
                    "Rate limit exceeded" in message
                    or "GIỚI HẠN API" in message
                    or "RetryError" in message
                    or "Process terminated" in message
                ):
                    time.sleep(self._retry_wait_seconds(message))
                    self._last_request_at = time.time()
                    continue
                raise RuntimeError(f"Failed to fetch {context}") from exc
        raise RuntimeError(f"Repeated rate-limit failures while fetching {context}.")

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_spacing_seconds:
            time.sleep(self.min_request_spacing_seconds - elapsed)

    def _retry_wait_seconds(self, message: str) -> float:
        match = re.search(r"Chờ\\s+(\\d+)\\s+giây", message)
        if match:
            return max(float(match.group(1)) + 1.0, self.min_request_spacing_seconds)
        return self.retry_cooldown_seconds

    @staticmethod
    def _flatten_column_name(column) -> str:
        if not isinstance(column, tuple):
            return str(column)
        last = str(column[-1]).strip()
        if last:
            return last
        return str(column[0]).strip()

    @staticmethod
    def _quarter_end_timestamp(row: pd.Series) -> pd.Timestamp:
        year = int(row["fiscal_year"])
        quarter = int(row["fiscal_quarter"])
        month = min(max(quarter, 1), 4) * 3
        day = monthrange(year, month)[1]
        return pd.Timestamp(year=year, month=month, day=day)
