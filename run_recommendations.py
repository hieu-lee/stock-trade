#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from vn30_strategy.backtest.walkforward import score_unlabeled_panel
from vn30_strategy.config import UNIVERSE_TITLES, UNIVERSE_SYMBOLS, get_default_config
from vn30_strategy.service.pipeline import _explain_row, build_feature_store, load_trained_artifacts, train_strategy
from vn30_strategy.utils import read_json


COMBINED_PROFILE = "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the latest VN30 recommendation report without forcing a fresh data pull.")
    parser.add_argument("--workspace", default=Path(__file__).resolve().parent, help="Workspace directory")
    profile_choices = sorted(UNIVERSE_SYMBOLS) + [COMBINED_PROFILE]
    parser.add_argument("--profile", default=COMBINED_PROFILE, choices=profile_choices, help="Universe profile to run")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh market data from vnstock before generating the report")
    parser.add_argument("--retrain", action="store_true", help="Retrain the strategy before generating the report")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the generated HTML report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    profile_names = _resolve_profile_names(args.profile)
    profile_reports = [_prepare_profile_report(workspace, profile_name, args) for profile_name in profile_names]

    if len(profile_reports) == 1:
        report_path = _write_single_profile_report(profile_reports[0], refreshed_data=args.refresh_data)
    else:
        report_path = _write_combined_html_report(workspace, profile_reports, refreshed_data=args.refresh_data)

    print(f"Report saved to: {report_path}")
    if not args.no_open:
        webbrowser.open(report_path.resolve().as_uri())
        print("Opened the report in your browser.")


def _has_trained_models(config) -> bool:
    return (
        (config.models_dir / "ranker.joblib").exists()
        and (config.models_dir / "regime.joblib").exists()
        and (config.models_dir / "best_params.json").exists()
    )


def _load_existing_summary(config) -> dict:
    summary_path = config.artifacts_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    return {
        "holdout_metrics": {},
        "in_sample_metrics": {},
        "validation_summary": {},
    }


def _load_or_build_panel(config, refresh_data: bool) -> pd.DataFrame:
    panel_path = config.processed_dir / "monthly_feature_panel.parquet"
    if refresh_data:
        print("Refreshing feature store with the latest market data...")
        return build_feature_store(config, refresh_data=True)
    if panel_path.exists():
        print("Using cached feature store to avoid API rate-limit disruption...")
        return pd.read_parquet(panel_path)
    print("No cached feature store found. Building from cached raw data...")
    return build_feature_store(config, refresh_data=False)


def _resolve_profile_names(profile_name: str) -> list[str]:
    if profile_name == COMBINED_PROFILE:
        return ["vn30_core", "new30"]
    return [profile_name]


def _prepare_profile_report(workspace: Path, profile_name: str, args: argparse.Namespace) -> dict:
    config = get_default_config(workspace, profile_name=profile_name)
    if args.retrain or not _has_trained_models(config):
        print(f"Training strategy artifacts for {config.profile_title}...")
        summary = train_strategy(config, refresh_data=args.refresh_data)
        panel = _load_or_build_panel(config, refresh_data=False)
    else:
        summary = _load_existing_summary(config)
        panel = _load_or_build_panel(config, refresh_data=args.refresh_data)
    ranker, regime = load_trained_artifacts(config)
    best_params = read_json(config.models_dir / "best_params.json")
    latest_date, scored = _score_latest_month(panel, ranker, regime, best_params)
    scored["profile_name"] = config.profile_name
    scored["profile_title"] = config.profile_title
    return {
        "config": config,
        "summary": summary,
        "best_params": best_params,
        "latest_date": latest_date,
        "scored": scored,
    }


def _score_latest_month(panel: pd.DataFrame, ranker, regime, best_params: dict) -> tuple[pd.Timestamp, pd.DataFrame]:
    unlabeled = panel.loc[~panel["is_trainable"]].copy()
    if unlabeled.empty:
        unlabeled = panel.groupby("symbol", as_index=False).tail(1).copy()
    latest_date = pd.to_datetime(unlabeled["date"]).max()
    latest_panel = unlabeled.loc[pd.to_datetime(unlabeled["date"]) == latest_date].copy()
    scored = score_unlabeled_panel(latest_panel, regime, ranker).sort_values("rank_probability", ascending=False).reset_index(drop=True)

    regime_threshold = float(best_params["regime_threshold"])
    rank_threshold = float(best_params["rank_threshold"])
    max_positions = int(best_params["max_positions"])

    scored["actionable"] = (scored["regime_probability"] >= regime_threshold) & (scored["rank_probability"] >= rank_threshold)
    scored["selected"] = False
    if scored["actionable"].any():
        chosen = scored.loc[scored["actionable"]].head(max_positions).index
        scored.loc[chosen, "selected"] = True

    scored["status"] = np.where(
        scored["selected"],
        "Recommend",
        np.where(scored["actionable"], "Qualified but outside position cap", "Watchlist"),
    )
    scored["explanation"] = scored.apply(_explain_row, axis=1)
    return latest_date, scored


def _write_single_profile_report(profile_report: dict, refreshed_data: bool) -> Path:
    return _write_html_report(
        profile_report["config"],
        profile_report["summary"],
        profile_report["best_params"],
        profile_report["latest_date"],
        profile_report["scored"],
        refreshed_data=refreshed_data,
    )


def _write_html_report(config, summary: dict, best_params: dict, latest_date: pd.Timestamp, scored: pd.DataFrame, refreshed_data: bool) -> Path:
    report_path = config.recommendations_dir / "latest_recommendations.html"
    selected = scored.loc[scored["selected"]].copy()
    watchlist = scored.head(10).copy()
    holdout = summary.get("holdout_metrics", {})

    for frame in (selected, watchlist):
        for column in ["rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "pe" in frame.columns:
            frame["pe"] = pd.to_numeric(frame["pe"], errors="coerce")

    selected_table = _render_table(
        selected,
        ["symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
        empty_message="No actionable recommendation right now. Stay in cash and review the watchlist below.",
    )
    watchlist_table = _render_table(
        watchlist,
        ["symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
        empty_message="No watchlist rows were generated.",
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VN30 Recommendation Brief</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero, .card {{
      background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 18px;
      box-shadow: 0 14px 35px rgba(2, 6, 23, 0.35);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      margin: 6px 0;
      color: #cbd5e1;
      line-height: 1.5;
    }}
    .badge {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      font-weight: 700;
      margin-top: 10px;
      background: {"#14532d" if holdout.get("pass_80pct_requirement") else "#7c2d12"};
      color: #f8fafc;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .metric {{
      padding: 18px;
    }}
    .metric .label {{
      color: #94a3b8;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric .value {{
      display: block;
      margin-top: 8px;
      font-size: 30px;
      font-weight: 800;
      color: #f8fafc;
    }}
    .section {{
      margin-top: 22px;
      padding: 22px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      overflow: hidden;
      border-radius: 12px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.16);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: #93c5fd;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .recommend {{
      color: #86efac;
      font-weight: 700;
    }}
    .watch {{
      color: #fcd34d;
      font-weight: 700;
    }}
    .muted {{
      color: #94a3b8;
    }}
    .note {{
      margin-top: 12px;
      font-size: 13px;
      color: #94a3b8;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>{html.escape(config.profile_title)} Recommendation Brief</h1>
      <p>Latest monthly snapshot: <strong>{latest_date.strftime("%Y-%m-%d")}</strong></p>
      <p>This page scores the newest available month-end snapshot and highlights only the names that clear the live regime and ranking thresholds.</p>
      <p>Data mode: <strong>{'Fresh vnstock refresh' if refreshed_data else 'Cached local data'}</strong></p>
      <div class="badge">Holdout target pass: {holdout.get("pass_80pct_requirement", False)}</div>
    </div>

    <div class="metrics">
      <div class="card metric"><span class="label">Holdout Hit Rate</span><span class="value">{_format_pct(holdout.get("monthly_hit_rate"))}</span></div>
      <div class="card metric"><span class="label">Holdout Traded Months</span><span class="value">{holdout.get("months_traded", 0)}</span></div>
      <div class="card metric"><span class="label">Avg Traded-Month Return</span><span class="value">{_format_pct(holdout.get("average_monthly_return"))}</span></div>
      <div class="card metric"><span class="label">Live Rank Threshold</span><span class="value">{_format_pct(best_params.get("rank_threshold"))}</span></div>
      <div class="card metric"><span class="label">Live Regime Threshold</span><span class="value">{_format_pct(best_params.get("regime_threshold"))}</span></div>
      <div class="card metric"><span class="label">Max Positions</span><span class="value">{best_params.get("max_positions", 0)}</span></div>
    </div>

    <div class="card section">
      <h2>Actionable Recommendations</h2>
      <p>These are the names that passed both the market-regime gate and the stock-ranking threshold.</p>
      {selected_table}
    </div>

    <div class="card section">
      <h2>Top Watchlist</h2>
      <p>These are the strongest names in the latest ranking even if they did not all clear the final execution filter.</p>
      {watchlist_table}
    </div>

    <div class="card section">
      <h2>How To Read This</h2>
      <p><strong>Rank probability</strong>: the model's estimate that the stock can reach at least 10% over the next 20 trading days.</p>
      <p><strong>Regime probability</strong>: how favorable the overall month looks for taking risk.</p>
      <p><strong>Status</strong>: <span class="recommend">Recommend</span> means tradable now, while <span class="watch">Watchlist</span> means monitor only.</p>
      <p class="note">Because this is a monthly-rotation strategy, you should mainly care about the latest month-end snapshot rather than intraday noise.</p>
    </div>
  </div>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _write_combined_html_report(workspace: Path, profile_reports: list[dict], refreshed_data: bool) -> Path:
    combined_dir = workspace / "artifacts" / "combined" / "recommendations"
    combined_dir.mkdir(parents=True, exist_ok=True)
    report_path = combined_dir / "latest_recommendations.html"

    latest_date = max(report["latest_date"] for report in profile_reports)
    combined_scored = pd.concat([report["scored"] for report in profile_reports], ignore_index=True)
    combined_selected = combined_scored.loc[combined_scored["selected"]].copy()
    combined_watchlist = combined_scored.loc[:, [
        "profile_title",
        "symbol",
        "status",
        "rank_probability",
        "regime_probability",
        "ret_20d",
        "ret_60d",
        "distance_ma200",
        "roe",
        "pe",
        "explanation",
    ]].copy()
    combined_watchlist = combined_watchlist.sort_values(["profile_title", "rank_probability"], ascending=[True, False]).reset_index(drop=True)

    for frame in (combined_selected, combined_watchlist):
        for column in ["rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "pe" in frame.columns:
            frame["pe"] = pd.to_numeric(frame["pe"], errors="coerce")

    profile_sections = []
    summary_cards = []
    for report in profile_reports:
        config = report["config"]
        summary = report["summary"]
        best_params = report["best_params"]
        holdout = summary.get("holdout_metrics", {})
        scored = report["scored"]
        selected = scored.loc[scored["selected"]].copy()
        watchlist = scored.head(10).copy()

        summary_cards.append(
            f"""
            <div class="card metric">
              <span class="label">{html.escape(config.profile_title)} Hit Rate</span>
              <span class="value">{_format_pct(holdout.get("monthly_hit_rate"))}</span>
              <div class="subvalue">traded months: {holdout.get("months_traded", 0)}</div>
            </div>
            """
        )
        summary_cards.append(
            f"""
            <div class="card metric">
              <span class="label">{html.escape(config.profile_title)} Avg Return</span>
              <span class="value">{_format_pct(holdout.get("average_monthly_return"))}</span>
              <div class="subvalue">pass 80%: {holdout.get("pass_80pct_requirement", False)}</div>
            </div>
            """
        )

        profile_sections.append(
            f"""
            <div class="card section">
              <h2>{html.escape(config.profile_title)}</h2>
              <p>Latest snapshot: <strong>{report["latest_date"].strftime("%Y-%m-%d")}</strong></p>
              <p>Thresholds: rank { _format_pct(best_params.get("rank_threshold")) }, regime { _format_pct(best_params.get("regime_threshold")) }, max positions {best_params.get("max_positions", 0)}.</p>
              <h3>Actionable</h3>
              {_render_table(
                  selected,
                  ["profile_title", "symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
                  empty_message=f"No actionable recommendation right now in {config.profile_title}.",
              )}
              <h3>Top Watchlist</h3>
              {_render_table(
                  watchlist,
                  ["profile_title", "symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
                  empty_message=f"No watchlist rows were generated for {config.profile_title}.",
              )}
            </div>
            """
        )

    combined_selected_table = _render_table(
        combined_selected,
        ["profile_title", "symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
        empty_message="No actionable recommendation right now across both universes. Stay in cash and monitor the watchlists below.",
    )
    combined_watchlist_table = _render_table(
        combined_watchlist,
        ["profile_title", "symbol", "status", "rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe", "pe", "explanation"],
        empty_message="No watchlist rows were generated across the 60-name view.",
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined 60-Symbol Recommendation Brief</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero, .card {{
      background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
      border: 1px solid rgba(148,163,184,0.18);
      border-radius: 18px;
      box-shadow: 0 14px 35px rgba(2, 6, 23, 0.35);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    p {{
      margin: 6px 0;
      color: #cbd5e1;
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .metric {{
      padding: 18px;
    }}
    .metric .label {{
      color: #94a3b8;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric .value {{
      display: block;
      margin-top: 8px;
      font-size: 30px;
      font-weight: 800;
      color: #f8fafc;
    }}
    .metric .subvalue {{
      margin-top: 8px;
      color: #94a3b8;
      font-size: 13px;
    }}
    .section {{
      margin-top: 22px;
      padding: 22px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      overflow: hidden;
      border-radius: 12px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.16);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: #93c5fd;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .recommend {{
      color: #86efac;
      font-weight: 700;
    }}
    .watch {{
      color: #fcd34d;
      font-weight: 700;
    }}
    .muted {{
      color: #94a3b8;
    }}
    .note {{
      margin-top: 12px;
      font-size: 13px;
      color: #94a3b8;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Combined 60-Symbol Recommendation Brief</h1>
      <p>Latest monthly snapshot across both universes: <strong>{latest_date.strftime("%Y-%m-%d")}</strong></p>
      <p>This page combines the latest information from both split models: <strong>{html.escape(UNIVERSE_TITLES["vn30_core"])}</strong> and <strong>{html.escape(UNIVERSE_TITLES["new30"])}</strong>.</p>
      <p>Data mode: <strong>{'Fresh vnstock refresh' if refreshed_data else 'Cached local data'}</strong></p>
      <p class="note">Scores come from two separately trained models, so profile labels matter. Use this report to see all 60 names in one place, not to assume the probabilities are perfectly cross-calibrated.</p>
    </div>

    <div class="metrics">
      {''.join(summary_cards)}
    </div>

    <div class="card section">
      <h2>Combined Actionable Recommendations</h2>
      <p>These rows are tradable now inside their own universe rules.</p>
      {combined_selected_table}
    </div>

    <div class="card section">
      <h2>Combined 60-Name Watchlist</h2>
      <p>This table shows the latest scored names from both universes together.</p>
      {combined_watchlist_table}
    </div>

    {''.join(profile_sections)}

    <div class="card section">
      <h2>How To Read This</h2>
      <p><strong>Rank probability</strong>: the model's estimate that the stock can reach at least 10% over the next 20 trading days.</p>
      <p><strong>Regime probability</strong>: how favorable the overall month looks for taking risk inside that universe.</p>
      <p><strong>Status</strong>: <span class="recommend">Recommend</span> means tradable now, while <span class="watch">Watchlist</span> means monitor only.</p>
      <p class="note">Because this is a monthly-rotation strategy, the month-end snapshot matters much more than day-to-day noise.</p>
    </div>
  </div>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _render_table(frame: pd.DataFrame, columns: list[str], empty_message: str) -> str:
    if frame.empty:
        return f'<p class="muted">{html.escape(empty_message)}</p>'

    rows = []
    for row in frame[columns].itertuples(index=False):
        cells = []
        for column, value in zip(columns, row):
            if column == "status":
                css_class = "recommend" if value == "Recommend" else "watch"
                rendered = f'<span class="{css_class}">{html.escape(str(value))}</span>'
            elif column in {"rank_probability", "regime_probability", "ret_20d", "ret_60d", "distance_ma200", "roe"}:
                rendered = html.escape(_format_pct(value))
            elif column == "pe":
                rendered = html.escape(_format_number(value))
            else:
                rendered = html.escape("" if pd.isna(value) else str(value))
            cells.append(f"<td>{rendered}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(f"<th>{html.escape(column.replace('_', ' '))}</th>" for column in columns)
    body = "".join(rows)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def _format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def _format_number(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
