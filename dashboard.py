from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from vn30_strategy.config import UNIVERSE_TITLES, UNIVERSE_SYMBOLS, get_default_config
from vn30_strategy.utils import read_json


st.set_page_config(page_title="VN30 Monthly Alpha", layout="wide")
profile = st.sidebar.selectbox(
    "Universe profile",
    options=list(UNIVERSE_SYMBOLS.keys()),
    format_func=lambda key: UNIVERSE_TITLES[key],
)
config = get_default_config(Path(__file__).resolve().parent, profile_name=profile)

st.title(f"{config.profile_title} Dashboard")
st.caption("A cash long-only monthly rotation strategy with separate universes.")

summary_path = config.artifacts_dir / "summary.json"
recommendations_path = config.recommendations_dir / "current_recommendations.csv"

if not summary_path.exists():
    st.warning(f"No trained strategy artifacts found yet for `{config.profile_title}`. Run `python main.py --profile {profile} train` first.")
    st.stop()

summary = read_json(summary_path)
holdout = summary["holdout_metrics"]
in_sample = summary["in_sample_metrics"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout Hit Rate", f"{holdout['monthly_hit_rate']:.2%}")
col2.metric("Passes 80% Target", "Yes" if holdout["pass_80pct_requirement"] else "No")
col3.metric("Holdout CAGR-like", f"{holdout['cagr_like']:.2%}")
col4.metric("Max Drawdown", f"{holdout['max_drawdown']:.2%}")

st.subheader("Strategy Tactic")
st.markdown(
    """
    - The regime gate decides whether the month is favorable enough to trade.
    - The ranking model scores each VN30 stock on its probability of reaching at least 10% return over the next 20 trading days.
    - Only the highest-conviction symbols above the threshold are selected; otherwise the strategy stays partly or fully in cash.
    """
)

st.subheader("Current Recommendations")
if recommendations_path.exists():
    recommendations = pd.read_csv(recommendations_path)
    st.dataframe(recommendations, use_container_width=True)
else:
    st.info("No current recommendations have been generated yet.")

st.subheader("Backtest Summary")
st.json({"in_sample": in_sample, "holdout": holdout})

st.subheader("Charts")
for label, image_path in summary["report_images"].items():
    file_path = Path(image_path)
    if file_path.exists():
        st.image(str(file_path), caption=label.replace("_", " ").title())
