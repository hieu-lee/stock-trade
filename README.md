# VN30 Strategy

`vn30_strategy` is a standalone VN30 monthly-rotation research and trading application built from scratch on top of `vnstock`.

## What it does

- Downloads VN30 cash-equity data, VNIndex/VN30 benchmark data, company overviews, financial ratios, and best-effort dividend data.
- Builds adjusted research panels and technical, fundamental, liquidity, and regime features.
- Trains a two-stage monthly strategy:
  - a regime gate that decides whether a month is worth trading;
  - a cross-sectional ranking model that estimates the probability of reaching `>= 10%` return over the next 20 trading days.
- Runs walk-forward backtests and reports whether the out-of-sample monthly hit rate reaches the requested `80%` threshold.
- Ships with both a CLI and a Streamlit dashboard.

## Universe profiles

The project now supports separate universe profiles:

- `vn30_core`: the original 30 VN30 symbols
- `new30`: the 30 newly added symbols
- `all60`: both groups combined for research

Each profile writes to its own `data/<profile>/...` and `artifacts/<profile>/...` directories so the models do not overwrite each other.

## Quick start

```bash
./.venv/bin/python -m pip install -e .
./.venv/bin/python main.py --profile vn30_core train
./.venv/bin/python main.py --profile vn30_core recommend
./.venv/bin/streamlit run dashboard.py
```

Convenience scripts:

```bash
./.venv/bin/python run_vn30_core.py
./.venv/bin/python run_new30.py
./.venv/bin/python run_vn30_core.py --retrain
./.venv/bin/python run_new30.py --retrain
```

## Notes

- The system treats the 80% target as a hard evaluation requirement and reports pass/fail honestly.
- Historical coverage depends on each symbol's listing/provider availability.
- `vnstock` does not expose a reliable VCI dividend endpoint in this workspace, so the pipeline attempts dividend retrieval and falls back gracefully when unavailable.
