# Really Dope Trading Bot

`really-dope-trading-bot` is a Python-only VN60 daily trading system designed around a clean separation of concerns:

- point-in-time daily data collection and caching
- as-of feature engineering on a 60-symbol universe
- alpha, downside-risk, and regime model layers
- target-weight portfolio optimization with deterministic action compilation
- event-driven backtesting with strict yearly acceptance gates
- FastAPI and Streamlit interfaces for one-click daily decisions

## Layout

- `src/rdtb/config.py`: core configuration and paths
- `src/rdtb/data/`: market data adapters, caching, and DuckDB helpers
- `src/rdtb/features/`: daily as-of panel construction and training targets
- `src/rdtb/models/`: model fitting, scoring, and artifact persistence
- `src/rdtb/portfolio/`: target weights and net actions
- `src/rdtb/backtest/`: next-open execution simulator
- `src/rdtb/research/`: walk-forward validation and deployability gates
- `src/rdtb/service/`: orchestration pipeline used by the CLI, API, and UI
- `src/rdtb/api/`: FastAPI endpoints
- `src/rdtb/ui/`: Streamlit dashboard

## Quick Start

```bash
cd really-dope-trading-bot
python -m pip install -e .
rdtb-train
streamlit run src/rdtb/ui/app.py
```

## Deployment Contract

The system treats the following as strict acceptance gates for deployable artifacts:

- every untouched final-test year must deliver at least `30%` return
- untouched final-test max drawdown must remain at or above `-15%`

If the research pipeline cannot satisfy those gates honestly, the system marks the artifact as non-deployable instead of silently relaxing the constraints.
