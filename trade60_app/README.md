# Trade60 App

`trade60_app` is a separate long-only daily trading system built next to the existing monthly VN30 research stack.

## What it does

- Uses the existing 60-symbol universe from the repo.
- Reuses the existing `vnstock` rate-limit-aware refresh logic.
- Builds a daily feature panel from 2006 to now.
- Trains a risk-aware next-session strategy with:
  - initial cash `100000`
  - long-only execution
  - no margin
  - no same-day buy/sell on one symbol
  - forced exit after `40` trading days
- Supports mixed-symbol next-session plans such as partial sells on one name and buys on other names in the same daily batch.
- Allows adding to an already-held symbol when its live score still supports a full-size allocation.
- Optimizes toward an annual return target of at least `8%`, while aiming to beat VNINDEX and keep max drawdown above `-10%`.
- Provides a Streamlit UI for entering current holdings and generating tomorrow's actions.

## Quick start

```bash
./.venv/bin/python trade60_main.py train --refresh
./.venv/bin/python trade60_main.py recommend --budget 100000 --portfolio my_holdings.csv
./.venv/bin/streamlit run trade60_dashboard.py
```

## Holdings CSV format

```csv
symbol,quantity,avg_cost,buy_date
FPT,100,121.5,2026-01-15
SSI,200,34.2,2026-02-10
```

`buy_date` is optional for the UI, but it should be filled in if you want the live 2-month max-hold rule to be checked exactly.
If `buy_date` is left blank, the system now defaults it to about one month ago and reports that assumption in the recommendation notes.
