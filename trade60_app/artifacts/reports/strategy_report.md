# Trade60 Strategy Report

## Objective
- Long-only daily allocation system across 60 symbols.
- Starting capital: 100,000
- Max holding period: 40 trading days
- Constraints: no margin, no same-day round-trip on one symbol, next-session execution only.

## Validation Metrics
- Annualized return: 16.52%
- Total return: 35.76%
- Benchmark return: -20.13%
- Max drawdown: -12.96%
- Win rate: 56.81%

## Clean Untouched Final-Test Metrics
- Annualized return: 36.05%
- Total return: 36.05%
- Benchmark return: 40.89%
- Excess return vs benchmark: -4.84%
- Beats 8% annual hurdle: True
- Beats benchmark: False
- Max drawdown: -6.56%

## Research Holdout Metrics Before Deployment Calibration
- Annualized return: 6.04%
- Total return: 12.45%
- Benchmark return: 52.38%
- Excess return vs benchmark: -39.92%

## Legacy Deployment-Calibrated Holdout Metrics
- Annualized return: 8.86%
- Total return: 18.52%
- Benchmark return: 52.38%
- Excess return vs benchmark: -33.86%

## Latest Baseline Next-Day Snapshot (No Portfolio)
| action   | symbol   |   quantity |   reference_price |   alpha_probability |   regime_probability | rationale                                                           |
|:---------|:---------|-----------:|------------------:|--------------------:|---------------------:|:--------------------------------------------------------------------|
| BUY      | PVS      |        428 |              42.7 |            0.617243 |             0.523407 | trading above its 50-day trend                                      |
| BUY      | STB      |        281 |              65   |            0.626335 |             0.523407 | beating VNINDEX over the last month, trading above its 50-day trend |
| BUY      | TNG      |        781 |              23.4 |            0.619371 |             0.523407 | beating VNINDEX over the last month, trading above its 50-day trend |

## Notes
- Latest signal date: 2026-03-11 00:00:00
- Additional notes: This snapshot assumes no open holdings. Defensive exposure mode is active, so the system will size fewer positions and keep extra cash.
- Top feature drivers:
| feature                  |   importance |
|:-------------------------|-------------:|
| breadth_turnover_20d     |    0.0849649 |
| breadth_above_ma200      |    0.0821348 |
| benchmark_drawdown_252d  |    0.0786123 |
| benchmark_distance_ma200 |    0.0749165 |
| benchmark_vol_20d        |    0.0711577 |
| distance_ma200           |    0.045801  |
| breadth_above_ma50       |    0.0452214 |
| volatility_20d           |    0.0406445 |
| breadth_ret_20d          |    0.0326221 |
| breadth_positive_10d     |    0.0300134 |
