# Stock Signal Prototype

This prototype uses the cloned `vnstock` repository plus `scikit-learn` to predict whether a stock can reach a peak at least 6% above the current close within the next 20 trading sessions.

## Environment

The virtual environment is in `.venv/`.

Install step already completed:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ./vnstock scikit-learn
```

## Run

From the workspace root:

```bash
.venv/bin/python stock_signal_project/run_experiment.py
```

The script inserts the cloned `vnstock` repo path explicitly before importing so it still works even though the workspace root also contains a folder named `vnstock`.

For the current configured run, the universe is the requested 30-symbol subset with a 15-year history window, and the selected final policy is a precision-first top-3 ranking.

## Production Symbol Signal

Use the per-symbol decision tool like this:

```bash
.venv/bin/python stock_signal_project/symbol_signal.py FPT
```

Behavior:

- If a saved model for the symbol already exists, the tool refreshes only a recent history window and reuses the saved model.
- If no saved model exists, the tool fetches up to 15 years of daily history, trains a per-symbol model, saves it, and then returns a decision.
- Output is `YES` or `NO`, where `YES` means buy at the next session open using the latest close as the decision reference for the 20-trading-day / 6% target.
- The production decision is stricter than the raw model threshold: it also requires minimum model-quality checks and a higher effective probability threshold before returning `YES`.

Useful flags:

```bash
.venv/bin/python stock_signal_project/symbol_signal.py FPT --json
.venv/bin/python stock_signal_project/symbol_signal.py FPT --force-retrain
```

## Outputs

Generated files live in `stock_signal_project/outputs/`:

- `experiment_report.md`
- `model_summary.csv`
- `decision_tree_*.csv`
- `random_forest_*.csv`
- `labeled_dataset.csv`
- `raw_history.csv`
- `latest_feature_rows.csv`
- `metadata.json`

Per-symbol production artifacts are stored in `stock_signal_project/artifacts/`.
