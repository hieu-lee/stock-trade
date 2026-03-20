from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.research.validation import _prepare_final_training_frame, _split_development_and_final_test, evaluate_deployability


class ValidationTests(unittest.TestCase):
    def test_evaluate_deployability_requires_both_final_years_and_drawdown(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])
        passing = {
            "yearly_returns": {"2024": 0.35, "2025": 0.31},
            "max_drawdown": -0.12,
        }
        failing = {
            "yearly_returns": {"2024": 0.35, "2025": 0.19},
            "max_drawdown": -0.12,
        }

        self.assertTrue(evaluate_deployability(passing, config))
        self.assertFalse(evaluate_deployability(failing, config))

    def test_prepare_final_training_frame_applies_final_embargo(self) -> None:
        config = replace(
            TradingBotConfig(project_dir=Path(__file__).resolve().parents[1]),
            development_end_year=2024,
            final_test_years=(2025,),
            fold_embargo_days=20,
        )
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-12-10",
                        "2024-12-12",
                        "2024-12-13",
                        "2024-12-31",
                        "2025-01-02",
                    ]
                ),
                "symbol": ["AAA"] * 5,
                "is_trainable": [True] * 5,
            }
        )

        development, final_test = _split_development_and_final_test(panel, config)
        training = _prepare_final_training_frame(development, config)

        self.assertEqual(pd.to_datetime(training["date"]).max(), pd.Timestamp("2024-12-12"))
        self.assertEqual(pd.to_datetime(final_test["date"]).min(), pd.Timestamp("2025-01-02"))


if __name__ == "__main__":
    unittest.main()
