from __future__ import annotations

import unittest
from pathlib import Path

from rdtb.config import TradingBotConfig
from rdtb.research.validation import evaluate_deployability


class ValidationTests(unittest.TestCase):
    def test_evaluate_deployability_requires_both_final_years_and_drawdown(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])
        passing = {
            "yearly_returns": {"2024": 0.35, "2025": 0.31},
            "max_drawdown": -0.12,
        }
        failing = {
            "yearly_returns": {"2024": 0.35, "2025": 0.22},
            "max_drawdown": -0.12,
        }

        self.assertTrue(evaluate_deployability(passing, config))
        self.assertFalse(evaluate_deployability(failing, config))


if __name__ == "__main__":
    unittest.main()
