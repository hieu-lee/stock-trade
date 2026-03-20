from __future__ import annotations

import unittest

import pandas as pd

from rdtb.models.train import DEFAULT_UTILITY_WEIGHTS, apply_utility_weights, normalize_utility_weights
from rdtb.research.constant_search import score_search_summary, search_summary_beats_target


class ConstantSearchTests(unittest.TestCase):
    def test_normalize_utility_weights_falls_back_to_defaults(self) -> None:
        weights = normalize_utility_weights({"alpha_rank": 0.0, "relative_strength_20d_rank": 0.0})
        self.assertEqual(weights, DEFAULT_UTILITY_WEIGHTS)

    def test_apply_utility_weights_resorts_symbols(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "symbol": "AAA",
                    "alpha_rank": 0.9,
                    "risk_adjusted_rank": 0.8,
                    "relative_strength_20d_rank": 0.3,
                    "downside_score": 0.7,
                    "regime_probability": 0.4,
                },
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "symbol": "BBB",
                    "alpha_rank": 0.4,
                    "risk_adjusted_rank": 0.9,
                    "relative_strength_20d_rank": 0.8,
                    "downside_score": 0.6,
                    "regime_probability": 0.4,
                },
            ]
        )

        rescored = apply_utility_weights(frame, {"alpha_rank": 1.0})

        self.assertEqual(str(rescored.iloc[0]["symbol"]), "AAA")
        self.assertGreater(float(rescored.iloc[0]["utility_score"]), float(rescored.iloc[1]["utility_score"]))

    def test_search_summary_scoring_prefers_target_beating_summary(self) -> None:
        target = {
            "median_holdout_annualized_return": 0.20,
            "median_holdout_sharpe": 1.2,
            "worst_holdout_max_drawdown": -0.30,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 3,
            "median_high_friction_annualized_return": 0.03,
        }
        weaker = {
            "median_holdout_annualized_return": 0.12,
            "median_holdout_sharpe": 0.9,
            "worst_holdout_max_drawdown": -0.35,
            "beat_benchmark_count": 2,
            "beat_equal_weight_count": 2,
            "beat_momentum_count": 1,
            "median_momentum_gap": -0.20,
            "median_high_friction_annualized_return": 0.00,
            "median_signal_delay_annualized_return": 0.05,
        }
        stronger = {
            "median_holdout_annualized_return": 0.24,
            "median_holdout_sharpe": 1.4,
            "worst_holdout_max_drawdown": -0.25,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 4,
            "beat_momentum_count": 2,
            "median_momentum_gap": -0.04,
            "median_high_friction_annualized_return": 0.05,
            "median_signal_delay_annualized_return": 0.08,
        }

        self.assertGreater(score_search_summary(stronger, target), score_search_summary(weaker, target))
        self.assertTrue(search_summary_beats_target(stronger, target))
        self.assertFalse(search_summary_beats_target(weaker, target))

    def test_search_summary_scoring_penalizes_contract_breaches(self) -> None:
        contract_friendly = {
            "median_holdout_annualized_return": 0.24,
            "median_holdout_sharpe": 1.1,
            "median_holdout_worst_year_return": 0.22,
            "worst_holdout_year_return": 0.20,
            "worst_holdout_max_drawdown": -0.12,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 2,
            "beat_momentum_count": 1,
            "median_momentum_gap": -0.08,
            "median_high_friction_annualized_return": 0.04,
            "median_signal_delay_annualized_return": 0.06,
        }
        flashy_but_fragile = {
            "median_holdout_annualized_return": 0.30,
            "median_holdout_sharpe": 1.3,
            "median_holdout_worst_year_return": 0.08,
            "worst_holdout_year_return": -0.04,
            "worst_holdout_max_drawdown": -0.24,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 3,
            "beat_momentum_count": 2,
            "median_momentum_gap": -0.02,
            "median_high_friction_annualized_return": 0.05,
            "median_signal_delay_annualized_return": 0.07,
        }

        self.assertGreater(score_search_summary(contract_friendly), score_search_summary(flashy_but_fragile))


if __name__ == "__main__":
    unittest.main()
