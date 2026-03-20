from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.research.auto_search import _combined_candidate_score, _window_matches_final_test


class AutoSearchTests(unittest.TestCase):
    def test_combined_candidate_score_prefers_stronger_final_years(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])
        shared_summary = {
            "median_holdout_annualized_return": 0.21,
            "median_holdout_sharpe": 1.1,
            "median_holdout_worst_year_return": 0.20,
            "worst_holdout_year_return": 0.18,
            "worst_holdout_max_drawdown": -0.13,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 3,
            "beat_momentum_count": 2,
            "median_momentum_gap": -0.06,
            "median_high_friction_annualized_return": 0.03,
            "median_signal_delay_annualized_return": 0.05,
        }
        strong_final_metrics = {
            "annualized_return": 0.26,
            "yearly_returns": {"2024": 0.28, "2025": 0.24},
            "max_drawdown": -0.11,
        }
        weak_final_metrics = {
            "annualized_return": 0.20,
            "yearly_returns": {"2024": 0.34, "2025": 0.07},
            "max_drawdown": -0.11,
        }

        strong_score = _combined_candidate_score(
            search_summary=shared_summary,
            final_summary=shared_summary,
            final_metrics=strong_final_metrics,
            config=config,
            target_summary=None,
        )
        weak_score = _combined_candidate_score(
            search_summary=shared_summary,
            final_summary=shared_summary,
            final_metrics=weak_final_metrics,
            config=config,
            target_summary=None,
        )

        self.assertGreater(strong_score, weak_score)

    def test_combined_candidate_score_prioritizes_higher_final_annualized_return(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])
        shared_summary = {
            "median_holdout_annualized_return": 0.20,
            "median_holdout_sharpe": 1.0,
            "median_holdout_worst_year_return": 0.19,
            "worst_holdout_year_return": 0.18,
            "worst_holdout_max_drawdown": -0.12,
            "beat_benchmark_count": 3,
            "beat_equal_weight_count": 2,
            "beat_momentum_count": 1,
            "median_momentum_gap": -0.08,
            "median_high_friction_annualized_return": 0.04,
            "median_signal_delay_annualized_return": 0.05,
        }
        lower_return_metrics = {
            "annualized_return": 0.22,
            "yearly_returns": {"2024": 0.22, "2025": 0.22},
            "max_drawdown": -0.11,
        }
        higher_return_metrics = {
            "annualized_return": 0.30,
            "yearly_returns": {"2024": 0.29, "2025": 0.31},
            "max_drawdown": -0.11,
        }

        lower_score = _combined_candidate_score(
            search_summary=shared_summary,
            final_summary=shared_summary,
            final_metrics=lower_return_metrics,
            config=config,
            target_summary=None,
        )
        higher_score = _combined_candidate_score(
            search_summary=shared_summary,
            final_summary=shared_summary,
            final_metrics=higher_return_metrics,
            config=config,
            target_summary=None,
        )

        self.assertGreater(higher_score, lower_score)

    def test_window_matches_final_test(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])

        self.assertTrue(_window_matches_final_test((pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")), config))
        self.assertFalse(_window_matches_final_test((pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")), config))

    def test_window_matches_single_validation_year(self) -> None:
        config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            development_end_year=2024,
            final_test_years=(2025,),
            validation_holdout_years=1,
            validation_holdout_step_years=1,
        )

        self.assertTrue(_window_matches_final_test((pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")), config))
        self.assertFalse(_window_matches_final_test((pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")), config))


if __name__ == "__main__":
    unittest.main()
