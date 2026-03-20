from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rdtb.config import TradingBotConfig
from rdtb.research.validation_matrix import (
    HoldoutValidationResult,
    ValidationMatrixReport,
    build_repeated_holdout_windows,
    delay_scored_panel,
    render_validation_matrix_markdown,
    summarize_validation_matrix,
)


class ValidationMatrixTests(unittest.TestCase):
    def test_build_repeated_holdout_windows_prefers_recent_two_year_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TradingBotConfig(project_dir=Path(tmp_dir))
            trainable = pd.DataFrame(
                {
                    "date": [pd.Timestamp(f"{year}-06-30") for year in range(2006, 2026)],
                }
            )

            windows = build_repeated_holdout_windows(trainable, config)

            self.assertEqual(
                [(start.year, end.year) for start, end in windows],
                [(2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)],
            )

    def test_build_repeated_holdout_windows_supports_single_year_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TradingBotConfig(
                project_dir=Path(tmp_dir),
                validation_holdout_years=1,
                validation_holdout_step_years=1,
            )
            trainable = pd.DataFrame(
                {
                    "date": [pd.Timestamp(f"{year}-06-30") for year in range(2006, 2027)],
                }
            )

            windows = build_repeated_holdout_windows(trainable, config)

            self.assertEqual(
                [(start.year, end.year) for start, end in windows],
                [(2021, 2021), (2022, 2022), (2023, 2023), (2024, 2024), (2025, 2025)],
            )

    def test_delay_scored_panel_shifts_symbol_scores_and_regime(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "symbol": "AAA",
                    "alpha_prediction": 1.0,
                    "risk_probability": 0.1,
                    "utility_score": 0.2,
                    "regime_anchor_probability": 0.7,
                    "regime_participation_probability": 0.4,
                    "regime_probability": 0.3,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "alpha_prediction": 2.0,
                    "risk_probability": 0.2,
                    "utility_score": 0.3,
                    "regime_anchor_probability": 0.8,
                    "regime_participation_probability": 0.5,
                    "regime_probability": 0.4,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "alpha_prediction": 3.0,
                    "risk_probability": 0.3,
                    "utility_score": 0.4,
                    "regime_anchor_probability": 0.9,
                    "regime_participation_probability": 0.6,
                    "regime_probability": 0.5,
                },
                {
                    "date": "2024-01-01",
                    "symbol": "BBB",
                    "alpha_prediction": 10.0,
                    "risk_probability": 0.4,
                    "utility_score": 0.5,
                    "regime_anchor_probability": 0.7,
                    "regime_participation_probability": 1.1,
                    "regime_probability": 0.8,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "BBB",
                    "alpha_prediction": 20.0,
                    "risk_probability": 0.5,
                    "utility_score": 0.6,
                    "regime_anchor_probability": 0.8,
                    "regime_participation_probability": 0.9,
                    "regime_probability": 0.7,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "BBB",
                    "alpha_prediction": 30.0,
                    "risk_probability": 0.6,
                    "utility_score": 0.7,
                    "regime_anchor_probability": 0.9,
                    "regime_participation_probability": 0.7,
                    "regime_probability": 0.6,
                },
            ]
        )

        delayed = delay_scored_panel(frame, days=1)
        aaa = delayed.loc[delayed["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
        bbb = delayed.loc[delayed["symbol"] == "BBB"].sort_values("date").reset_index(drop=True)

        self.assertTrue(pd.isna(aaa.loc[0, "alpha_prediction"]))
        self.assertEqual(float(aaa.loc[1, "alpha_prediction"]), 1.0)
        self.assertEqual(float(aaa.loc[2, "alpha_prediction"]), 2.0)
        self.assertTrue(pd.isna(bbb.loc[0, "utility_score"]))
        self.assertEqual(float(bbb.loc[1, "utility_score"]), 0.5)
        self.assertTrue(pd.isna(aaa.loc[0, "regime_probability"]))
        self.assertEqual(float(aaa.loc[1, "regime_probability"]), 0.3)
        self.assertEqual(float(aaa.loc[2, "regime_probability"]), 0.4)
        self.assertTrue(pd.isna(aaa.loc[0, "regime_anchor_probability"]))
        self.assertEqual(float(aaa.loc[1, "regime_anchor_probability"]), 0.7)
        self.assertEqual(float(aaa.loc[2, "regime_anchor_probability"]), 0.8)
        self.assertTrue(pd.isna(aaa.loc[0, "regime_participation_probability"]))
        self.assertEqual(float(aaa.loc[1, "regime_participation_probability"]), 0.4)
        self.assertEqual(float(aaa.loc[2, "regime_participation_probability"]), 0.5)
        self.assertTrue(pd.isna(bbb.loc[0, "regime_probability"]))
        self.assertEqual(float(bbb.loc[1, "regime_probability"]), 0.8)
        self.assertEqual(float(bbb.loc[2, "regime_probability"]), 0.7)

    def test_summarize_validation_matrix_counts_baseline_wins_and_pass_criteria(self) -> None:
        holdouts = [
            HoldoutValidationResult(
                label="2020-2021",
                train_start="2006-01-01",
                train_end="2019-12-12",
                test_start="2020-01-01",
                test_end="2021-12-31",
                strategy_metrics={"annualized_return": 0.25, "sharpe": 1.4, "max_drawdown": -0.10},
                benchmark_metrics={"annualized_return": 0.10, "sharpe": 0.8, "max_drawdown": -0.18},
                equal_weight_metrics={"annualized_return": 0.12, "sharpe": 0.9, "max_drawdown": -0.15},
                momentum_metrics={"annualized_return": 0.20, "sharpe": 1.1, "max_drawdown": -0.12},
                stress_metrics={
                    "high_friction": {"annualized_return": 0.11},
                    "signal_delay_1d": {"annualized_return": 0.09},
                },
            ),
            HoldoutValidationResult(
                label="2022-2023",
                train_start="2006-01-01",
                train_end="2021-12-12",
                test_start="2022-01-01",
                test_end="2023-12-31",
                strategy_metrics={"annualized_return": 0.18, "sharpe": 1.1, "max_drawdown": -0.14},
                benchmark_metrics={"annualized_return": 0.05, "sharpe": 0.5, "max_drawdown": -0.20},
                equal_weight_metrics={"annualized_return": 0.07, "sharpe": 0.6, "max_drawdown": -0.17},
                momentum_metrics={"annualized_return": 0.09, "sharpe": 0.7, "max_drawdown": -0.16},
                stress_metrics={
                    "high_friction": {"annualized_return": 0.04},
                    "signal_delay_1d": {"annualized_return": 0.02},
                },
            ),
        ]

        summary = summarize_validation_matrix(holdouts, seed_stability=None)

        self.assertEqual(summary["holdout_count"], 2)
        self.assertEqual(summary["beat_benchmark_count"], 2)
        self.assertEqual(summary["beat_equal_weight_count"], 2)
        self.assertEqual(summary["beat_momentum_count"], 2)
        self.assertAlmostEqual(float(summary["median_holdout_annualized_return"]), 0.215)
        self.assertAlmostEqual(float(summary["worst_holdout_year_return"]), 0.18)
        self.assertTrue(bool(summary["pass_criteria"]["median_holdout_annualized_return_gt_min_year_return"]))
        self.assertTrue(bool(summary["pass_criteria"]["median_holdout_worst_year_return_gt_min_year_return"]))
        self.assertTrue(bool(summary["pass_criteria"]["beats_vnindex_in_majority"]))

    def test_render_validation_matrix_markdown_contains_key_sections(self) -> None:
        report = ValidationMatrixReport(
            generated_at="2026-03-16T00:00:00",
            policy_source="deployment_manifest",
            holdouts=[
                HoldoutValidationResult(
                    label="2024-2025",
                    train_start="2006-01-01",
                    train_end="2023-12-12",
                    test_start="2024-01-01",
                    test_end="2025-12-31",
                    strategy_metrics={"annualized_return": 0.20, "sharpe": 1.2, "max_drawdown": -0.12},
                    benchmark_metrics={"annualized_return": 0.09},
                    equal_weight_metrics={"annualized_return": 0.08},
                    momentum_metrics={"annualized_return": 0.12},
                    stress_metrics={
                        "high_friction": {"annualized_return": 0.05},
                        "signal_delay_1d": {"annualized_return": 0.04},
                    },
                )
            ],
            seed_stability=None,
            summary={
                "overall_pass": False,
                "median_holdout_annualized_return": 0.20,
                "median_holdout_sharpe": 1.2,
                "worst_holdout_max_drawdown": -0.12,
                "beat_benchmark_count": 1,
                "beat_equal_weight_count": 1,
                "holdout_count": 1,
                "median_high_friction_annualized_return": 0.05,
            },
        )

        markdown = render_validation_matrix_markdown(report)

        self.assertIn("# Validation Matrix", markdown)
        self.assertIn("## Holdouts", markdown)
        self.assertIn("2024-2025", markdown)
        self.assertIn("deployment_manifest", markdown)


if __name__ == "__main__":
    unittest.main()
