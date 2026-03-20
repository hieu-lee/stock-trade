from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from rdtb.config import TradingBotConfig
from rdtb.models.train import (
    ConstantProbabilityClassifier,
    ModelBundle,
    ModelStack,
    RegimeModelStack,
    build_regime_symbol_frame,
    get_regime_features,
    score_panel,
)


class ModelTrainingTests(unittest.TestCase):
    def test_build_regime_symbol_frame_preserves_per_symbol_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "target_regime": 1.0,
                    "benchmark_ret_20d": 0.04,
                    "foreign_flow_score": 0.20,
                    "order_pressure_score": 0.10,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "BBB",
                    "target_regime": 1.0,
                    "benchmark_ret_20d": 0.04,
                    "foreign_flow_score": 0.60,
                    "order_pressure_score": 0.50,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "AAA",
                    "target_regime": 0.0,
                    "benchmark_ret_20d": -0.02,
                    "foreign_flow_score": -0.10,
                    "order_pressure_score": 0.20,
                },
                {
                    "date": "2024-01-03",
                    "symbol": "BBB",
                    "target_regime": 0.0,
                    "benchmark_ret_20d": -0.02,
                    "foreign_flow_score": 0.10,
                    "order_pressure_score": 0.00,
                },
            ]
        )

        regime_frame = build_regime_symbol_frame(
            frame,
            ["benchmark_ret_20d", "foreign_flow_score", "order_pressure_score"],
            include_target=True,
        )

        self.assertEqual(len(regime_frame), 4)
        self.assertEqual(regime_frame["symbol"].tolist(), ["AAA", "BBB", "AAA", "BBB"])
        self.assertAlmostEqual(float(regime_frame.loc[0, "benchmark_ret_20d"]), 0.04)
        self.assertAlmostEqual(float(regime_frame.loc[1, "foreign_flow_score"]), 0.60)
        self.assertAlmostEqual(float(regime_frame.loc[3, "order_pressure_score"]), 0.00)

    def test_get_regime_features_can_drop_symbol_specific_regime_features(self) -> None:
        config = TradingBotConfig(
            project_dir=Path(__file__).resolve().parents[1],
            regime_use_market_only_features=True,
        )

        features = get_regime_features(config)

        self.assertIn("benchmark_ret_20d", features)
        self.assertNotIn("relative_strength_10d", features)
        self.assertNotIn("foreign_flow_score", features)

    def test_score_panel_assigns_symbol_specific_regime_probabilities(self) -> None:
        alpha_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("estimator", DummyRegressor(strategy="constant", constant=0.5)),
            ]
        )
        alpha_pipeline.fit(pd.DataFrame({"ret_1d": [0.0, 1.0]}), [0.5, 0.5])
        risk_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("estimator", DummyClassifier(strategy="constant", constant=0)),
            ]
        )
        risk_pipeline.fit(pd.DataFrame({"ret_1d": [0.0, 1.0]}), [0, 1])
        anchor_feature_columns = ["benchmark_ret_20d"]
        anchor_pipeline = Pipeline([("estimator", ConstantProbabilityClassifier(probability=0.8))])
        anchor_pipeline.fit(pd.DataFrame({"benchmark_ret_20d": [0.02]}), [1])
        participation_feature_columns = [
            "relative_strength_10d",
            "relative_strength_20d",
            "beta_60d",
        ]
        participation_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("estimator", LogisticRegression(random_state=0, max_iter=1000)),
            ]
        )
        participation_training = pd.DataFrame(
            {
                "relative_strength_10d": [-0.40, 0.45, -0.25, 0.55],
                "relative_strength_20d": [-0.35, 0.40, -0.15, 0.50],
                "beta_60d": [0.75, 1.20, 0.80, 1.30],
            }
        )
        participation_pipeline.fit(participation_training, [0, 1, 0, 1])
        model_stack = ModelStack(
            alpha=ModelBundle(alpha_pipeline, ["ret_1d"], "target_alpha_blend"),
            risk=ModelBundle(risk_pipeline, ["ret_1d"], "target_downside"),
            regime=RegimeModelStack(
                anchor=ModelBundle(anchor_pipeline, anchor_feature_columns, "target_regime"),
                participation=ModelBundle(
                    participation_pipeline,
                    participation_feature_columns,
                    "target_regime_participation",
                ),
            ),
        )
        panel = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "symbol": "AAA",
                    "ret_1d": 0.01,
                    "benchmark_ret_20d": 0.03,
                    "relative_strength_10d": -0.20,
                    "relative_strength_20d": -0.10,
                    "relative_strength_20d_rank": 0.25,
                    "beta_60d": 0.85,
                },
                {
                    "date": "2024-01-02",
                    "symbol": "BBB",
                    "ret_1d": 0.01,
                    "benchmark_ret_20d": 0.03,
                    "relative_strength_10d": 0.35,
                    "relative_strength_20d": 0.30,
                    "relative_strength_20d_rank": 0.75,
                    "beta_60d": 1.25,
                },
            ]
        )

        scored = score_panel(panel, model_stack)
        aaa_anchor = float(scored.loc[scored["symbol"] == "AAA", "regime_anchor_probability"].iloc[0])
        bbb_anchor = float(scored.loc[scored["symbol"] == "BBB", "regime_anchor_probability"].iloc[0])
        aaa_participation = float(scored.loc[scored["symbol"] == "AAA", "regime_participation_probability"].iloc[0])
        bbb_participation = float(scored.loc[scored["symbol"] == "BBB", "regime_participation_probability"].iloc[0])
        aaa_probability = float(scored.loc[scored["symbol"] == "AAA", "regime_probability"].iloc[0])
        bbb_probability = float(scored.loc[scored["symbol"] == "BBB", "regime_probability"].iloc[0])

        self.assertAlmostEqual(aaa_anchor, 0.8)
        self.assertAlmostEqual(bbb_anchor, 0.8)
        self.assertLess(aaa_participation, bbb_participation)
        self.assertLess(aaa_probability, bbb_probability)
        self.assertAlmostEqual(aaa_probability, aaa_anchor * aaa_participation)
        self.assertAlmostEqual(bbb_probability, bbb_anchor * bbb_participation)
        self.assertNotAlmostEqual(aaa_probability, bbb_probability)


if __name__ == "__main__":
    unittest.main()
