from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from rdtb.config import TradingBotConfig
from rdtb.portfolio.optimizer import default_policy
from rdtb.service.pipeline import _config_from_manifest, _policy_from_manifest, _utility_weights_from_manifest


class PipelineManifestTests(unittest.TestCase):
    def test_manifest_restores_config_policy_and_utility_weights(self) -> None:
        config = TradingBotConfig(project_dir=Path(__file__).resolve().parents[1])
        policy_payload = asdict(default_policy(config))
        policy_payload["max_positions"] = 5
        policy_payload["regime_transition_slope"] = 11.0
        manifest = {
            "best_config_overrides": {
                "use_company_metadata_features": True,
                "regime_target_return_threshold": 0.025,
            },
            "best_policy": policy_payload,
            "best_utility_weights": {
                "alpha_rank": 2.0,
                "risk_adjusted_rank": 0.0,
                "relative_strength_20d_rank": 0.0,
                "downside_score": 0.0,
                "regime_probability": 0.0,
            },
        }

        effective_config = _config_from_manifest(config, manifest)
        policy = _policy_from_manifest(effective_config, manifest)
        weights = _utility_weights_from_manifest(manifest)

        self.assertTrue(effective_config.use_company_metadata_features)
        self.assertAlmostEqual(float(effective_config.regime_target_return_threshold), 0.025)
        self.assertEqual(int(policy.max_positions), 5)
        self.assertAlmostEqual(float(policy.regime_transition_slope), 11.0)
        assert weights is not None
        self.assertAlmostEqual(float(weights["alpha_rank"]), 1.0)
        self.assertAlmostEqual(float(sum(weights.values())), 1.0)


if __name__ == "__main__":
    unittest.main()
