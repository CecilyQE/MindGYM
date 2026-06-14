"""Smoke tests: every tier-A experiment has a registered generative env."""

import unittest
from pathlib import Path

import yaml

from psycenvir.core.generative_registry import list_generative_experiments, make_generative_env

_TIERS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "generated" / "generative_setting_tiers.yaml"
)


class GenerativeSettingTierATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with _TIERS_PATH.open(encoding="utf-8") as handle:
            tiers = yaml.safe_load(handle)
        cls.tier_a = tuple(tiers["tier_A"])
        cls.tier_c = tuple(tiers["tier_C"])
        cls.registered = set(list_generative_experiments())

    def test_tier_a_subset_of_registered_generative(self) -> None:
        missing = sorted(set(self.tier_a) - self.registered)
        self.assertEqual(missing, [], "tier-A experiments missing generative: {}".format(missing))

    def test_tier_c_not_registered(self) -> None:
        overlap = sorted(set(self.tier_c) & self.registered)
        self.assertEqual(overlap, [], "tier-C should not have generative: {}".format(overlap))

    def test_each_tier_a_resets(self) -> None:
        for experiment_id in self.tier_a:
            env = make_generative_env(experiment_id, seed=0)
            observation, info = env.reset(seed=0)
            self.assertIsInstance(observation, str)
            self.assertIn("generative_grounding", info)


if __name__ == "__main__":
    unittest.main()
