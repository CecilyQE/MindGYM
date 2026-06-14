"""Generative envs expose transcript/paper grounding metadata."""

import unittest

from psycenvir import make_generative_env
from psycenvir.core.generative_registry import list_generative_experiments
from psycenvir.generative.grounding import (
    GenerativeGroundingKind,
    get_grounding_profile,
)
from psycenvir.generative.kool_cost_exp2 import KOOL_COST_EXP2_ID


class GenerativeGroundingTests(unittest.TestCase):
    def test_all_generative_ids_have_profiles(self) -> None:
        for experiment_id in list_generative_experiments():
            profile = get_grounding_profile(experiment_id)
            self.assertIsInstance(profile.kind, GenerativeGroundingKind)
            self.assertTrue(profile.sources)

    def test_make_generative_env_annotates_info(self) -> None:
        env = make_generative_env("badham2017deficits/exp1.csv", seed=0, trials_per_problem=2)
        _, info = env.reset()
        self.assertIn("generative_grounding", info)
        self.assertIn("generative_sources", info)
        self.assertIn(info["generative_grounding"], {k.value for k in GenerativeGroundingKind})

    def test_kool_cost_exp2_uses_transcript_topologies(self) -> None:
        profile = get_grounding_profile(KOOL_COST_EXP2_ID)
        topologies = profile.default_config.get("session_topologies", [])
        self.assertGreater(len(topologies), 0)
        sample = topologies[0]
        self.assertIn("ships", sample)
        self.assertIn("aliens_by_planet", sample)
        self.assertIn("planet_by_ship", sample)

        env = make_generative_env(KOOL_COST_EXP2_ID, seed=0, n_days=1)
        obs, info = env.reset(seed=1)
        self.assertEqual(info["generative_grounding"], "transcript_calibrated")
        self.assertIn("pressing", obs)


if __name__ == "__main__":
    unittest.main()
