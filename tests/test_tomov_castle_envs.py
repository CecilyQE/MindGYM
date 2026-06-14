import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import TOMOV_CASTLE_EXPERIMENT_IDS, make_task_env
from psycenvir.psych101.parse import parse_tomov_castle_trials

CASTLE_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == TOMOV_CASTLE_EXPERIMENT_IDS[0]
)


class TomovCastleEnvTests(unittest.TestCase):
    def test_generative_two_step_round(self):
        env = make_generative_env(TOMOV_CASTLE_EXPERIMENT_IDS[0], seed=0, n_rounds=1)
        env.reset()
        _, reward0, _, _, _ = env.step("P")
        self.assertEqual(reward0, 0.0)
        _, reward1, terminated, truncated, info = env.step("G")
        self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertIsInstance(reward1, float)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_recorded_counterfactual(self):
        trials = parse_tomov_castle_trials(CASTLE_TEXT)
        trial = next(t for t in trials if t.room_number > 0)
        alt = next(action for action in trial.valid_actions if action != trial.human_action)
        env = make_task_env(TOMOV_CASTLE_EXPERIMENT_IDS[0], CASTLE_TEXT)
        env.reset()
        for prior in trials:
            if prior is trial:
                break
            env.step(prior.human_action)
        _, reward, _, _, _ = env.step(alt)
        self.assertEqual(reward, trial.outcomes_by_action[alt])


if __name__ == "__main__":
    unittest.main()
