import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import SCHULZ_FINDING_EXP1_ID, SCHULZ_FINDING_EXPERIMENT_IDS, make_task_env
from psycenvir.psych101.parse import parse_schulz_finding_trials

SCHULZ_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == SCHULZ_FINDING_EXP1_ID
)


class SchulzFindingEnvTests(unittest.TestCase):
    def test_generative_round_episode(self):
        env = make_generative_env(
            SCHULZ_FINDING_EXP1_ID, seed=0, n_rounds=2, trials_per_round=3, n_arms=4
        )
        env.reset()
        total = 0.0
        for step in range(6):
            _, reward, terminated, truncated, info = env.step("1")
            total += reward
            self.assertFalse(truncated)
            if step < 5:
                self.assertFalse(terminated)
            else:
                self.assertTrue(terminated)
        self.assertGreater(total, 0.0)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_exp2_has_eight_arms(self):
        text = next(
            json.loads(line)["text"]
            for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
            if json.loads(line)["experiment"] == SCHULZ_FINDING_EXPERIMENT_IDS[1]
        )
        trials = parse_schulz_finding_trials(text)
        self.assertEqual(trials[0].valid_actions, tuple(str(index) for index in range(1, 9)))

    def test_recorded_counterfactual(self):
        trials = parse_schulz_finding_trials(SCHULZ_TEXT)
        trial = trials[0]
        alt = next(action for action in trial.valid_actions if action != trial.human_action)
        env = make_task_env(SCHULZ_FINDING_EXP1_ID, SCHULZ_TEXT)
        env.reset()
        _, reward, _, _, _ = env.step(alt)
        self.assertEqual(reward, trial.outcomes_by_action[alt])


if __name__ == "__main__":
    unittest.main()
