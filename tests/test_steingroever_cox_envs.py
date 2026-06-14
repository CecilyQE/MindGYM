import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import COX_PAIR_EXP1_ID, STEINGROEVER_IGT_EXP1_ID, make_task_env
from psycenvir.psych101.parse import parse_cox_pair_recognition_trials, parse_steingroever_igt_trials

STEINGROEVER_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == STEINGROEVER_IGT_EXP1_ID
)
COX_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == COX_PAIR_EXP1_ID
)


class SteingroeverCoxEnvTests(unittest.TestCase):
    def test_steingroever_generative(self):
        env = make_generative_env(STEINGROEVER_IGT_EXP1_ID, seed=0, n_trials=5)
        env.reset()
        total = 0.0
        for _ in range(5):
            _, reward, terminated, truncated, info = env.step("H")
            total += reward
            self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_steingroever_recorded_counterfactual(self):
        trials = parse_steingroever_igt_trials(STEINGROEVER_TEXT)
        trial = trials[5]
        alt = next(deck for deck in trial.valid_actions if deck != trial.human_action)
        env = make_task_env(STEINGROEVER_IGT_EXP1_ID, STEINGROEVER_TEXT)
        env.reset()
        for prior in trials[:5]:
            env.step(prior.human_action)
        _, reward, _, _, _ = env.step(alt)
        win, loss = trial.outcomes_by_action[alt]
        self.assertEqual(reward, win - loss)

    def test_cox_generative(self):
        env = make_generative_env(COX_PAIR_EXP1_ID, seed=0, n_test_trials=5)
        env.reset()
        for _ in range(5):
            _, reward, terminated, truncated, info = env.step("D")
            self.assertIn(reward, (0.0, 1.0))
            self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_cox_recorded_correctness(self):
        trials = parse_cox_pair_recognition_trials(COX_TEXT)
        trial = trials[0]
        env = make_task_env(COX_PAIR_EXP1_ID, COX_TEXT)
        env.reset()
        _, reward, _, _, info = env.step(trial.correct_action)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["is_correct"])


if __name__ == "__main__":
    unittest.main()
