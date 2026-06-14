import json
import re
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import PLONSKY_ID, WULFF_SAMPLING_ID, make_task_env
from psycenvir.psych101.parse import parse_plonsky_gamble_trials, parse_wulff_sampling_problems

WULFF_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == WULFF_SAMPLING_ID
)
PLONSKY_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == PLONSKY_ID
)


class WulffPlonskyEnvTests(unittest.TestCase):
    def test_wulff_sampling_generative_phases(self):
        env = make_generative_env(
            "wulff2018sampling/exp1.csv", seed=0, n_problems=1, max_samples_before_stop=5
        )
        env.reset()
        _, _, _, _, _ = env.step("K")
        _, _, _, _, _ = env.step("D")
        _, reward, terminated, truncated, info = env.step("X")
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, reward_final, terminated, _, info = env.step("K")
        self.assertIsInstance(reward_final, float)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_wulff_sampling_recorded_final_draw(self):
        problems = parse_wulff_sampling_problems(WULFF_TEXT)
        problem = problems[0]
        arm0 = problem.sampling_arms[0]
        env = make_task_env(WULFF_SAMPLING_ID, WULFF_TEXT)
        env.reset()
        env.step(arm0)
        env.step(problem.stop_action)
        arm = problem.human_final_action
        _, reward, _, _, _ = env.step(arm)
        self.assertEqual(reward, problem.final_outcomes_by_action[arm])

    def test_plonsky_generative_feedback(self):
        env = make_generative_env(
            "plonsky2018when/exp1.csv",
            seed=0,
            n_problems=1,
            trials_per_problem=6,
            no_feedback_trials=2,
        )
        observation, _ = env.reset()
        keys = re.findall(r"Option (\w+) delivers", observation)
        self.assertEqual(len(keys), 2)
        _, r0, _, _, _ = env.step(keys[0])
        self.assertEqual(r0, 0.0)
        env.step(keys[0])
        env.step(keys[1])
        env.step(keys[0])
        env.step(keys[1])
        _, reward, terminated, _, info = env.step(keys[0])
        self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_plonsky_recorded_no_feedback_advances(self):
        trials = parse_plonsky_gamble_trials(PLONSKY_TEXT)
        env = make_task_env(PLONSKY_ID, PLONSKY_TEXT)
        env.reset()
        for trial in trials[:5]:
            self.assertFalse(trial.has_feedback)
            _, reward, terminated, truncated, _ = env.step(trial.human_action)
            self.assertEqual(reward, 0.0)
            self.assertFalse(truncated)
            self.assertFalse(terminated)
        self.assertTrue(trials[5].has_feedback)

    def test_plonsky_recorded_counterfactual(self):
        trials = parse_plonsky_gamble_trials(PLONSKY_TEXT)
        index = next(i for i, trial in enumerate(trials) if trial.has_feedback)
        trial = trials[index]
        alt = trial.valid_actions[1] if trial.human_action == trial.valid_actions[0] else trial.valid_actions[0]
        env = make_task_env(PLONSKY_ID, PLONSKY_TEXT)
        env.reset()
        for prior in trials[:index]:
            env.step(prior.human_action)
        _, reward, _, _, _ = env.step(alt)
        self.assertEqual(reward, trial.outcomes_by_action[alt])


if __name__ == "__main__":
    unittest.main()
