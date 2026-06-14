import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import BAHRAMI_ID, HILBIG_ID, make_task_env
from psycenvir.psych101.parse import parse_hilbig_product_trials

BAHRAMI_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == BAHRAMI_ID
)
HILBIG_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == HILBIG_ID
)


class BahramiHilbigEnvTests(unittest.TestCase):
    def test_bahrami_generative_and_recorded(self):
        env = make_generative_env("bahrami2020four/exp.csv", seed=0, n_trials=5, include_human_ref=True)
        _, info = env.reset()
        arm = next(iter(info["outcomes_by_action"]))
        _, reward, _, _, _ = env.step(arm)

        recorded = make_task_env(BAHRAMI_ID, BAHRAMI_TEXT)
        recorded.reset()
        _, reward_rec, _, _, _ = recorded.step("L")
        self.assertIsInstance(reward, float)
        self.assertIsInstance(reward_rec, float)

    def test_hilbig_normative_choice(self):
        trial = parse_hilbig_product_trials(HILBIG_TEXT)[0]
        env = make_task_env(HILBIG_ID, HILBIG_TEXT)
        env.reset()
        _, reward, _, _, _ = env.step(trial.correct_action)
        self.assertEqual(reward, 1.0)

        gen = make_generative_env("hilbig2014generalized/exp1.csv", seed=0, n_trials=3)
        gen.reset()
        _, reward_gen, _, _, info = gen.step("A")
        self.assertIn(reward_gen, (0.0, 1.0))
        self.assertEqual(info["fidelity_level"], "generative_exact")


if __name__ == "__main__":
    unittest.main()
