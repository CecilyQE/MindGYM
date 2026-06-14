import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import (
    ENKAVI_GONOGO_EXP1_ID,
    KOOL_COST_EXP1_ID,
    make_task_env,
)
from psycenvir.models import GONOGO_NO_PRESS
from psycenvir.psych101.parse import parse_gonogo_trials, parse_kool_cost_exp1_days

KOOL_EXP1_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == KOOL_COST_EXP1_ID
)
GONOGO_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == ENKAVI_GONOGO_EXP1_ID
)


class KoolGonogoEnvTests(unittest.TestCase):
    def test_kool_exp1_parse_coverage(self) -> None:
        days = parse_kool_cost_exp1_days(KOOL_EXP1_TEXT)
        self.assertGreater(len(days), 100)

    def test_kool_exp1_generative(self) -> None:
        env = make_generative_env(KOOL_COST_EXP1_ID, seed=0, n_days=1)
        env.reset(seed=0)
        ship = env._days[0].ships[0]
        _, reward, terminated, _, info = env.step(ship)
        self.assertTrue(terminated)
        self.assertGreaterEqual(reward, 0.0)
        self.assertEqual(info["generative_grounding"], "transcript_calibrated")

    def test_kool_exp1_recorded_counterfactual(self) -> None:
        days = parse_kool_cost_exp1_days(KOOL_EXP1_TEXT)
        day = days[0]
        alt_ship = day.ships[1] if day.human_ship == day.ships[0] else day.ships[0]
        env = make_task_env(KOOL_COST_EXP1_ID, KOOL_EXP1_TEXT)
        env.reset()
        _, reward, _, _, _ = env.step(alt_ship)
        expected = int(round(day.pooled_treasure_by_ship[alt_ship])) * day.multiplier
        self.assertEqual(reward, float(expected))

    def test_gonogo_parse_and_recorded(self) -> None:
        trials = parse_gonogo_trials(GONOGO_TEXT)
        self.assertGreater(len(trials), 300)
        env = make_task_env(ENKAVI_GONOGO_EXP1_ID, GONOGO_TEXT)
        env.reset()
        trial = trials[0]
        action = trial.human_key or GONOGO_NO_PRESS
        _, reward, _, _, info = env.step(action)
        self.assertIn(reward, (0.0, 1.0))
        self.assertEqual(info["fidelity_level"], "exact_transition")

    def test_gonogo_generative(self) -> None:
        env = make_generative_env(
            ENKAVI_GONOGO_EXP1_ID, seed=0, n_practice_trials=2, n_test_trials=3
        )
        env.reset(seed=0)
        go_key = env._go_key
        for _ in range(5):
            step = env._steps[env._step_idx]
            action = go_key if step.stimulus == "colour1" else GONOGO_NO_PRESS
            _, _, terminated, truncated, info = env.step(action)
            self.assertFalse(truncated)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["generative_grounding"], "transcript_calibrated")


if __name__ == "__main__":
    unittest.main()
