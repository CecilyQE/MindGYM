import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import KOOL_COST_EXP2_ID, make_task_env
from psycenvir.psych101.parse import parse_kool_cost_exp2_days

KOOL_EXP2_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == KOOL_COST_EXP2_ID
)


class KoolCostExp2EnvTests(unittest.TestCase):
    def test_generative_two_step(self):
        env = make_generative_env(KOOL_COST_EXP2_ID, seed=0, n_days=1)
        env.reset(seed=0)
        ship = env._days[0].spaceship_phase.ships[0]
        planet = env._days[0].spaceship_phase.planet_by_ship[ship]
        alien = env._days[0].spaceship_phase.aliens_by_planet[planet][0]
        _, _, _, _, info = env.step(ship)
        self.assertTrue(info["awaiting_alien"])
        _, reward, terminated, truncated, info = env.step(alien)
        self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertFalse(info["awaiting_alien"])
        self.assertGreaterEqual(reward, 0.0)

    def test_recorded_human_path(self):
        days = parse_kool_cost_exp2_days(KOOL_EXP2_TEXT)
        day = days[0]
        env = make_task_env(KOOL_COST_EXP2_ID, KOOL_EXP2_TEXT)
        env.reset()
        env.step(day.human_ship)
        _, reward, _, _, _ = env.step(day.human_alien)
        self.assertEqual(reward, float(day.human_base_treasure * day.multiplier))

    def test_recorded_alien_counterfactual(self):
        days = parse_kool_cost_exp2_days(KOOL_EXP2_TEXT)
        day = days[0]
        aliens = day.aliens_by_planet[day.human_planet]
        alt_alien = aliens[1] if day.human_alien == aliens[0] else aliens[0]
        env = make_task_env(KOOL_COST_EXP2_ID, KOOL_EXP2_TEXT)
        env.reset()
        env.step(day.human_ship)
        _, reward, _, _, _ = env.step(alt_alien)
        expected = day.pooled_treasure_by_alien[alt_alien] * day.multiplier
        self.assertEqual(reward, expected)


if __name__ == "__main__":
    unittest.main()
