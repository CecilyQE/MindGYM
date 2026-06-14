import re
import unittest

from psycenvir import make_generative_env

_SPACESHIP_PAIR_RE = re.compile(r"spaceships (\S+) and (\S+)")
_ALIEN_LINE_RE = re.compile(r"alien (\S+) and alien (\S+)")


class GenerativeEnvTests(unittest.TestCase):
    def test_badham_generative_differs_by_seed_and_is_causal(self):
        env_a = make_generative_env("badham2017deficits/exp1.csv", seed=1, trials_per_problem=4)
        env_b = make_generative_env("badham2017deficits/exp1.csv", seed=2, trials_per_problem=4)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        self.assertNotEqual(obs_a, obs_b)

        env = make_generative_env(
            "badham2017deficits/exp1.csv",
            seed=0,
            trials_per_problem=2,
            n_problems=1,
            label_pool=("A", "B"),
        )
        env.reset()
        feedback_a, reward_a, _, _, info = env.step("A")
        wrong = "B" if info["correct_action"] == "A" else "A"
        feedback_wrong, reward_wrong, _, _, _ = env.step(wrong)
        self.assertIn("The correct category is", feedback_a)
        self.assertIn("The correct category is", feedback_wrong)
        self.assertEqual(reward_a + reward_wrong, 1.0)

    def test_wu_generative_arm_changes_reward(self):
        env = make_generative_env(
            "wu2018generalisation/exp1.csv",
            seed=0,
            n_environments=1,
            choices_short=2,
            choices_long=2,
        )
        env.reset()
        _, reward_a, _, _, _ = env.step("1")
        _, reward_b, _, _, _ = env.step("2")
        self.assertNotEqual(reward_a, reward_b)

    def test_frey_risk_generative_pump_and_collect(self):
        env = make_generative_env(
            "frey2017risk/exp1.csv", seed=0, n_balloons=2, min_threshold=100, max_threshold=100
        )
        env.reset()
        _, reward_pump, _, _, _ = env.step("H")
        self.assertEqual(reward_pump, 0.0)
        _, reward_collect, terminated, truncated, info = env.step("W")
        self.assertGreater(reward_collect, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_peterson_generative_counterfactual(self):
        env = make_generative_env(
            "peterson2021using/exp1.csv",
            seed=0,
            n_blocks=1,
            trials_per_block=1,
            outcome_pairs=[(10.0, 1.0)],
            action_pairs=[("Z", "L")],
        )
        env.reset()
        observation, reward, terminated, truncated, _ = env.step("Z")
        self.assertEqual(reward, 10.0)
        self.assertIn("would have received 1.0", observation)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_same_seed_reproduces_badham_episode(self):
        env_a = make_generative_env(
            "badham2017deficits/exp1.csv", seed=42, n_problems=1, trials_per_problem=3, label_pool=("A", "B")
        )
        env_b = make_generative_env(
            "badham2017deficits/exp1.csv", seed=42, n_problems=1, trials_per_problem=3, label_pool=("A", "B")
        )
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        self.assertEqual(obs_a, obs_b)

    def test_wilson_generative_counterfactual(self):
        env = make_generative_env(
            "wilson2014humans/exp1.csv",
            seed=0,
            n_games=1,
            instructed_trials=0,
            free_trials_choices=(1, 1),
            include_human_ref=True,
        )
        env.reset()
        _, _, _, _, info = env.step("C")
        outcomes = info["outcomes_by_action"]
        self.assertNotEqual(outcomes["C"], outcomes["A"])

    def test_gershman_generative_mapping(self):
        env = make_generative_env(
            "gershman2020reward/exp1.csv", seed=1, n_games=1, trials_per_game=6
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step("S")
        self.assertIn(reward, (0.0, 1.0))
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_speekenbrink_generative_weather(self):
        env = make_generative_env("speekenbrink2008learning/exp1.csv", seed=0, n_trials=5)
        env.reset()
        _, reward, _, _, info = env.step("E")
        self.assertIn(reward, (0.0, 1.0))
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_kool_when_generative_spaceship(self):
        env = make_generative_env("kool2016when/exp1.csv", seed=0, n_days=5, timeout_probability=0.0)
        observation, _ = env.reset()
        match = _SPACESHIP_PAIR_RE.search(observation)
        ship = match.group(1)
        _, reward, _, _, info = env.step(ship)
        self.assertIsNotNone(reward)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_kool_two_step_generative(self):
        env = make_generative_env("kool2016when/exp2.csv", seed=0, n_days=2)
        observation, _ = env.reset()
        match = _SPACESHIP_PAIR_RE.search(observation)
        observation, _, _, _, _ = env.step(match.group(1))
        alien_match = _ALIEN_LINE_RE.search(observation)
        _, reward, terminated, truncated, info = env.step(alien_match.group(1))
        self.assertGreaterEqual(reward, 0.0)
        self.assertFalse(truncated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_gershman_deconstruct_generative(self):
        env = make_generative_env(
            "gershman2018deconstructing/exp1.csv",
            seed=0,
            n_games=1,
            trials_per_game=3,
            include_human_ref=True,
        )
        _, info = env.reset()
        _, reward, _, _, info = env.step(next(iter(info["outcomes_by_action"])))
        self.assertEqual(info["fidelity_level"], "generative_exact")
        self.assertIsInstance(reward, float)

    def test_bahrami_generative_four_arms(self):
        env = make_generative_env(
            "bahrami2020four/exp.csv", seed=0, n_trials=4, include_human_ref=True
        )
        _, info = env.reset()
        arm = next(iter(info["outcomes_by_action"]))
        _, reward, _, _, _ = env.step(arm)
        self.assertIsInstance(reward, float)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_wulff_description_generative(self):
        env = make_generative_env("wulff2018description/exp1.csv", seed=0, n_problems=2)
        env.reset()
        _, reward, _, _, info = env.step("W")
        self.assertIsInstance(reward, float)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_frey_cct_generative_turn_and_stop(self):
        env = make_generative_env("frey2017cct/exp1.csv", seed=0, n_rounds=2)
        env.reset()
        _, reward_turn, _, _, _ = env.step("E")
        self.assertNotEqual(reward_turn, 0.0)
        _, reward_stop, terminated, truncated, info = env.step("C")
        self.assertEqual(reward_stop, 0.0)
        self.assertFalse(truncated)
        self.assertEqual(info["fidelity_level"], "generative_exact")
        self.assertFalse(terminated)


if __name__ == "__main__":
    unittest.main()
