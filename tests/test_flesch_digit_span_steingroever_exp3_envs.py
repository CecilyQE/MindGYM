import json
import unittest

from psycenvir import make_generative_env
from psycenvir.core.registry import (
    ENKAVI_DIGIT_SPAN_EXP1_ID,
    FLESCH_TREE_EXP1_ID,
    STEINGROEVER_IGT_EXP3_ID,
    make_task_env,
)
from psycenvir.psych101.parse import (
    parse_digit_span_recall_trials,
    parse_flesch_tree_trials,
    parse_steingroever_igt_trials,
)

FLESCH_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == FLESCH_TREE_EXP1_ID
    and "and get -50 points. You would have gotten" in json.loads(line)["text"]
)
DIGIT_SPAN_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == ENKAVI_DIGIT_SPAN_EXP1_ID
)
STEINGROEVER_EXP3_TEXT = next(
    json.loads(line)["text"]
    for line in open("data/raw/prompts_training.jsonl", encoding="utf-8")
    if json.loads(line)["experiment"] == STEINGROEVER_IGT_EXP3_ID
)


class FleschDigitSpanSteingroeverExp3Tests(unittest.TestCase):
    def test_flesch_generative(self):
        env = make_generative_env(FLESCH_TREE_EXP1_ID, seed=0, n_training_trials=4, n_test_trials=2)
        env.reset(seed=0)
        accept = env.accept_key
        for _ in range(6):
            _, reward, terminated, truncated, info = env.step(accept)
            self.assertFalse(truncated)
            self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_transcript_calibrated")

    def test_flesch_generative_silent_no_score_in_observation(self):
        env = make_generative_env(FLESCH_TREE_EXP1_ID, seed=1, n_training_trials=0, n_test_trials=2)
        env.reset(seed=1)
        obs, reward, _, _, info = env.step(env.accept_key)
        self.assertEqual(reward, 0.0)
        self.assertFalse(info["has_feedback"])
        self.assertNotIn("latent_reward", info)
        self.assertNotIn("points.", obs.lower())
        self.assertNotIn("would have", obs.lower())

    def test_flesch_recorded_counterfactual(self):
        trials = parse_flesch_tree_trials(FLESCH_TEXT)
        feedback = next(
            t
            for t in trials
            if t.has_feedback and t.outcomes_by_action is not None
        )
        accept_key, reject_key = feedback.valid_actions
        alt = reject_key if feedback.human_action == accept_key else accept_key
        env = make_task_env(FLESCH_TREE_EXP1_ID, FLESCH_TEXT)
        env.reset()
        for prior in trials[: trials.index(feedback)]:
            env.step(prior.human_action)
        _, reward, _, _, _ = env.step(alt)
        self.assertEqual(reward, feedback.outcomes_by_action[alt])

    def test_digit_span_generative(self):
        env = make_generative_env(ENKAVI_DIGIT_SPAN_EXP1_ID, seed=0, n_spans=2, min_length=2, max_length=3)
        env.reset()
        for _ in range(20):
            _, reward, terminated, truncated, info = env.step("0")
            self.assertIn(reward, (0.0, 1.0))
            if terminated:
                break
            self.assertFalse(truncated)
        self.assertTrue(terminated)

    def test_digit_span_recorded_correctness(self):
        trials = parse_digit_span_recall_trials(DIGIT_SPAN_TEXT)
        trial = trials[0]
        env = make_task_env(ENKAVI_DIGIT_SPAN_EXP1_ID, DIGIT_SPAN_TEXT, include_human_ref=True)
        env.reset()
        _, reward, _, _, info = env.step(trial.correct_action)
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["is_correct"])

    def test_steingroever_exp3_generative(self):
        env = make_generative_env(STEINGROEVER_IGT_EXP3_ID, seed=0, n_trials=5)
        env.reset()
        for deck in ("U", "F", "I", "S", "U"):
            _, reward, terminated, truncated, info = env.step(deck)
            self.assertFalse(truncated)
            self.assertIsInstance(reward, float)
        self.assertTrue(terminated)
        self.assertEqual(info["fidelity_level"], "generative_exact")

    def test_steingroever_exp3_recorded(self):
        trials = parse_steingroever_igt_trials(STEINGROEVER_EXP3_TEXT)
        env = make_task_env(STEINGROEVER_IGT_EXP3_ID, STEINGROEVER_EXP3_TEXT)
        env.reset()
        _, reward, _, _, _ = env.step(trials[0].human_action)
        win, loss = trials[0].outcomes_by_action[trials[0].human_action]
        self.assertEqual(reward, win - loss)


if __name__ == "__main__":
    unittest.main()
