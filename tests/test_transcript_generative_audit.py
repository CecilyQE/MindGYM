"""Transcript audit uses generative env when schedule can be frozen from text."""

import json
import unittest
from pathlib import Path

from psycenvir.audit.transcript_human_path import audit_transcript_human_path
from psycenvir.core.registry import PETERSON_ID
from psycenvir.generative.from_transcript import make_generative_env_from_transcript

JSONL = Path("data/raw/prompts_training.jsonl")


class TranscriptGenerativeAuditTests(unittest.TestCase):
    def test_peterson_generative_no_feedback_block(self):
        text = (
            "Option L delivers 30.0 points with 100.0% chance, or 30.0 points with 0.0% chance.\n"
            "Option B delivers either 0.0 points with unknown chance, or 42.0 points with unknown chance.\n"
            "You press <<L>>.\nYou press <<B>>.\nYou press <<L>>.\nYou press <<B>>.\nYou press <<B>>."
        )
        env = make_generative_env_from_transcript(PETERSON_ID, text)
        env.reset()
        for action in ["L", "B", "L", "B", "B"]:
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, 0.0)
            self.assertIn("You press", observation)
            self.assertNotIn("You receive", observation)
            self.assertFalse(truncated)
            self.assertEqual(
                info.get("fidelity_level"), "generative_transcript_calibrated"
            )
        self.assertTrue(terminated)

    def test_peterson_audit_uses_generative_for_mixed_session(self):
        if not JSONL.exists():
            self.skipTest("Missing {}".format(JSONL))
        with JSONL.open(encoding="utf-8") as handle:
            text = next(
                json.loads(line)["text"]
                for line in handle
                if json.loads(line)["experiment"] == PETERSON_ID
                and "You receive" in json.loads(line)["text"]
            )
        result = audit_transcript_human_path(PETERSON_ID, text)
        self.assertTrue(result.ok, result.issues)


if __name__ == "__main__":
    unittest.main()
