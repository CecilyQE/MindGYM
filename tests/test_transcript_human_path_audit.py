"""Transcript human-path fidelity: TaskEnv steps must match parsed human schedule."""

import json
import unittest
from pathlib import Path

from psycenvir.audit.transcript_human_path import (
    audit_transcript_human_path,
    supported_task_env_experiments,
)
from psycenvir import make_generative_env
from psycenvir.core.registry import FLESCH_TREE_EXP1_ID

JSONL = Path("data/raw/prompts_training.jsonl")


class TranscriptHumanPathAuditTests(unittest.TestCase):
    def test_flesch_generative_silent_api_matches_human_visibility(self):
        env = make_generative_env(FLESCH_TREE_EXP1_ID, seed=0, n_training_trials=0, n_test_trials=2)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(env.accept_key)
        self.assertEqual(reward, 0.0)
        self.assertFalse(info["has_feedback"])
        self.assertNotIn("latent_reward", info)

    def test_each_supported_experiment_has_passing_transcript(self):
        if not JSONL.exists():
            self.skipTest("Missing {}".format(JSONL))
        checked = set()
        with JSONL.open(encoding="utf-8") as handle:
            for session_index, line in enumerate(handle):
                row = json.loads(line)
                experiment_id = row["experiment"]
                if experiment_id not in supported_task_env_experiments():
                    continue
                if experiment_id in checked:
                    continue
                result = audit_transcript_human_path(
                    experiment_id, row["text"], session_index
                )
                if result.ok:
                    checked.add(experiment_id)
        missing = sorted(set(supported_task_env_experiments()) - checked)
        self.assertEqual(
            missing,
            [],
            "No passing human-path transcript yet for: {}".format(missing),
        )


if __name__ == "__main__":
    unittest.main()
