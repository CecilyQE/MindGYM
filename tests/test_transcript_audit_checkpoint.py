"""Checkpoint resume for transcript human-path audit."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from psycenvir.audit.transcript_human_path import audit_jsonl
from psycenvir.core.registry import PETERSON_ID


class TranscriptAuditCheckpointTests(unittest.TestCase):
    def test_interrupt_resume_continues_without_reprocessing(self):
        jsonl = Path("data/raw/prompts_training.jsonl")
        if not jsonl.exists():
            self.skipTest("Missing {}".format(jsonl))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "audit.checkpoint.json"
            module = __import__(
                "psycenvir.audit.transcript_human_path", fromlist=["audit_transcript_human_path"]
            )
            original = module.audit_transcript_human_path
            calls = {"n": 0}

            def interrupt_after_two(experiment_id, text, session_index=0):
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise KeyboardInterrupt
                return original(experiment_id, text, session_index)

            with mock.patch.object(module, "audit_transcript_human_path", side_effect=interrupt_after_two):
                with self.assertRaises(KeyboardInterrupt):
                    audit_jsonl(
                        jsonl_path=jsonl,
                        experiment_filter=[PETERSON_ID],
                        max_sessions_per_experiment=5,
                        checkpoint_path=checkpoint,
                        progress_every=1,
                        progress_stream=io.StringIO(),
                    )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(saved["audited_sessions"], 2)
            self.assertTrue(checkpoint.exists())

            calls["n"] = 0
            with mock.patch.object(module, "audit_transcript_human_path", side_effect=original):
                summary = audit_jsonl(
                    jsonl_path=jsonl,
                    experiment_filter=[PETERSON_ID],
                    max_sessions_per_experiment=5,
                    checkpoint_path=checkpoint,
                    resume=True,
                    progress_every=500,
                    progress_stream=io.StringIO(),
                )

            self.assertEqual(summary["per_experiment"][PETERSON_ID]["sessions"], 5)
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
