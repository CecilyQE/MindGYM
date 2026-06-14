#!/usr/bin/env python3
"""Audit Psych-101 transcripts via generative envs with human-path replay."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psycenvir.audit.transcript_human_path import (
    DEFAULT_JSONL,
    DEFAULT_PROGRESS_EVERY,
    audit_jsonl,
    default_checkpoint_path,
    supported_task_env_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Psych-101 JSONL export (default: data/raw/prompts_training.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/transcript_human_path_audit.json"),
        help="Write JSON summary here.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file (default: <output>.checkpoint.json).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint writes (progress still prints unless --quiet).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print/save progress every N audited sessions (default: {}).".format(
            DEFAULT_PROGRESS_EVERY
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress lines on stderr.",
    )
    parser.add_argument(
        "--max-per-experiment",
        type=int,
        default=None,
        help="Optional cap on sessions per experiment (default: all).",
    )
    args = parser.parse_args()

    checkpoint_path = None if args.no_checkpoint else (args.checkpoint or default_checkpoint_path(args.output))
    progress_stream = None if args.quiet else sys.stderr

    summary = audit_jsonl(
        jsonl_path=args.jsonl,
        max_sessions_per_experiment=args.max_per_experiment,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        progress_every=args.progress_every,
        progress_stream=progress_stream,
    )
    summary["task_env_experiments"] = list(supported_task_env_experiments())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    total_sessions = sum(row["sessions"] for row in summary["per_experiment"].values())
    total_failed = sum(row["failed"] for row in summary["per_experiment"].values())
    print("Supported experiments:", len(summary["task_env_experiments"]))
    print("Sessions audited:", total_sessions)
    print("Failed sessions:", total_failed)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
