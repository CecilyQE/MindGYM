#!/usr/bin/env python3
"""Audit generative env grounding against Psych-101 transcripts and paper metadata."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psycenvir.core.generative_registry import list_generative_experiments  # noqa: E402
from psycenvir.generative.grounding import (  # noqa: E402
    GenerativeGroundingKind,
    get_grounding_profile,
)

OUT_PATH = ROOT / "results" / "generative_grounding_audit.md"


def main() -> None:
    rows: list[tuple[str, str, str, str]] = []
    for experiment_id in sorted(list_generative_experiments()):
        profile = get_grounding_profile(experiment_id)
        caveats = "; ".join(profile.caveats) if profile.caveats else "—"
        sources = ", ".join(profile.sources)
        rows.append((experiment_id, profile.kind.value, sources, caveats))

    lines = [
        "# Generative grounding audit",
        "",
        "Every `make_generative_env` factory entry is annotated with a grounding profile.",
        "Kinds: `transcript_calibrated` (JSONL-derived defaults), `paper_documented` "
        "(design from publication), `mixed`, `partial`.",
        "",
        "| experiment_id | kind | sources | caveats |",
        "|---------------|------|---------|---------|",
    ]
    for experiment_id, kind, sources, caveats in rows:
        lines.append(f"| `{experiment_id}` | {kind} | {sources} | {caveats} |")

    partial = [r for r in rows if r[1] == GenerativeGroundingKind.PARTIAL.value]
    mixed = [r for r in rows if r[1] == GenerativeGroundingKind.MIXED.value]
    lines.extend(
        [
            "",
            f"**Total:** {len(rows)} generative environments.",
            f"**Partial:** {len(partial)} — review before claiming transcript fidelity.",
            f"**Mixed:** {len(mixed)} — paper structure + heuristic or placeholder simulation.",
            "",
        ]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
