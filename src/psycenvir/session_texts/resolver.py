"""Resolve participant-facing info / consent / debrief for each experiment."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from psycenvir.models import ExperimentSessionTexts, SessionTextBlock, SessionTextSource
from psycenvir.session_texts.instruction_registry import get_registered_info
from psycenvir.session_texts.paper_templates import get_paper_template

DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[3] / "data" / "generated" / "session_texts.yaml"
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "data" / "generated" / "experiments_manifest.csv"
)
DEFAULT_JSONL = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "prompts_training.jsonl"
)

_TASK_START_MARKERS = (
    "You press <<",
    "You say that",
    "You say <<",
    "You estimate <<",
    "Round 1:",
    "Round 1\n",
    "Option ",
    "You see the letter",
    "You see a ",
    "\n\nB:",
    "You encounter a new problem",
    "You will be playing a game for",
    "Each day you will",
    "You will be taking one of the spaceships",
    "You are going to visit",
    "You are shown the letters",
    "Progladine:",
    "Caldionine can take",
)


def _block_from_dict(raw: Dict[str, object]) -> SessionTextBlock:
    return SessionTextBlock(
        text=str(raw.get("text", "")),
        source=SessionTextSource(str(raw.get("source", SessionTextSource.NOT_IN_PAPER.value))),
        note=str(raw.get("note", "")),
    )


def load_session_text_catalog(path: Optional[Path] = None) -> Dict[str, ExperimentSessionTexts]:
    catalog_path = path or DEFAULT_CATALOG
    if not catalog_path.is_file():
        return {}
    with catalog_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    rows: Dict[str, ExperimentSessionTexts] = {}
    for experiment_id, raw in (payload.get("experiments") or {}).items():
        rows[str(experiment_id)] = ExperimentSessionTexts(
            experiment_id=str(experiment_id),
            paper_citation=str(raw.get("paper_citation", "")),
            info=_block_from_dict(raw.get("info", {})),
            consent=_block_from_dict(raw.get("consent", {})),
            debrief=_block_from_dict(raw.get("debrief", {})),
        )
    return rows


def extract_info_from_transcript(text: str) -> str:
    lowered = text
    earliest = len(text)
    for marker in _TASK_START_MARKERS:
        idx = lowered.find(marker)
        if idx > 40 and idx < earliest:
            earliest = idx
    prefix = text[:earliest].strip()
    if len(prefix) < 40:
        return text[: min(800, len(text))].strip()
    return prefix


def resolve_session_texts(
    experiment_id: str,
    *,
    transcript_text: Optional[str] = None,
) -> ExperimentSessionTexts:
    paper = get_paper_template(experiment_id)
    info_entry = get_registered_info(experiment_id)
    if info_entry is not None:
        info_text, info_source, info_note = info_entry
        info = SessionTextBlock(info_text.strip(), info_source, info_note)
    elif transcript_text:
        info = SessionTextBlock(
            extract_info_from_transcript(transcript_text),
            SessionTextSource.FROM_PSYCH101_TRANSCRIPT,
            "Instruction prefix extracted from one Psych-101 transcript before first task event.",
        )
    else:
        info = SessionTextBlock(
            "",
            SessionTextSource.NOT_IN_PAPER,
            "No task instruction registered and no transcript sample available.",
        )
    return ExperimentSessionTexts(
        experiment_id=experiment_id,
        paper_citation=paper.citation,
        info=info,
        consent=paper.consent,
        debrief=paper.debrief,
    )


def list_experiment_ids(manifest_path: Optional[Path] = None) -> List[str]:
    path = manifest_path or DEFAULT_MANIFEST
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["experiment_id"] for row in reader]


def _sample_transcripts(jsonl_path: Path) -> Dict[str, str]:
    samples: Dict[str, str] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            experiment_id = row["experiment"]
            if experiment_id not in samples:
                samples[experiment_id] = row["text"]
    return samples


def build_session_text_catalog(
    experiment_ids: Iterable[str],
    *,
    jsonl_path: Optional[Path] = None,
) -> Dict[str, ExperimentSessionTexts]:
    samples = _sample_transcripts(jsonl_path or DEFAULT_JSONL)
    catalog: Dict[str, ExperimentSessionTexts] = {}
    for experiment_id in experiment_ids:
        catalog[experiment_id] = resolve_session_texts(
            experiment_id,
            transcript_text=samples.get(experiment_id),
        )
    return catalog


def write_session_text_catalog(
    catalog: Dict[str, ExperimentSessionTexts],
    path: Path,
) -> None:
    payload = {
        "version": 1,
        "experiments": {experiment_id: row.as_dict() for experiment_id, row in sorted(catalog.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def render_coverage_markdown(catalog: Dict[str, ExperimentSessionTexts]) -> str:
    lines = [
        "# Session Texts Coverage",
        "",
        "Participant-facing **info** (task instructions), **consent**, and **debrief** per experiment.",
        "Source tags: `verbatim_from_paper`, `reconstructed_from_paper`, `from_task_instruction_module`, "
        "`from_psych101_transcript`, `not_in_paper`.",
        "",
        "| experiment | info source | consent source | debrief source | paper |",
        "|------------|-------------|----------------|----------------|-------|",
    ]
    for experiment_id in sorted(catalog):
        row = catalog[experiment_id]
        lines.append(
            "| `{id}` | `{info}` | `{consent}` | `{debrief}` | {paper} |".format(
                id=experiment_id,
                info=row.info.source.value,
                consent=row.consent.source.value,
                debrief=row.debrief.source.value,
                paper=row.paper_citation.replace("|", "/"),
            )
        )
    lines.extend(["", "## Summary", ""])
    for label, getter in (
        ("info", lambda row: row.info.source),
        ("consent", lambda row: row.consent.source),
        ("debrief", lambda row: row.debrief.source),
    ):
        counts: Dict[str, int] = {}
        for row in catalog.values():
            key = getter(row).value
            counts[key] = counts.get(key, 0) + 1
        lines.append(
            "- **{label}**: {counts}".format(
                label=label,
                counts=", ".join("{}={}".format(k, counts[k]) for k in sorted(counts)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def get_session_texts(experiment_id: str, catalog_path: Optional[Path] = None) -> ExperimentSessionTexts:
    catalog = load_session_text_catalog(catalog_path)
    if experiment_id in catalog:
        return catalog[experiment_id]
    return resolve_session_texts(experiment_id)
