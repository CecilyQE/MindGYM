"""Generate the Phase 0 experiment manifest from a local Psych-101 export."""

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List

import yaml

_GENERATIVE_SETTING_TIERS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "generated" / "generative_setting_tiers.yaml"
)


def _load_generative_setting_tiers() -> Dict[str, str]:
    if not _GENERATIVE_SETTING_TIERS_PATH.exists():
        return {}
    with _GENERATIVE_SETTING_TIERS_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    mapping: Dict[str, str] = {}
    for experiment_id in raw.get("tier_A", []):
        mapping[str(experiment_id)] = "A"
    for experiment_id in raw.get("tier_C", []):
        mapping[str(experiment_id)] = "C"
    return mapping


PRESS_RE = re.compile(r"\bpress\s+<<", re.IGNORECASE)
SAY_RE = re.compile(r"\bYou\s+say(?:\s+that\b[^\n]*?)?\s+<<", re.IGNORECASE)

P1_IDS = {
    "badham2017deficits/exp1.csv",
    "peterson2021using/exp1.csv",
    "frey2017cct/exp1.csv",
}
P2_IDS = {
    "frey2017risk/exp1.csv",
    "wu2018generalisation/exp1.csv",
    "collsiöö2023MCPL/exp1.csv",
    "collsiöö2023MCPL/exp3.csv",
    "hilbig2014generalized/exp1.csv",
    "bahrami2020four/exp.csv",
    "enkavi2019recentprobes/exp1.csv",
    "gershman2020reward/exp1.csv",
    "cox2017information/exp1.csv",
}
MANIFEST_FIELDS = [
    "experiment_id",
    "n_participants",
    "n_press_median",
    "n_say_median",
    "feedback_class",
    "sim_tier_auto",
    "sim_tier_final",
    "v1_priority",
    "family",
    "reward_mode",
    "has_score",
    "has_monetary_bonus",
    "ethics_recorded",
    "payoff_rule_recorded",
    "verbatim_text_available",
    "sources",
    "notes",
    "generative_setting_tier",
]
REVIEWED_OVERRIDES = {
    "badham2017deficits/exp1.csv": {
        "sim_tier_final": "1_exact",
        "family": "category_learning",
        "has_monetary_bonus": "false",
        "ethics_recorded": "true",
        "payoff_rule_recorded": "true",
        "verbatim_text_available": "false",
        "sources": "Psych-101; Badham, Sanborn & Maylor (2017)",
        "notes": "Reviewed: correctness feedback is exact; compensation is course credit or fixed GBP 5, not performance bonus.",
    },
    "peterson2021using/exp1.csv": {
        "sim_tier_final": "1_partial",
        "family": "gamble_choice",
        "has_monetary_bonus": "true",
        "ethics_recorded": "true",
        "payoff_rule_recorded": "true",
        "verbatim_text_available": "false",
        "sources": "Psych-101; Peterson et al. (2021); Thomas et al. (2024)",
        "notes": "Reviewed: exact only on feedback blocks; fresh outcomes require Corr/Amb joint sampling.",
    },
    "frey2017cct/exp1.csv": {
        "sim_tier_final": "1_partial",
        "family": "columbia_card_task",
        "has_monetary_bonus": "true",
        "ethics_recorded": "true",
        "payoff_rule_recorded": "partial",
        "verbatim_text_available": "false",
        "sources": "Psych-101; Frey et al. (2017); CCT instructions",
        "notes": "Reviewed: exact recorded paths and early stops; fresh deck generation and exact money conversion remain open.",
    },
    "frey2017risk/exp1.csv": {
        "sim_tier_final": "2_pending",
        "family": "balloon_risk_task",
        "has_monetary_bonus": "true",
        "ethics_recorded": "true",
        "payoff_rule_recorded": "partial",
        "verbatim_text_available": "false",
        "sources": "Psych-101; Frey et al. (2017)",
        "notes": "Reviewed candidate: explosion distribution and exact money conversion must be recovered before implementation.",
    },
    "wu2018generalisation/exp1.csv": {
        "sim_tier_final": "2_pending",
        "family": "spatial_bandit",
        "has_monetary_bonus": "unknown",
        "ethics_recorded": "unknown",
        "payoff_rule_recorded": "partial",
        "verbatim_text_available": "unknown",
        "sources": "Psych-101; Wu et al. (2018)",
        "notes": "Reviewed candidate: numeric reward mechanism is clear; compensation and generator details remain to be recovered.",
    },
    "collsiöö2023MCPL/exp1.csv": {
        "sim_tier_final": "2_pending",
        "family": "judgment_learning",
        "sources": "Psych-101",
        "notes": "Reviewed candidate: feedback ceases before test trials and some test stimuli have no recovered label, so exact simulation remains blocked.",
    },
    "collsiöö2023MCPL/exp3.csv": {
        "sim_tier_final": "1_exact",
        "family": "judgment_learning",
        "has_monetary_bonus": "unknown",
        "ethics_recorded": "unknown",
        "payoff_rule_recorded": "unknown",
        "verbatim_text_available": "unknown",
        "sources": "Psych-101",
        "notes": "Implemented: all recorded trials expose correct-concentration feedback; exact transitions on recovered stimulus schedules.",
    },
    "enkavi2019recentprobes/exp1.csv": {
        "sim_tier_final": "1_exact",
        "family": "recognition_memory",
        "has_monetary_bonus": "unknown",
        "ethics_recorded": "unknown",
        "payoff_rule_recorded": "unknown",
        "verbatim_text_available": "unknown",
        "sources": "Psych-101",
        "notes": "Implemented: present/absent correctness is deterministically computed from displayed letter set; original trials provide no feedback or scalar reward.",
    },
    "gershman2020reward/exp1.csv": {
        "sim_tier_final": "2_pending",
        "family": "stimulus_response_learning",
        "sources": "Psych-101",
        "notes": "Reviewed candidate: labels are recoverable for most stimulus-game instances, but seven session-local stimuli never expose a rewarded response, preventing full exact-transition registration.",
    },
    "cox2017information/exp1.csv": {
        "sim_tier_final": "2_pending",
        "family": "episodic_memory_battery",
        "sources": "Psych-101",
        "notes": "Reviewed candidate: sessions mix pair recognition, item recognition, and free recall blocks; an exact environment requires family-specific subtask parsers and recall scoring rules.",
    },
}


def _read_table(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        def iter_jsonl() -> Iterator[Dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

        return iter_jsonl()

    import pandas as pd

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".json":
        frame = pd.read_json(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Input must be .parquet, .jsonl, .json, or .csv.")
    return frame.to_dict(orient="records")


def classify_feedback(texts: Iterable[str]) -> str:
    joined = "\n".join(texts).lower()
    if "the correct category is" in joined or "correct concentration" in joined:
        return "F1_correctness"
    if "turn over a loss card" in joined or "round has now ended" in joined:
        return "F4_state_machine"
    if "balloon" in joined and ("explodes" in joined or "explode" in joined):
        return "F3_stochastic_outcome"
    if "receive" in joined and "points" in joined:
        return "F2_outcome"
    return "F_other"


def infer_reward_mode(feedback_class: str) -> str:
    if feedback_class == "F1_correctness":
        return "correctness"
    if feedback_class in {"F2_outcome", "F3_stochastic_outcome", "F4_state_machine"}:
        return "outcome"
    return "none"


def build_manifest(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    setting_tiers = _load_generative_setting_tiers()
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in records:
        experiment_id = row.get("experiment", row.get("experiment_id"))
        text = row.get("text")
        if not experiment_id or not isinstance(text, str):
            raise ValueError("Each record must contain an experiment/experiment_id and text.")
        group = grouped.setdefault(
            str(experiment_id),
            {
                "n_participants": 0,
                "press_counts": [],
                "say_counts": [],
                "texts_for_classification": [],
            },
        )
        group["n_participants"] += 1
        group["press_counts"].append(len(PRESS_RE.findall(text)))
        group["say_counts"].append(len(SAY_RE.findall(text)))
        # Feedback classification depends only on marker presence, not full transcripts.
        lowered = text.lower()
        markers = []
        if "the correct category is" in lowered or "correct concentration" in lowered:
            markers.append("the correct category is")
        if "turn over a loss card" in lowered or "round has now ended" in lowered:
            markers.append("round has now ended")
        if "balloon" in lowered and ("explodes" in lowered or "explode" in lowered):
            markers.append("balloon explodes")
        if "receive" in lowered and "points" in lowered:
            markers.append("receive points")
        group["texts_for_classification"].extend(markers)

    manifest: List[Dict[str, Any]] = []
    for experiment_id, group in sorted(grouped.items()):
        feedback_class = classify_feedback(group["texts_for_classification"])
        press_counts = group["press_counts"]
        say_counts = group["say_counts"]
        has_action = max(press_counts + say_counts) > 0
        if feedback_class != "F_other":
            auto_tier = "1"
        elif has_action:
            auto_tier = "2"
        else:
            auto_tier = "3"
        priority = (
            "P1" if experiment_id in P1_IDS else "P2" if experiment_id in P2_IDS else "defer"
        )
        manifest.append(
            {
                "experiment_id": experiment_id,
                "n_participants": group["n_participants"],
                "n_press_median": median(press_counts),
                "n_say_median": median(say_counts),
                "feedback_class": feedback_class,
                "sim_tier_auto": auto_tier,
                "sim_tier_final": "",
                "v1_priority": priority,
                "family": "",
                "reward_mode": infer_reward_mode(feedback_class),
                "has_score": feedback_class != "F_other",
                "has_monetary_bonus": "",
                "ethics_recorded": "",
                "payoff_rule_recorded": "",
                "verbatim_text_available": "",
                "sources": "Psych-101",
                "notes": "Auto-generated; requires manual source review.",
                "generative_setting_tier": setting_tiers.get(experiment_id, ""),
                **REVIEWED_OVERRIDES.get(experiment_id, {}),
            }
        )
    return manifest


def write_manifest(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local Psych-101 table export.")
    parser.add_argument("output", type=Path, help="Manifest CSV output path.")
    args = parser.parse_args()
    rows = build_manifest(_read_table(args.input))
    write_manifest(rows, args.output)
    print("Wrote {} experiment rows to {}.".format(len(rows), args.output))


if __name__ == "__main__":
    main()
