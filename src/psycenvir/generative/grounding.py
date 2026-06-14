"""Transcript- and paper-grounded calibration for generative environments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from psycenvir.psych101.parse import FLESCH_RESPONSE_KEYS_RE, parse_instruction_prefix

DEFAULT_PSYCH101_JSONL = Path(__file__).resolve().parents[3] / "data" / "raw" / "prompts_training.jsonl"
CALIBRATION_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "generated" / "generative_calibration.json"
)
_CALIBRATION_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


class GenerativeGroundingKind(str, Enum):
    """How strongly the generative env is tied to Psych-101 / paper evidence."""

    TRANSCRIPT_CALIBRATED = "transcript_calibrated"
    PAPER_DOCUMENTED = "paper_documented"
    MIXED = "mixed"
    PARTIAL = "partial"


@dataclass(frozen=True)
class GenerativeGroundingProfile:
    experiment_id: str
    kind: GenerativeGroundingKind
    sources: Tuple[str, ...]
    default_config: Dict[str, Any] = field(default_factory=dict)
    caveats: Tuple[str, ...] = ()
    notes: str = ""

    def info_fields(self) -> Dict[str, Any]:
        return {
            "generative_grounding": self.kind.value,
            "generative_sources": list(self.sources),
            "generative_caveats": list(self.caveats),
        }


def _iter_transcripts(
    experiment_id: str, jsonl_path: Path = DEFAULT_PSYCH101_JSONL
) -> Iterable[str]:
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("experiment") == experiment_id:
                yield row["text"]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _calibrate_kool_cost_exp1(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_kool_cost_exp1_days

    topologies: List[Dict[str, Any]] = []
    day_counts: List[int] = []
    mult_rates: List[float] = []
    for text in texts:
        try:
            days = parse_kool_cost_exp1_days(text)
        except Exception:
            continue
        if not days:
            continue
        day = days[0]
        n_days = len(days)
        n_mult = sum(1 for item in days if item.multiplier > 1)
        mult_rates.append(n_mult / n_days)
        day_counts.append(n_days)
        topologies.append(
            {
                "ships": day.ships,
                "planet_by_ship": day.planet_by_ship,
            }
        )
    if not topologies:
        return {}
    return {
        "session_topologies": topologies,
        "n_days": int(round(_median(day_counts))),
        "multiplier_probability": sum(mult_rates) / len(mult_rates),
        "multiplier_value": 5,
    }


def _calibrate_gonogo(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_gonogo_trials

    go_keys: List[str] = []
    practice_counts: List[int] = []
    test_counts: List[int] = []
    colour1_rates: List[float] = []
    for text in texts:
        try:
            trials = parse_gonogo_trials(text)
        except Exception:
            continue
        if not trials:
            continue
        go_keys.append(trials[0].go_key)
        practice = sum(1 for trial in trials if trial.is_practice)
        practice_counts.append(practice)
        test_counts.append(len(trials) - practice)
        colour1_rates.append(
            sum(1 for trial in trials if trial.stimulus == "colour1") / len(trials)
        )
    if not go_keys:
        return {}
    return {
        "go_keys": sorted(set(go_keys)),
        "n_practice_trials": int(round(_median(practice_counts))),
        "n_test_trials": int(round(_median(test_counts))),
        "colour1_probability": sum(colour1_rates) / len(colour1_rates),
    }


def _calibrate_kool_cost_exp2(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_kool_cost_exp2_days

    topologies: List[Dict[str, Any]] = []
    day_counts: List[int] = []
    mult_rates: List[float] = []
    for text in texts:
        try:
            days = parse_kool_cost_exp2_days(text)
        except Exception:
            continue
        if not days:
            continue
        day = days[0]
        n_days = len(days)
        n_mult = sum(1 for item in days if item.multiplier > 1)
        mult_rates.append(n_mult / n_days)
        day_counts.append(n_days)
        topologies.append(
            {
                "ships": day.ships,
                "aliens_by_planet": day.aliens_by_planet,
                "planet_by_ship": day.planet_by_ship,
                "instruction": text.split("\n\n")[0].strip(),
            }
        )
    if not topologies:
        return {}
    return {
        "session_topologies": topologies,
        "n_days": int(round(_median(day_counts))),
        "multiplier_probability": _median(mult_rates),
        "multiplier_value": 5,
        "ship_planet_skew": 0.8,
    }


def _calibrate_flesch(texts: List[str]) -> Dict[str, Any]:
    key_pairs: List[Tuple[str, str]] = []
    training_counts: List[int] = []
    test_counts: List[int] = []
    feedback_re = re.compile(
        r"You get a tree with level \d+ of leafiness and level \d+ of branchiness in the [^.]+\. "
        r"You press <<\w+>> and get -?\d+ points\.",
        re.I,
    )
    no_feedback_re = re.compile(
        r"You get a tree with level \d+ of leafiness and level \d+ of branchiness in the (?:North|South) garden\. "
        r"You press <<\w+>>\.\s*",
        re.I,
    )
    for text in texts:
        match = FLESCH_RESPONSE_KEYS_RE.search(text)
        if match:
            key_pairs.append((match.group(1).strip(), match.group(2).strip()))
        training_counts.append(len(feedback_re.findall(text)))
        test_counts.append(len(no_feedback_re.findall(text)))
    if not key_pairs:
        return {}
    return {
        "response_key_pairs": key_pairs,
        "n_training_trials": int(round(_median(training_counts))),
        "n_test_trials": int(round(_median(test_counts))),
    }


def _calibrate_digit_span(texts: List[str]) -> Dict[str, Any]:
    end_keys: List[str] = []
    span_lengths: List[int] = []
    spans_per_session: List[int] = []
    end_key_re = re.compile(
        r"please press '([^']+)' to indicate the end",
        re.I,
    )
    block_re = re.compile(
        r"The digits are the following: \[([^\]]+)\]",
        re.I,
    )
    for text in texts:
        session_end = end_key_re.search(text)
        if session_end:
            end_keys.append(session_end.group(1).strip())
        blocks = list(block_re.finditer(text))
        spans_per_session.append(len(blocks))
        for block in blocks:
            span_lengths.append(len(block.group(1).split(",")))
    if not span_lengths:
        return {}
    return {
        "end_keys": end_keys,
        "min_length": min(span_lengths),
        "max_length": max(span_lengths),
        "n_spans": int(round(_median(spans_per_session))),
    }


def _calibrate_plonsky(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_plonsky_gamble_trials

    feedback_counts: List[int] = []
    silent_counts: List[int] = []
    for text in texts:
        try:
            trials = parse_plonsky_gamble_trials(text)
        except Exception:
            continue
        feedback_counts.append(sum(1 for trial in trials if trial.has_feedback))
        silent_counts.append(sum(1 for trial in trials if not trial.has_feedback))
    if not feedback_counts:
        return {}
    return {
        "no_feedback_trials": int(round(_median(silent_counts))),
        "trials_per_problem": int(round(_median(feedback_counts) + _median(silent_counts))),
    }


def _calibrate_recent_probes(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_recent_probe_actions, parse_recent_probe_trials

    trial_counts: List[int] = []
    present_keys: List[str] = []
    absent_keys: List[str] = []
    for text in texts:
        try:
            trials = parse_recent_probe_trials(text)
            actions = parse_recent_probe_actions(text)
        except Exception:
            continue
        trial_counts.append(len(trials))
        present_keys.append(actions[0])
        absent_keys.append(actions[1])
    if not trial_counts:
        return {}
    return {
        "n_trials": int(round(_median(trial_counts))),
        "present_key": present_keys[0] if present_keys else "K",
        "absent_key": absent_keys[0] if absent_keys else "D",
    }


def _calibrate_collsi_exp3(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import parse_collsi_judgment_trials

    counts: List[int] = []
    for text in texts:
        try:
            trials = parse_collsi_judgment_trials(text)
        except Exception:
            continue
        counts.append(len(trials))
    if not counts:
        return {}
    return {"n_feedback_trials": int(round(_median(counts))), "n_silent_trials": 0}


def _calibrate_collsi_exp1(texts: List[str]) -> Dict[str, Any]:
    from psycenvir.psych101.parse import COLLSI_ACTION_RE, parse_collsi_judgment_trials

    feedback_counts: List[int] = []
    action_counts: List[int] = []
    for text in texts:
        try:
            trials = parse_collsi_judgment_trials(text)
        except Exception:
            continue
        feedback_counts.append(len(trials))
        action_counts.append(len(COLLSI_ACTION_RE.findall(text)))
    if not feedback_counts:
        return {}
    silent = [max(actions - feedback, 0) for actions, feedback in zip(action_counts, feedback_counts)]
    return {
        "n_feedback_trials": int(round(_median(feedback_counts))),
        "n_silent_trials": int(round(_median(silent))) if silent else 0,
    }


def _calibrate_garcia(texts: List[str]) -> Dict[str, Any]:
    press_counts: List[int] = []
    for text in texts:
        press_counts.append(text.lower().count("you press"))
    if not press_counts:
        return {}
    total = int(round(_median(press_counts)))
    part = total // 3
    return {"part_trial_counts": [part, part, total - 2 * part]}


def _wilson_calibrator(experiment_id: str):
    def _calibrate(texts: List[str]) -> Dict[str, Any]:
        from psycenvir.generative.two_arm_slot import WILSON_EXPERIMENT_CONFIGS

        del texts
        return dict(WILSON_EXPERIMENT_CONFIGS.get(experiment_id, {}))

    return _calibrate


def _calibrate_tomov_subway(texts: List[str]) -> Dict[str, Any]:
    round_counts: List[int] = []
    for text in texts:
        round_counts.append(len(re.findall(r"The new starting station is", text)))
    if not round_counts:
        return {"n_rounds": 20, "n_stations": 6}
    return {"n_rounds": int(round(_median(round_counts))), "n_stations": 6}


def _calibrate_krueger(texts: List[str]) -> Dict[str, Any]:
    round_counts: List[int] = []
    for text in texts:
        round_counts.append(len(re.findall(r"A new round begins", text)))
    if not round_counts:
        return {"n_rounds": 40}
    return {"n_rounds": int(round(_median(round_counts)))}


_CALIBRATORS = {
    "kool2017cost/exp1.csv": _calibrate_kool_cost_exp1,
    "kool2017cost/exp2.csv": _calibrate_kool_cost_exp2,
    "enkavi2019gonogo/exp1.csv": _calibrate_gonogo,
    "flesch2018comparing/exp1.csv": _calibrate_flesch,
    "enkavi2019digitspan/exp1.csv": _calibrate_digit_span,
    "plonsky2018when/exp1.csv": _calibrate_plonsky,
    "enkavi2019recentprobes/exp1.csv": _calibrate_recent_probes,
    "collsiöö2023MCPL/exp1.csv": _calibrate_collsi_exp1,
    "collsiöö2023MCPL/exp3.csv": _calibrate_collsi_exp3,
    "garcia2023experiential/exp1.csv": _calibrate_garcia,
    "garcia2023experiential/exp2.csv": _calibrate_garcia,
    "garcia2023experiential/exp3.csv": _calibrate_garcia,
    "garcia2023experiential/exp4.csv": _calibrate_garcia,
    "wilson2014humans/exp2.csv": _wilson_calibrator("wilson2014humans/exp2.csv"),
    "wilson2014humans/exp3.csv": _wilson_calibrator("wilson2014humans/exp3.csv"),
    "wilson2014humans/exp4.csv": _wilson_calibrator("wilson2014humans/exp4.csv"),
    "wilson2014humans/exp5.csv": _wilson_calibrator("wilson2014humans/exp5.csv"),
    "tomov2020discovery/exp2.csv": _calibrate_tomov_subway,
    "tomov2020discovery/exp4.csv": _calibrate_tomov_subway,
    "tomov2020discovery/exp5.csv": _calibrate_tomov_subway,
    "tomov2020discovery/exp7.csv": _calibrate_tomov_subway,
    "krueger2022identifying/exp1.csv": _calibrate_krueger,
}


_STATIC_PROFILES: Dict[str, GenerativeGroundingProfile] = {
    "badham2017deficits/exp1.csv": GenerativeGroundingProfile(
        experiment_id="badham2017deficits/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Badham2017", "psych101:stimulus_inventory"),
        default_config={"n_problems": 4, "trials_per_problem": 88},
        notes="8 geometric stimuli and 3 rule dimensions match the published paradigm.",
    ),
    "wu2018generalisation/exp1.csv": GenerativeGroundingProfile(
        experiment_id="wu2018generalisation/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wu2018", "psych101:instruction"),
        default_config={"n_environments": 16, "choices_short": 5, "choices_long": 10},
        notes="Spatially correlated 30-arm structure from task instructions.",
    ),
    "frey2017risk/exp1.csv": GenerativeGroundingProfile(
        experiment_id="frey2017risk/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Frey2017", "psych101:instruction"),
        notes="Balloon pump/stop structure from published BART-style instructions.",
    ),
    "peterson2021using/exp1.csv": GenerativeGroundingProfile(
        experiment_id="peterson2021using/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("choices13k:problem_schema", "paper:Peterson2021"),
        default_config={"n_blocks": 20, "trials_per_block": 5},
        notes="Fresh problems are sampled from choices13k with Corr joint sampling and Amb probability hiding.",
    ),
    "wilson2014humans/exp1.csv": GenerativeGroundingProfile(
        experiment_id="wilson2014humans/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wilson2014", "psych101:instruction"),
        default_config={
            "n_games": 3,
            "instructed_trials": 20,
            "free_trials_choices": (30, 30),
            "arms": ("C", "A"),
        },
        notes="Two-arm slot task with instructed then free phases.",
    ),
    "lefebvre2017behavioural/exp1.csv": GenerativeGroundingProfile(
        experiment_id="lefebvre2017behavioural/exp1.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Lefebvre2017", "psych101:reward_pools"),
        default_config={"n_casinos": 4, "visits_per_casino": 25},
        notes="Casino visit structure from instructions; machine payoffs are simulated.",
    ),
    "lefebvre2017behavioural/exp2.csv": GenerativeGroundingProfile(
        experiment_id="lefebvre2017behavioural/exp2.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Lefebvre2017", "psych101:reward_pools"),
        default_config={"n_casinos": 4, "visits_per_casino": 25},
    ),
    "gershman2020reward/exp1.csv": GenerativeGroundingProfile(
        experiment_id="gershman2020reward/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Gershman2020", "psych101:instruction"),
        default_config={"n_games": 13, "trials_per_game": 6},
    ),
    "enkavi2019adaptivenback/exp1.csv": GenerativeGroundingProfile(
        experiment_id="enkavi2019adaptivenback/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Enkavi2019", "psych101:instruction"),
        default_config={"n_blocks": 20, "initial_n": 2, "block_base_trials": 20},
        notes="Adaptive n-back match/non-match rule; RT omitted in v0.",
    ),
    "ruggeri2022globalizability/exp1.csv": GenerativeGroundingProfile(
        experiment_id="ruggeri2022globalizability/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Ruggeri2022", "psych101:choice_items"),
        notes="No objective correctness or reward feedback; choices are preferences.",
    ),
    "feng2021dynamics/exp1.csv": GenerativeGroundingProfile(
        experiment_id="feng2021dynamics/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Feng2021", "psych101:instruction"),
        default_config={"arms": ("I", "H"), "n_games": 160, "instructed_trials": 4, "free_trials_choices": (1, 6)},
        notes="Wilson-like two-arm instructed/free slot-machine task.",
    ),
    "sadeghiyeh2020temporal/exp1.csv": GenerativeGroundingProfile(
        experiment_id="sadeghiyeh2020temporal/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Sadeghiyeh2020", "psych101:instruction"),
        default_config={"arms": ("J", "R"), "n_games": 80, "instructed_trials": 4, "free_trials_choices": (1, 6)},
        notes="Wilson-like two-arm instructed/free slot-machine task.",
    ),
    "somerville2017charting/exp1.csv": GenerativeGroundingProfile(
        experiment_id="somerville2017charting/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Somerville2017", "psych101:instruction"),
        default_config={"arms": ("F", "N"), "n_games": 80, "instructed_trials": 4, "free_trials_choices": (1, 6)},
        notes="Wilson-like two-arm instructed/free slot-machine task.",
    ),
    "waltz2020differential/exp1.csv": GenerativeGroundingProfile(
        experiment_id="waltz2020differential/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Waltz2020", "psych101:instruction"),
        default_config={"arms": ("M", "U"), "n_games": 60, "instructed_trials": 4, "free_trials_choices": (1, 6)},
        notes="Wilson-like two-arm instructed/free slot-machine task.",
    ),
    "wu2023chunking/exp1.csv": GenerativeGroundingProfile(
        experiment_id="wu2023chunking/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wu2023", "psych101:instruction"),
        default_config={"n_trials": 1000},
        notes="Deterministic instructed-key correctness; RT generated for feedback format only.",
    ),
    "wu2023chunking/exp2.csv": GenerativeGroundingProfile(
        experiment_id="wu2023chunking/exp2.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wu2023", "psych101:instruction"),
        default_config={"n_trials": 1000},
        notes="Deterministic instructed-key correctness; RT generated for feedback format only.",
    ),
    "xiong2023neural/exp1.csv": GenerativeGroundingProfile(
        experiment_id="xiong2023neural/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Xiong2023", "psych101:instruction"),
        default_config={
            "arms": ("M", "V"),
            "hazard_rates": (0.1, 0.2, 0.3, 0.4),
            "games_per_hazard": 12,
            "trials_per_game": 100,
        },
        notes="Independent hazard-rate resets; fresh rewards are distributional, not transcript-exact.",
    ),
    "zorowitz2023data/exp1.csv": GenerativeGroundingProfile(
        experiment_id="zorowitz2023data/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Zorowitz2023", "psych101:instruction"),
        default_config={"n_trials": 201, "transition_probability": 0.8},
        notes="Two-step spaceship/alien treasure task with slow drifting alien reward probabilities.",
    ),
    "speekenbrink2008learning/exp1.csv": GenerativeGroundingProfile(
        experiment_id="speekenbrink2008learning/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Speekenbrink2008", "psych101:card_rules"),
        default_config={"n_trials": 100},
    ),
    "frey2017cct/exp1.csv": GenerativeGroundingProfile(
        experiment_id="frey2017cct/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Frey2017", "psych101:task_rules"),
        default_config={"n_rounds": 84},
        notes="32-card Columbia-style task; fresh deck generation not transcript-seeded.",
    ),
    "kool2016when/exp1.csv": GenerativeGroundingProfile(
        experiment_id="kool2016when/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Kool2016", "psych101:instruction"),
        default_config={"n_days": 125},
    ),
    "kool2016when/exp2.csv": GenerativeGroundingProfile(
        experiment_id="kool2016when/exp2.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Kool2016", "psych101:instruction"),
        default_config={"n_days": 125},
    ),
    "kool2017cost/exp1.csv": GenerativeGroundingProfile(
        experiment_id="kool2017cost/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Kool2017", "psych101:session_topologies"),
        default_config={"n_days": 200, "multiplier_value": 5},
        notes="Each episode samples a recorded session's ship/planet mapping.",
    ),
    "enkavi2019gonogo/exp1.csv": GenerativeGroundingProfile(
        experiment_id="enkavi2019gonogo/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Enkavi2019", "psych101:go_keys_and_trial_counts"),
        default_config={"n_practice_trials": 10, "n_test_trials": 350},
        notes="Go key and practice/test counts calibrated from Psych-101 sessions.",
    ),
    "kool2017cost/exp2.csv": GenerativeGroundingProfile(
        experiment_id="kool2017cost/exp2.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Kool2017", "psych101:session_topologies"),
        default_config={"n_days": 200, "multiplier_value": 5},
        notes="Each episode samples a recorded session's ship/planet/alien topology.",
    ),
    "gershman2018deconstructing/exp1.csv": GenerativeGroundingProfile(
        experiment_id="gershman2018deconstructing/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Gershman2018", "psych101:instruction"),
        default_config={"n_games": 20, "trials_per_game": 20},
    ),
    "gershman2018deconstructing/exp2.csv": GenerativeGroundingProfile(
        experiment_id="gershman2018deconstructing/exp2.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Gershman2018", "psych101:instruction"),
        default_config={"n_games": 20, "trials_per_game": 20},
    ),
    "bahrami2020four/exp.csv": GenerativeGroundingProfile(
        experiment_id="bahrami2020four/exp.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Bahrami2020", "psych101:reward_pools"),
        default_config={"n_trials": 120},
    ),
    "hilbig2014generalized/exp1.csv": GenerativeGroundingProfile(
        experiment_id="hilbig2014generalized/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Hilbig2014", "psych101:expert_weights"),
        default_config={"n_trials": 60},
    ),
    "wulff2018description/exp1.csv": GenerativeGroundingProfile(
        experiment_id="wulff2018description/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wulff2018", "psych101:lottery_lines"),
    ),
    "wulff2018sampling/exp1.csv": GenerativeGroundingProfile(
        experiment_id="wulff2018sampling/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Wulff2018", "psych101:KDX_phases"),
    ),
    "plonsky2018when/exp1.csv": GenerativeGroundingProfile(
        experiment_id="plonsky2018when/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Plonsky2018", "psych101:trial_counts"),
    ),
    "schulz2020finding/exp1.csv": GenerativeGroundingProfile(
        experiment_id="schulz2020finding/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Schulz2020", "psych101:instruction"),
        default_config={"n_rounds": 30, "trials_per_round": 10, "n_arms": 8},
    ),
    "tomov2021multitask/exp1.csv": GenerativeGroundingProfile(
        experiment_id="tomov2021multitask/exp1.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Tomov2021", "psych101:door_labels"),
        notes="Simplified castle graph; not full transcript topology.",
        caveats=("Generative graph is schematic, not transcript-isomorphic.",),
    ),
    "tomov2021multitask/exp3.csv": GenerativeGroundingProfile(
        experiment_id="tomov2021multitask/exp3.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Tomov2021", "psych101:door_labels"),
        caveats=("Generative graph is schematic, not transcript-isomorphic.",),
    ),
    "steingroever2015data/exp1.csv": GenerativeGroundingProfile(
        experiment_id="steingroever2015data/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Steingroever2015", "psych101:IGT_payoffs"),
        default_config={"n_trials": 100},
    ),
    "steingroever2015data/exp3.csv": GenerativeGroundingProfile(
        experiment_id="steingroever2015data/exp3.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Steingroever2015", "psych101:IGT_payoffs"),
        default_config={"n_trials": 100},
    ),
    "cox2017information/exp1.csv": GenerativeGroundingProfile(
        experiment_id="cox2017information/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Cox2017", "psych101:word_pool"),
        default_config={"n_studied_pairs": 20, "n_test_trials": 60},
    ),
    "flesch2018comparing/exp1.csv": GenerativeGroundingProfile(
        experiment_id="flesch2018comparing/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Flesch2018", "psych101:response_key_pairs"),
        notes="Accept/reject keys resampled per episode from recorded sessions.",
    ),
    "enkavi2019digitspan/exp1.csv": GenerativeGroundingProfile(
        experiment_id="enkavi2019digitspan/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Enkavi2019", "psych101:end_keys_and_lengths"),
    ),
    "enkavi2019recentprobes/exp1.csv": GenerativeGroundingProfile(
        experiment_id="enkavi2019recentprobes/exp1.csv",
        kind=GenerativeGroundingKind.TRANSCRIPT_CALIBRATED,
        sources=("paper:Enkavi2019", "psych101:probe_rules"),
        notes="Objective correctness from displayed letter set; no trial feedback.",
    ),
    "collsiöö2023MCPL/exp1.csv": GenerativeGroundingProfile(
        experiment_id="collsiöö2023MCPL/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Collsio2023", "psych101:judgment_feedback"),
        notes="Training trials with feedback; silent test trials per paradigm design.",
    ),
    "collsiöö2023MCPL/exp3.csv": GenerativeGroundingProfile(
        experiment_id="collsiöö2023MCPL/exp3.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Collsio2023", "psych101:judgment_feedback"),
    ),
    "garcia2023experiential/exp1.csv": GenerativeGroundingProfile(
        experiment_id="garcia2023experiential/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Garcia2023", "psych101:instruction"),
    ),
    "krueger2022identifying/exp1.csv": GenerativeGroundingProfile(
        experiment_id="krueger2022identifying/exp1.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Krueger2022", "psych101:instruction"),
        notes="Staged gamble-key then color/stop action API; checks cost 4 points.",
    ),
    "steingroever2015data/exp2.csv": GenerativeGroundingProfile(
        experiment_id="steingroever2015data/exp2.csv",
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Steingroever2015", "psych101:IGT_payoffs"),
        default_config={"n_trials": 95},
    ),
    "tomov2020discovery/exp2.csv": GenerativeGroundingProfile(
        experiment_id="tomov2020discovery/exp2.csv",
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Tomov2020", "psych101:direction_keys"),
        caveats=("Fresh grid graphs are schematic, not transcript-isomorphic.",),
    ),
}

for _garcia_index in range(2, 5):
    _STATIC_PROFILES["garcia2023experiential/exp{}.csv".format(_garcia_index)] = GenerativeGroundingProfile(
        experiment_id="garcia2023experiential/exp{}.csv".format(_garcia_index),
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Garcia2023", "psych101:instruction"),
    )

for _ludwig_index in range(3):
    _STATIC_PROFILES["ludwig2023human/exp{}.csv".format(_ludwig_index)] = GenerativeGroundingProfile(
        experiment_id="ludwig2023human/exp{}.csv".format(_ludwig_index),
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Ludwig2023", "psych101:instruction"),
        default_config={"n_blocks": 6, "trials_per_block": 15},
        notes="Two-step fruit-market task with animal-preference dot-product reward.",
    )

def _merge_wilson_static_profiles() -> None:
    from psycenvir.generative.two_arm_slot import WILSON_EXPERIMENT_CONFIGS

    for experiment_id, config in WILSON_EXPERIMENT_CONFIGS.items():
        if experiment_id in _STATIC_PROFILES:
            continue
        _STATIC_PROFILES[experiment_id] = GenerativeGroundingProfile(
            experiment_id=experiment_id,
            kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
            sources=("paper:Wilson2014", "psych101:instruction"),
            default_config=dict(config),
        )


_merge_wilson_static_profiles()

for _tomov_subway_id in (
    "tomov2020discovery/exp4.csv",
    "tomov2020discovery/exp5.csv",
    "tomov2020discovery/exp7.csv",
):
    _STATIC_PROFILES[_tomov_subway_id] = GenerativeGroundingProfile(
        experiment_id=_tomov_subway_id,
        kind=GenerativeGroundingKind.MIXED,
        sources=("paper:Tomov2020", "psych101:direction_keys"),
        caveats=("Fresh grid graphs are schematic, not transcript-isomorphic.",),
    )

for index in range(2, 6):
    _STATIC_PROFILES["schulz2020finding/exp{}.csv".format(index)] = GenerativeGroundingProfile(
        experiment_id="schulz2020finding/exp{}.csv".format(index),
        kind=GenerativeGroundingKind.PAPER_DOCUMENTED,
        sources=("paper:Schulz2020", "psych101:instruction"),
        default_config={"n_rounds": 30, "trials_per_round": 10, "n_arms": 8},
    )


def calibrate_experiment(
    experiment_id: str, jsonl_path: Path = DEFAULT_PSYCH101_JSONL
) -> Dict[str, Any]:
    calibrator = _CALIBRATORS.get(experiment_id)
    if calibrator is None:
        return {}
    texts = list(_iter_transcripts(experiment_id, jsonl_path))
    if not texts:
        return {}
    return calibrator(texts)


def build_grounding_profile(
    experiment_id: str, jsonl_path: Path = DEFAULT_PSYCH101_JSONL
) -> GenerativeGroundingProfile:
    static = _STATIC_PROFILES.get(experiment_id)
    if static is None:
        return GenerativeGroundingProfile(
            experiment_id=experiment_id,
            kind=GenerativeGroundingKind.PARTIAL,
            sources=("unknown",),
            notes="No grounding profile registered.",
        )
    dynamic = calibrate_experiment(experiment_id, jsonl_path)
    merged_config = dict(static.default_config)
    merged_config.update(dynamic)
    return GenerativeGroundingProfile(
        experiment_id=experiment_id,
        kind=static.kind,
        sources=static.sources,
        default_config=merged_config,
        caveats=static.caveats,
        notes=static.notes,
    )


def load_or_build_calibration_cache(
    jsonl_path: Path = DEFAULT_PSYCH101_JSONL,
    cache_path: Path = CALIBRATION_CACHE_PATH,
) -> Dict[str, Dict[str, Any]]:
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            _CALIBRATION_CACHE = json.load(handle)
        for experiment_id, static in _STATIC_PROFILES.items():
            raw = _CALIBRATION_CACHE.get(experiment_id)
            if raw is not None and (
                raw.get("kind") == static.kind.value
                and tuple(raw.get("sources", ())) == static.sources
                and tuple(raw.get("caveats", ())) == static.caveats
                and raw.get("notes", "") == static.notes
            ):
                continue
            merged_config = dict(static.default_config)
            if raw is not None:
                merged_config.update(raw.get("default_config", {}))
            _CALIBRATION_CACHE[experiment_id] = {
                "kind": static.kind.value,
                "sources": list(static.sources),
                "default_config": merged_config,
                "caveats": list(static.caveats),
                "notes": static.notes,
            }
        return _CALIBRATION_CACHE
    cache: Dict[str, Dict[str, Any]] = {}
    for experiment_id in _STATIC_PROFILES:
        profile = build_grounding_profile(experiment_id, jsonl_path)
        cache[experiment_id] = {
            "kind": profile.kind.value,
            "sources": list(profile.sources),
            "default_config": profile.default_config,
            "caveats": list(profile.caveats),
            "notes": profile.notes,
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    _CALIBRATION_CACHE = cache
    return cache


def get_grounding_profile(
    experiment_id: str, jsonl_path: Path = DEFAULT_PSYCH101_JSONL
) -> GenerativeGroundingProfile:
    cache = load_or_build_calibration_cache(jsonl_path)
    raw = cache.get(experiment_id)
    if raw is None:
        return build_grounding_profile(experiment_id, jsonl_path)
    return GenerativeGroundingProfile(
        experiment_id=experiment_id,
        kind=GenerativeGroundingKind(raw["kind"]),
        sources=tuple(raw["sources"]),
        default_config=dict(raw.get("default_config", {})),
        caveats=tuple(raw.get("caveats", [])),
        notes=raw.get("notes", ""),
    )


def apply_generative_defaults(
    experiment_id: str,
    config: Mapping[str, Any],
    profile: Optional[GenerativeGroundingProfile] = None,
) -> Dict[str, Any]:
    profile = profile or get_grounding_profile(experiment_id)
    merged = dict(profile.default_config)
    merged.update(config)
    return merged


class GroundedGenerativeEnv:
    """Delegates to an inner generative env and annotates info with grounding metadata."""

    def __init__(self, inner: Any, profile: GenerativeGroundingProfile) -> None:
        self._inner = inner
        self._profile = profile

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        observation, info = self._inner.reset(seed)
        info = dict(info)
        info.update(self._profile.info_fields())
        return observation, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._inner.step(action)
        info = dict(info)
        info.update(self._profile.info_fields())
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
