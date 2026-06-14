"""Replay each transcript's human actions through generative envs and check outcomes.

Replays each transcript through ``make_generative_env_from_transcript``: generative
``step()`` logic with the episode schedule frozen from that transcript (no fresh
``reset(seed)`` draw).
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TextIO

from psycenvir.errors import EnvironmentNotReadyError, TranscriptParseError
from psycenvir.generative.from_transcript import make_generative_env_from_transcript

DEFAULT_JSONL = Path(__file__).resolve().parents[3] / "data" / "raw" / "prompts_training.jsonl"
CHECKPOINT_VERSION = 1
DEFAULT_PROGRESS_EVERY = 500


@dataclass
class StepExpectation:
    action: str
    expected_reward: Optional[float] = None
    reward_tolerance: float = 1e-6
    observation_must_contain: Tuple[str, ...] = ()
    observation_must_not_contain: Tuple[str, ...] = ()


@dataclass
class SessionAuditResult:
    experiment_id: str
    session_index: int
    n_steps: int
    ok: bool
    skipped: bool = False
    issues: List[str] = field(default_factory=list)


def _expect_reward(trial: Any, attribute: str = "human_action") -> Optional[float]:
    action = getattr(trial, attribute)
    outcomes = getattr(trial, "outcomes_by_action", None)
    if outcomes is None:
        return None
    if isinstance(outcomes[action], tuple):
        win, loss = outcomes[action]
        return float(win) - float(loss)
    return float(outcomes[action])


def _flesch_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_flesch_tree_trials

    steps: List[StepExpectation] = []
    for trial in parse_flesch_tree_trials(text):
        if trial.has_feedback and trial.outcomes_by_action is not None:
            reward = _expect_reward(trial)
            steps.append(
                StepExpectation(
                    action=trial.human_action,
                    expected_reward=reward,
                    observation_must_contain=("and get",),
                )
            )
        else:
            steps.append(
                StepExpectation(
                    action=trial.human_action,
                    expected_reward=0.0,
                    observation_must_not_contain=("and get", "would have gotten"),
                )
            )
    return steps


def _trial_list_expectations(
    text: str,
    parser: Callable[[str], Sequence[Any]],
    *,
    reward_from_trial: Optional[Callable[[Any], Optional[float]]] = None,
) -> List[StepExpectation]:
    reward_from_trial = reward_from_trial or _expect_reward
    return [
        StepExpectation(
            action=trial.human_action,
            expected_reward=reward_from_trial(trial),
        )
        for trial in parser(text)
    ]


def _peterson_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_peterson_gamble_blocks

    steps: List[StepExpectation] = []
    for block in parse_peterson_gamble_blocks(text):
        for trial in block.trials:
            if block.has_feedback:
                outcomes = trial.outcomes_by_action
                if outcomes is None:
                    continue
                steps.append(
                    StepExpectation(
                        action=trial.human_action,
                        expected_reward=float(outcomes[trial.human_action]),
                        observation_must_contain=("You receive",),
                    )
                )
            else:
                steps.append(
                    StepExpectation(
                        action=trial.human_action,
                        expected_reward=0.0,
                        observation_must_not_contain=(
                            "You receive",
                            "would have received",
                        ),
                    )
                )
    return steps


def _frey_cct_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_frey_cct_rounds

    steps: List[StepExpectation] = []
    for round_data in parse_frey_cct_rounds(text):
        for event in round_data.events:
            steps.append(StepExpectation(action=event.human_action))
    return steps


def _kool_cost_exp1_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_kool_cost_exp1_days

    return [StepExpectation(action=day.human_ship) for day in parse_kool_cost_exp1_days(text)]


def _kool_cost_exp2_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_kool_cost_exp2_days

    steps: List[StepExpectation] = []
    for day in parse_kool_cost_exp2_days(text):
        steps.append(StepExpectation(action=day.human_ship, expected_reward=0.0))
        steps.append(StepExpectation(action=day.human_alien))
    return steps


def _gonogo_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.models import GONOGO_NO_PRESS
    from psycenvir.psych101.parse import parse_gonogo_trials

    steps: List[StepExpectation] = []
    for trial in parse_gonogo_trials(text):
        action = trial.human_key if trial.human_key is not None else GONOGO_NO_PRESS
        if trial.stimulus == "colour1":
            correct = trial.human_key == trial.go_key
        else:
            correct = trial.human_key is None
        steps.append(
            StepExpectation(action=action, expected_reward=1.0 if correct else 0.0)
        )
    return steps


def _digit_span_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_digit_span_recall_trials

    return [
        StepExpectation(
            action=trial.human_action,
            expected_reward=1.0 if trial.human_action == trial.correct_action else 0.0,
        )
        for trial in parse_digit_span_recall_trials(text)
    ]


def _plonsky_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_plonsky_gamble_trials

    steps: List[StepExpectation] = []
    for trial in parse_plonsky_gamble_trials(text):
        if trial.has_feedback and trial.outcomes_by_action is not None:
            steps.append(
                StepExpectation(
                    action=trial.human_action,
                    expected_reward=_expect_reward(trial),
                )
            )
        else:
            steps.append(
                StepExpectation(
                    action=trial.human_action,
                    expected_reward=0.0,
                )
            )
    return steps


def _build_expectation_registry() -> Dict[str, Callable[[str], List[StepExpectation]]]:
    from psycenvir.core import registry as reg
    from psycenvir.generative.adaptive_nback import ENKAVI_ADAPTIVE_NBACK_EXP1_ID
    from psycenvir.generative.balloon import FREY_RISK_EXPERIMENT_ID
    from psycenvir.generative.chunking import WU_CHUNKING_EXP1_ID, WU_CHUNKING_EXP2_ID
    from psycenvir.generative.collsi_judgment import COLLSI_EXP1_ID
    from psycenvir.generative.garcia_experiential import GARCIA_EXPERIENTIAL_IDS
    from psycenvir.generative.hazard_bandit import XIONG_NEURAL_EXP1_ID
    from psycenvir.generative.kool_spaceship import KOOL_WHEN_EXP1_ID
    from psycenvir.generative.kool_two_step import KOOL_WHEN_EXP2_ID
    from psycenvir.generative.krueger_identifying import KRUEGER_IDENTIFYING_EXP1_ID
    from psycenvir.generative.ludwig_fruit import LUDWIG_HUMAN_EXPERIMENT_IDS
    from psycenvir.generative.ruggeri_choice import RUGGERI_GLOBALIZABILITY_EXP1_ID
    from psycenvir.generative.steingroever_igt import STEINGROEVER_IGT_EXP2_ID
    from psycenvir.generative.two_arm_slot import WILSON_EXPERIMENT_CONFIGS
    from psycenvir.generative.volatile_bandit import GERSHMAN_DECONSTRUCT_EXP1_ID
    from psycenvir.generative.wu_bandit import WU_EXPERIMENT_ID
    from psycenvir.generative.zorowitz_space import ZOROWITZ_DATA_EXP1_ID
    from psycenvir.psych101 import parse as p

    registry: Dict[str, Callable[[str], List[StepExpectation]]] = {
        reg.BADHAM_ID: lambda text: _trial_list_expectations(text, p.parse_badham_category_trials),
        reg.PETERSON_ID: _peterson_expectations,
        reg.FREY_CCT_ID: _frey_cct_expectations,
        FREY_RISK_EXPERIMENT_ID: _frey_risk_expectations,
        COLLSI_EXP1_ID: _collsi_exp1_expectations,
        reg.COLLSI_EXP3_ID: lambda text: _trial_list_expectations(
            text, p.parse_collsi_judgment_trials
        ),
        reg.ENKAVI_RECENT_PROBES_ID: lambda text: _trial_list_expectations(
            text, p.parse_recent_probe_trials, reward_from_trial=lambda _: 0.0
        ),
        ENKAVI_ADAPTIVE_NBACK_EXP1_ID: _adaptive_nback_expectations,
        reg.GERSHMAN_MAPPING_ID: lambda text: _trial_list_expectations(
            text, p.parse_gershman_mapping_trials
        ),
        RUGGERI_GLOBALIZABILITY_EXP1_ID: _ruggeri_expectations,
        WU_CHUNKING_EXP1_ID: _chunking_expectations,
        WU_CHUNKING_EXP2_ID: _chunking_expectations,
        reg.SPEEKENBRINK_WEATHER_ID: lambda text: _trial_list_expectations(
            text, p.parse_speekenbrink_weather_trials
        ),
        reg.LEFEBVRE_EXP1_ID: lambda text: _trial_list_expectations(
            text, p.parse_lefebvre_casino_trials
        ),
        reg.LEFEBVRE_EXP2_ID: lambda text: _trial_list_expectations(
            text, p.parse_lefebvre_casino_trials
        ),
        reg.BAHRAMI_ID: lambda text: _trial_list_expectations(text, p.parse_bahrami_four_arm_trials),
        reg.HILBIG_ID: lambda text: _trial_list_expectations(text, p.parse_hilbig_product_trials),
        reg.WULFF_DESCRIPTION_ID: lambda text: _trial_list_expectations(
            text, p.parse_wulff_description_trials
        ),
        reg.PLONSKY_ID: _plonsky_expectations,
        GERSHMAN_DECONSTRUCT_EXP1_ID: lambda text: _trial_list_expectations(
            text, p.parse_gershman_bandit_trials
        ),
        reg.GERSHMAN_DECONSTRUCT_EXP2_ID: lambda text: _trial_list_expectations(
            text, p.parse_gershman_bandit_trials
        ),
        reg.FLESCH_TREE_EXP1_ID: _flesch_expectations,
        reg.ENKAVI_DIGIT_SPAN_EXP1_ID: _digit_span_expectations,
        reg.KOOL_COST_EXP1_ID: _kool_cost_exp1_expectations,
        reg.KOOL_COST_EXP2_ID: _kool_cost_exp2_expectations,
        reg.ENKAVI_GONOGO_EXP1_ID: _gonogo_expectations,
        reg.COX_PAIR_EXP1_ID: lambda text: _trial_list_expectations(
            text,
            p.parse_cox_pair_recognition_trials,
            reward_from_trial=lambda trial: 1.0 if trial.human_action == trial.correct_action else 0.0,
        ),
        KOOL_WHEN_EXP1_ID: _kool_exp1_expectations,
        KOOL_WHEN_EXP2_ID: _kool_exp2_expectations,
        KRUEGER_IDENTIFYING_EXP1_ID: _krueger_expectations,
        WU_EXPERIMENT_ID: _wu_bandit_expectations,
        XIONG_NEURAL_EXP1_ID: _xiong_expectations,
        ZOROWITZ_DATA_EXP1_ID: _zorowitz_expectations,
    }
    for experiment_id in GARCIA_EXPERIENTIAL_IDS:
        registry[experiment_id] = _garcia_expectations
    for experiment_id in LUDWIG_HUMAN_EXPERIMENT_IDS:
        registry[experiment_id] = _ludwig_expectations
    for experiment_id in reg.SCHULZ_FINDING_EXPERIMENT_IDS:
        registry[experiment_id] = lambda text, eid=experiment_id: _trial_list_expectations(
            text, p.parse_schulz_finding_trials
        )
    for experiment_id in reg.TOMOV_SUBWAY_EXPERIMENT_IDS:
        registry[experiment_id] = lambda text, eid=experiment_id: _trial_list_expectations(
            text, p.parse_tomov_subway_trials
        )
    for experiment_id in WILSON_EXPERIMENT_CONFIGS:
        registry[experiment_id] = lambda text, eid=experiment_id: _trial_list_expectations(
            text, p.parse_wilson_slot_trials
        )
    for experiment_id in (
        reg.STEINGROEVER_IGT_EXP1_ID,
        STEINGROEVER_IGT_EXP2_ID,
        reg.STEINGROEVER_IGT_EXP3_ID,
    ):
        registry[experiment_id] = lambda text, eid=experiment_id: _trial_list_expectations(
            text,
            p.parse_steingroever_igt_trials,
            reward_from_trial=lambda trial: float(
                trial.outcomes_by_action[trial.human_action][0]
            )
            - float(trial.outcomes_by_action[trial.human_action][1]),
        )
    for experiment_id in reg.TOMOV_CASTLE_EXPERIMENT_IDS:
        registry[experiment_id] = lambda text, eid=experiment_id: _trial_list_expectations(
            text, p.parse_tomov_castle_trials
        )
    registry[reg.WULFF_SAMPLING_ID] = _wulff_sampling_expectations
    return registry


def _wulff_sampling_expectations(text: str) -> List[StepExpectation]:
    from psycenvir.psych101.parse import parse_wulff_sampling_problems

    steps: List[StepExpectation] = []
    for problem in parse_wulff_sampling_problems(text):
        if problem.sample_sequence:
            for arm, _ in problem.sample_sequence:
                steps.append(
                    StepExpectation(
                        action=arm,
                        expected_reward=0.0,
                        observation_must_contain=("observe", "points"),
                    )
                )
        else:
            for arm in problem.sampling_arms:
                for _ in problem.sample_pools.get(arm, []):
                    steps.append(
                        StepExpectation(
                            action=arm,
                            expected_reward=0.0,
                            observation_must_contain=("observe", "points"),
                        )
                    )
        if problem.stop_action is not None:
            steps.append(StepExpectation(action=problem.stop_action, expected_reward=0.0))
        steps.append(
            StepExpectation(
                action=problem.human_final_action,
                expected_reward=float(
                    problem.final_outcomes_by_action[problem.human_final_action]
                ),
            )
        )
    return steps


def _chunking_expectations(text: str) -> List[StepExpectation]:
    steps: List[StepExpectation] = []
    for match in re.finditer(
        r"The instruction is to press (?P<target>\w+), you press <<(?P<action>[^<>]+)>> in (?P<rt>\d+) ms\. That is (?P<correctness>correct|incorrect)\.",
        text,
        re.IGNORECASE,
    ):
        steps.append(
            StepExpectation(
                action=match.group("action").strip(),
                expected_reward=1.0 if match.group("correctness").lower() == "correct" else 0.0,
                observation_must_contain=("That is {}".format(match.group("correctness").lower()),),
            )
        )
    return steps


def _ruggeri_expectations(text: str) -> List[StepExpectation]:
    return [
        StepExpectation(action=match.group("action").strip(), expected_reward=0.0)
        for match in re.finditer(
            r"You have the choice between .*? \(press \w+\) or .*? \(press \w+\)\. You press <<(?P<action>[^<>]+)>>\.",
            text,
            re.IGNORECASE,
        )
    ]


def _adaptive_nback_expectations(text: str) -> List[StepExpectation]:
    steps: List[StepExpectation] = []
    key_match = re.search(
        r"matches\s+the\s+letter\s+N\s+trials\s+ago,\s+press\s+(?P<match>\w+),\s+otherwise\s+press\s+(?P<nonmatch>\w+)",
        text,
        re.IGNORECASE,
    )
    match_key = key_match.group("match").upper() if key_match else "W"
    nonmatch_key = key_match.group("nonmatch").upper() if key_match else "D"
    block_index = -1
    n_back = 1
    letters: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        block_match = re.match(r"Block\s+(\d+),\s*N\s*=\s*(\d+):", line, re.IGNORECASE)
        if block_match:
            block_index = int(block_match.group(1))
            n_back = int(block_match.group(2))
            letters = []
            continue
        letter_match = re.match(
            r"You see the letter (?P<letter>\w)(?: and press <<(?P<action>[^<>]+)>>)?\.",
            line,
            re.IGNORECASE,
        )
        if not letter_match:
            continue
        letter = letter_match.group("letter").upper()
        action = letter_match.group("action")
        if action is not None:
            correct_action = match_key if len(letters) >= n_back and letter == letters[-n_back] else nonmatch_key
            steps.append(
                StepExpectation(
                    action=action.strip(),
                    expected_reward=1.0 if action.strip().upper() == correct_action else 0.0,
                )
            )
        letters.append(letter)
    return steps


def _collsi_exp1_expectations(text: str) -> List[StepExpectation]:
    steps: List[StepExpectation] = []
    trial_re = re.compile(
        r"Progladine:\s*.*?\.\s*Amalydine:\s*.*?\.\s*"
        r"You say that the Caldionine concentration is <<(?P<action>[^<>]+)>>"
        r"(?:\.\s*That is (?P<correctness>correct|incorrect)\.\s*The correct concentration "
        r"of Caldionine is(?: indeed)? .*?)?\.",
        re.IGNORECASE,
    )
    for match in trial_re.finditer(text):
        correctness = match.group("correctness")
        if correctness is None:
            steps.append(StepExpectation(action=match.group("action").strip(), expected_reward=0.0))
        else:
            steps.append(
                StepExpectation(
                    action=match.group("action").strip(),
                    expected_reward=1.0 if correctness.lower() == "correct" else 0.0,
                )
            )
    return steps


def _frey_risk_expectations(text: str) -> List[StepExpectation]:
    steps: List[StepExpectation] = []
    balloon_re = re.compile(
        r"You press (?P<actions>(?:<<[A-Z]>>\s*)+)\.\s*(?P<feedback>The balloon was inflated too much and explodes|You stop inflating the balloon and get (?P<points>\d+) points)\.",
        re.IGNORECASE,
    )
    for match in balloon_re.finditer(text):
        actions = [action.upper() for action in re.findall(r"<<([^<>]+)>>", match.group("actions"))]
        for index, action in enumerate(actions):
            reward = 0.0
            if index == len(actions) - 1 and match.group("points") is not None:
                reward = float(match.group("points"))
            steps.append(StepExpectation(action=action, expected_reward=reward))
    return steps


def _garcia_expectations(text: str) -> List[StepExpectation]:
    trial_re = re.compile(
        r"You can choose between option \w+ and option \w+\.\s*"
        r"You press <<(?P<action>[^<>]+)>> and get (?P<reward>-?\d+(?:\.\d+)?) points\.\s*"
        r"You would have gotten -?\d+(?:\.\d+)? points had you chosen option \w+ instead\.",
        re.IGNORECASE,
    )
    return [
        StepExpectation(action=match.group("action").strip(), expected_reward=float(match.group("reward")))
        for match in trial_re.finditer(text)
    ]


def _kool_exp1_expectations(text: str) -> List[StepExpectation]:
    trial_re = re.compile(
        r"You are presented with spaceships \w+ and \w+\.\s*"
        r"You press <<(?P<action>[^<>]+)>>\.\s*You end up on planet \w+\.\s*"
        r"You find (?:(?P<amount>\d+) pieces of (?P<kind>space treasure|antimatter)|nothing)\.",
        re.IGNORECASE,
    )
    steps: List[StepExpectation] = []
    for match in trial_re.finditer(text):
        amount = float(match.group("amount") or 0)
        if (match.group("kind") or "").lower() == "antimatter":
            amount = -amount
        steps.append(StepExpectation(action=match.group("action").strip(), expected_reward=amount))
    return steps


def _kool_exp2_expectations(text: str) -> List[StepExpectation]:
    trial_re = re.compile(
        r"You are presented with spaceships \w+ and \w+\.\s*"
        r"You press <<(?P<ship>[^<>]+)>>\.\s*You end up on planet \w+\.\s*"
        r"You see alien \w+ and alien \w+\.\s*"
        r"You press <<(?P<alien>[^<>]+)>>\.\s*You find (?P<reward>\d+) pieces of space treasure\.",
        re.IGNORECASE,
    )
    steps: List[StepExpectation] = []
    for match in trial_re.finditer(text):
        steps.append(StepExpectation(action=match.group("ship").strip(), expected_reward=0.0))
        steps.append(
            StepExpectation(action=match.group("alien").strip(), expected_reward=float(match.group("reward")))
        )
    return steps


def _krueger_expectations(text: str) -> List[StepExpectation]:
    event_re = re.compile(
        r"You press <<(?P<gamble>[^<>]+)>> and then type <<(?P<follow>[^<>]+)>>\.\s*"
        r"(?:(?:The payoff for this combination would be -?\d+ points)|"
        r"(?:A \w+ ball is chosen, and you earn (?P<earn>-?\d+) points))\.",
        re.IGNORECASE,
    )
    steps: List[StepExpectation] = []
    for match in event_re.finditer(text):
        steps.append(StepExpectation(action=match.group("gamble").strip(), expected_reward=0.0))
        if match.group("earn") is None:
            steps.append(StepExpectation(action=match.group("follow").strip(), expected_reward=-4.0))
        else:
            steps.append(
                StepExpectation(action=match.group("follow").strip(), expected_reward=float(match.group("earn")))
            )
    return steps


def _ludwig_expectations(text: str) -> List[StepExpectation]:
    event_re = re.compile(
        r"You press <<(?P<action>[^<>]+)>> and find .*? which has the vitamins "
        r"\[[^\]]+\]\.\s*You get (?P<reward>-?\d+) points\.",
        re.IGNORECASE,
    )
    return [
        StepExpectation(action=match.group("action").strip(), expected_reward=float(match.group("reward")))
        for match in event_re.finditer(text)
    ]


def _wu_bandit_expectations(text: str) -> List[StepExpectation]:
    return [
        StepExpectation(action=match.group("action").strip(), expected_reward=float(match.group("reward")))
        for match in re.finditer(
            r"You press <<(?P<action>\d+)>> and receive (?P<reward>-?\d+) points\.",
            text,
            re.IGNORECASE,
        )
    ]


def _xiong_expectations(text: str) -> List[StepExpectation]:
    return [
        StepExpectation(action=match.group("action").strip(), expected_reward=float(match.group("reward")))
        for match in re.finditer(
            r"You press <<(?P<action>[^<>]+)>> and get (?P<reward>-?\d+) points\.",
            text,
            re.IGNORECASE,
        )
    ]


def _zorowitz_expectations(text: str) -> List[StepExpectation]:
    trial_re = re.compile(
        r"You are presented with two spaceships called \w+ and \w+\.\s*"
        r"You press <<(?P<ship>[^<>]+)>>\.\s*You end up on the \w+ planet\.\s*"
        r"You see a \w+ alien named \w+ and a \w+ alien named \w+\.\s*"
        r"You press <<(?P<alien>[^<>]+)>>\.\s*You find (?P<outcome>treasure|junk)\.",
        re.IGNORECASE,
    )
    steps: List[StepExpectation] = []
    for match in trial_re.finditer(text):
        steps.append(StepExpectation(action=match.group("ship").strip(), expected_reward=0.0))
        steps.append(
            StepExpectation(
                action=match.group("alien").strip(),
                expected_reward=1.0 if match.group("outcome").lower() == "treasure" else 0.0,
            )
        )
    return steps


EXPECTATION_BUILDERS = _build_expectation_registry()


def make_env_for_transcript_audit(experiment_id: str, text: str) -> Any:
    """Generative audit env with schedule frozen from *text*."""
    return make_generative_env_from_transcript(experiment_id, text)


def supported_task_env_experiments() -> Tuple[str, ...]:
    return tuple(sorted(EXPECTATION_BUILDERS.keys()))


def audit_transcript_human_path(
    experiment_id: str, text: str, session_index: int = 0
) -> SessionAuditResult:
    if experiment_id not in EXPECTATION_BUILDERS:
        return SessionAuditResult(
            experiment_id=experiment_id,
            session_index=session_index,
            n_steps=0,
            ok=False,
            issues=["No TaskEnv human-path auditor registered for this experiment."],
        )
    issues: List[str] = []
    try:
        expectations = EXPECTATION_BUILDERS[experiment_id](text)
        env = make_env_for_transcript_audit(experiment_id, text)
    except (TranscriptParseError, EnvironmentNotReadyError, KeyError, ValueError) as error:
        return SessionAuditResult(
            experiment_id=experiment_id,
            session_index=session_index,
            n_steps=0,
            ok=False,
            issues=["Setup failed: {!r}".format(error)],
        )
    except Exception as error:
        return SessionAuditResult(
            experiment_id=experiment_id,
            session_index=session_index,
            n_steps=0,
            ok=False,
            issues=["Setup failed: {!r}".format(error)],
        )

    env.reset()
    terminated = False
    for step_idx, expected in enumerate(expectations):
        observation, reward, terminated, truncated, info = env.step(expected.action)
        if truncated:
            issues.append(
                "Step {} action <<{}>> truncated: {}".format(
                    step_idx, expected.action, info.get("unsupported_counterfactual", info)
                )
            )
            break
        if expected.expected_reward is not None:
            if abs(reward - expected.expected_reward) > expected.reward_tolerance:
                issues.append(
                    "Step {} reward mismatch: env={} expected={} action={}".format(
                        step_idx, reward, expected.expected_reward, expected.action
                    )
                )
        lowered = observation.lower()
        for fragment in expected.observation_must_contain:
            if fragment.lower() not in lowered:
                issues.append(
                    "Step {} observation missing {!r} (action={})".format(
                        step_idx, fragment, expected.action
                    )
                )
        for fragment in expected.observation_must_not_contain:
            if fragment.lower() in lowered:
                issues.append(
                    "Step {} observation should not contain {!r} (action={})".format(
                        step_idx, fragment, expected.action
                    )
                )
        if terminated and step_idx < len(expectations) - 1:
            issues.append("Step {} ended episode early".format(step_idx))
            break

    if not issues and len(expectations) > 0:
        if not terminated:
            issues.append("Human path finished but env not terminated.")
    return SessionAuditResult(
        experiment_id=experiment_id,
        session_index=session_index,
        n_steps=len(expectations),
        ok=not issues,
        issues=issues,
    )


def audit_jsonl(
    jsonl_path: Path = DEFAULT_JSONL,
    experiment_filter: Optional[Iterable[str]] = None,
    max_sessions_per_experiment: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    progress_stream: Optional[TextIO] = None,
) -> Dict[str, Any]:
    allowed = set(experiment_filter) if experiment_filter else None
    progress_stream = progress_stream if progress_stream is not None else sys.stderr
    fingerprint = _jsonl_fingerprint(jsonl_path)
    per_experiment: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []
    last_line_index = -1
    audited_sessions = 0
    expected_audited_sessions: Optional[int] = None
    started_at = _utc_now()

    if checkpoint_path is not None and resume and checkpoint_path.exists():
        loaded = _load_checkpoint(checkpoint_path)
        _validate_checkpoint(
            loaded,
            fingerprint=fingerprint,
            allowed=allowed,
            max_sessions_per_experiment=max_sessions_per_experiment,
        )
        per_experiment = loaded["per_experiment"]
        failures = loaded.get("failures_sample", [])
        last_line_index = int(loaded["last_line_index"])
        audited_sessions = int(loaded["audited_sessions"])
        expected_audited_sessions = loaded.get("expected_audited_sessions")
        started_at = loaded.get("started_at", started_at)
        _emit_progress(
            progress_stream,
            line_index=last_line_index,
            audited_sessions=audited_sessions,
            expected_audited_sessions=expected_audited_sessions,
            per_experiment=per_experiment,
            experiment_id="(resumed)",
            started_at=started_at,
            resumed=True,
        )

    if expected_audited_sessions is None:
        expected_audited_sessions = _count_supported_sessions(
            jsonl_path, allowed, max_sessions_per_experiment
        )
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                allowed=allowed,
                max_sessions_per_experiment=max_sessions_per_experiment,
                last_line_index=last_line_index,
                audited_sessions=audited_sessions,
                expected_audited_sessions=expected_audited_sessions,
                per_experiment=per_experiment,
                failures=failures,
                started_at=started_at,
            )

    loop_started = time.monotonic()
    completed = False
    try:
        with jsonl_path.open(encoding="utf-8") as handle:
            for session_index, line in enumerate(handle):
                if session_index <= last_line_index:
                    continue
                if not line.strip():
                    last_line_index = session_index
                    continue
                row = json.loads(line)
                experiment_id = row["experiment"]
                audited_this_line = False
                if allowed is None or experiment_id in allowed:
                    if experiment_id in EXPECTATION_BUILDERS:
                        stats = per_experiment.setdefault(
                            experiment_id,
                            {
                                "sessions": 0,
                                "ok": 0,
                                "failed": 0,
                                "skipped": 0,
                                "skipped_setup": 0,
                            },
                        )
                        if (
                            max_sessions_per_experiment is None
                            or stats["sessions"] < max_sessions_per_experiment
                        ):
                            stats["sessions"] += 1
                            audited_sessions += 1
                            audited_this_line = True
                            result = audit_transcript_human_path(
                                experiment_id, row["text"], session_index
                            )
                            if result.skipped:
                                stats["skipped"] += 1
                            elif result.ok:
                                stats["ok"] += 1
                            else:
                                stats["failed"] += 1
                                if len(failures) < 200:
                                    failures.append(
                                        {
                                            "experiment_id": experiment_id,
                                            "session_index": session_index,
                                            "issues": result.issues,
                                        }
                                    )
                                if result.n_steps == 0:
                                    stats["skipped_setup"] += 1
                last_line_index = session_index
                if audited_this_line and progress_every > 0 and audited_sessions % progress_every == 0:
                    _emit_progress(
                        progress_stream,
                        line_index=last_line_index,
                        audited_sessions=audited_sessions,
                        expected_audited_sessions=expected_audited_sessions,
                        per_experiment=per_experiment,
                        experiment_id=experiment_id,
                        started_at=started_at,
                        loop_started=loop_started,
                    )
                    if checkpoint_path is not None:
                        _save_checkpoint(
                            checkpoint_path,
                            fingerprint=fingerprint,
                            allowed=allowed,
                            max_sessions_per_experiment=max_sessions_per_experiment,
                            last_line_index=last_line_index,
                            audited_sessions=audited_sessions,
                            expected_audited_sessions=expected_audited_sessions,
                            per_experiment=per_experiment,
                            failures=failures,
                            started_at=started_at,
                        )
        completed = True
    finally:
        if checkpoint_path is not None and not completed:
            _save_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                allowed=allowed,
                max_sessions_per_experiment=max_sessions_per_experiment,
                last_line_index=last_line_index,
                audited_sessions=audited_sessions,
                expected_audited_sessions=expected_audited_sessions,
                per_experiment=per_experiment,
                failures=failures,
                started_at=started_at,
            )
            _emit_progress(
                progress_stream,
                line_index=last_line_index,
                audited_sessions=audited_sessions,
                expected_audited_sessions=expected_audited_sessions,
                per_experiment=per_experiment,
                experiment_id="(interrupted)",
                started_at=started_at,
                loop_started=loop_started,
                interrupted=True,
            )

    if checkpoint_path is not None and completed and checkpoint_path.exists():
        checkpoint_path.unlink()

    _emit_progress(
        progress_stream,
        line_index=last_line_index,
        audited_sessions=audited_sessions,
        expected_audited_sessions=expected_audited_sessions,
        per_experiment=per_experiment,
        experiment_id="(done)",
        started_at=started_at,
        loop_started=loop_started,
        done=True,
    )

    return {
        "jsonl_path": str(jsonl_path),
        "supported_experiments": len(EXPECTATION_BUILDERS),
        "per_experiment": per_experiment,
        "failures_sample": failures,
        "audited_sessions": audited_sessions,
    }


def default_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonl_fingerprint(jsonl_path: Path) -> Dict[str, Any]:
    resolved = jsonl_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError("JSONL not found: {!r}".format(resolved))
    stat = resolved.stat()
    return {"jsonl_path": str(resolved), "jsonl_size": stat.st_size}


def _count_supported_sessions(
    jsonl_path: Path,
    allowed: Optional[set],
    max_sessions_per_experiment: Optional[int],
) -> int:
    per_experiment: Dict[str, int] = {}
    total = 0
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            experiment_id = row["experiment"]
            if allowed is not None and experiment_id not in allowed:
                continue
            if experiment_id not in EXPECTATION_BUILDERS:
                continue
            count = per_experiment.get(experiment_id, 0)
            if max_sessions_per_experiment is not None and count >= max_sessions_per_experiment:
                continue
            per_experiment[experiment_id] = count + 1
            total += 1
    return total


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    with checkpoint_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_checkpoint(
    loaded: Dict[str, Any],
    fingerprint: Dict[str, Any],
    allowed: Optional[set],
    max_sessions_per_experiment: Optional[int],
) -> None:
    if loaded.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Checkpoint version mismatch (got {!r}, expected {}).".format(
                loaded.get("version"), CHECKPOINT_VERSION
            )
        )
    if loaded.get("jsonl_path") != fingerprint["jsonl_path"]:
        raise ValueError("Checkpoint JSONL path does not match current --jsonl.")
    if loaded.get("jsonl_size") != fingerprint["jsonl_size"]:
        raise ValueError("Checkpoint JSONL size changed; delete checkpoint and rerun.")
    if loaded.get("max_sessions_per_experiment") != max_sessions_per_experiment:
        raise ValueError("Checkpoint --max-per-experiment does not match.")
    saved_filter = loaded.get("experiment_filter")
    current_filter = sorted(allowed) if allowed is not None else None
    if saved_filter != current_filter:
        raise ValueError("Checkpoint experiment filter does not match.")


def _save_checkpoint(
    checkpoint_path: Path,
    fingerprint: Dict[str, Any],
    allowed: Optional[set],
    max_sessions_per_experiment: Optional[int],
    last_line_index: int,
    audited_sessions: int,
    expected_audited_sessions: Optional[int],
    per_experiment: Dict[str, Dict[str, int]],
    failures: List[Dict[str, Any]],
    started_at: str,
) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "jsonl_path": fingerprint["jsonl_path"],
        "jsonl_size": fingerprint["jsonl_size"],
        "experiment_filter": sorted(allowed) if allowed is not None else None,
        "max_sessions_per_experiment": max_sessions_per_experiment,
        "last_line_index": last_line_index,
        "audited_sessions": audited_sessions,
        "expected_audited_sessions": expected_audited_sessions,
        "per_experiment": per_experiment,
        "failures_sample": failures,
        "started_at": started_at,
        "updated_at": _utc_now(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(checkpoint_path)


def _emit_progress(
    stream: TextIO,
    line_index: int,
    audited_sessions: int,
    expected_audited_sessions: Optional[int],
    per_experiment: Dict[str, Dict[str, int]],
    experiment_id: str,
    started_at: str,
    loop_started: Optional[float] = None,
    resumed: bool = False,
    interrupted: bool = False,
    done: bool = False,
) -> None:
    total_ok = sum(row.get("ok", 0) for row in per_experiment.values())
    total_failed = sum(row.get("failed", 0) for row in per_experiment.values())
    parts = ["[transcript-audit]"]
    if done:
        parts.append("done")
    elif interrupted:
        parts.append("interrupted")
    elif resumed:
        parts.append("resumed")
    if expected_audited_sessions:
        pct = 100.0 * audited_sessions / expected_audited_sessions
        parts.append("{}/{} sessions ({:.1f}%)".format(audited_sessions, expected_audited_sessions, pct))
    else:
        parts.append("{} sessions".format(audited_sessions))
    parts.append("line={}".format(line_index))
    parts.append("exp={}".format(experiment_id))
    parts.append("ok={} fail={}".format(total_ok, total_failed))
    if loop_started is not None:
        elapsed = time.monotonic() - loop_started
        parts.append("elapsed={:.0f}s".format(elapsed))
        if expected_audited_sessions and audited_sessions > 0 and not done:
            remaining = elapsed * (expected_audited_sessions - audited_sessions) / audited_sessions
            parts.append("eta~{:.0f}s".format(remaining))
    print(" ".join(parts), file=stream, flush=True)
