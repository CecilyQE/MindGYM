"""Build generative audit envs with episode schedule frozen from a Psych-101 transcript."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from psycenvir.core.generative_registry import (
    _GENERATIVE_BUILDERS,
    apply_generative_defaults,
    list_generative_experiments,
)
from psycenvir.core.registry import (
    BADHAM_ID,
    BAHRAMI_ID,
    COLLSI_EXP3_ID,
    COX_PAIR_EXP1_ID,
    ENKAVI_DIGIT_SPAN_EXP1_ID,
    ENKAVI_GONOGO_EXP1_ID,
    ENKAVI_RECENT_PROBES_ID,
    FLESCH_TREE_EXP1_ID,
    FREY_CCT_ID,
    GERSHMAN_DECONSTRUCT_EXP2_ID,
    GERSHMAN_MAPPING_ID,
    HILBIG_ID,
    KOOL_COST_EXP1_ID,
    KOOL_COST_EXP2_ID,
    LEFEBVRE_EXP1_ID,
    LEFEBVRE_EXP2_ID,
    PETERSON_ID,
    PLONSKY_ID,
    SCHULZ_FINDING_EXPERIMENT_IDS,
    SPEEKENBRINK_WEATHER_ID,
    STEINGROEVER_IGT_EXP1_ID,
    STEINGROEVER_IGT_EXP3_ID,
    TOMOV_CASTLE_EXPERIMENT_IDS,
    TOMOV_SUBWAY_EXPERIMENT_IDS,
    WILSON_EXP1_ID,
    WULFF_DESCRIPTION_ID,
    WULFF_SAMPLING_ID,
)
from psycenvir.errors import EnvironmentNotReadyError
from psycenvir.generative.adaptive_nback import (
    ENKAVI_ADAPTIVE_NBACK_EXP1_ID,
    _NBackTrial,
)
from psycenvir.generative.balloon import FREY_RISK_EXPERIMENT_ID
from psycenvir.generative.chunking import WU_CHUNKING_EXP1_ID, WU_CHUNKING_EXP2_ID, _ChunkingTrial
from psycenvir.generative.collsi_judgment import COLLSI_EXP1_ID
from psycenvir.generative.flesch_tree import FleschTreeGenerativeEnv, _FleschTrial
from psycenvir.generative.garcia_experiential import GARCIA_EXPERIENTIAL_IDS, _GarciaTrial
from psycenvir.generative.grounding import GroundedGenerativeEnv, get_grounding_profile
from psycenvir.generative.hazard_bandit import XIONG_NEURAL_EXP1_ID, _HazardTrial
from psycenvir.generative.kool_spaceship import KOOL_WHEN_EXP1_ID, _KoolDay
from psycenvir.generative.kool_two_step import KOOL_WHEN_EXP2_ID, _KoolTwoStepDay, _SpaceshipPhase
from psycenvir.generative.krueger_identifying import KRUEGER_IDENTIFYING_EXP1_ID, _KruegerRound
from psycenvir.generative.ludwig_fruit import LUDWIG_HUMAN_EXPERIMENT_IDS
from psycenvir.generative.peterson import PetersonGenerativeEnv
from psycenvir.generative.plonsky_gamble import PlonskyGambleGenerativeEnv, _PlonskyTrial
from psycenvir.generative.ruggeri_choice import RUGGERI_GLOBALIZABILITY_EXP1_ID, _RuggeriChoice
from psycenvir.generative.schulz_finding import _SchulzTrial
from psycenvir.core.base import render_initial_observation
from psycenvir.generative.steingroever_igt import STEINGROEVER_IGT_EXP2_ID
from psycenvir.generative.two_arm_slot import WILSON_EXPERIMENT_CONFIGS, _SlotTrial
from psycenvir.generative.volatile_bandit import GERSHMAN_DECONSTRUCT_EXP1_ID, _VolatileTrial
from psycenvir.generative.transcript_bound import TranscriptBoundAuditEnv
from psycenvir.generative.wu_bandit import WU_EXPERIMENT_ID
from psycenvir.generative.zorowitz_space import ZOROWITZ_DATA_EXP1_ID, _ZorowitzTrial
from psycenvir.models import FleschTreeTrial, PlonskyGambleTrial
from psycenvir.psych101.parse import (
    BADHAM_TRIAL_RE,
    BAHRAMI_TRIAL_RE,
    COLLSI_TRIAL_RE,
    COX_PAIR_TEST_TRIAL_RE,
    DIGIT_SPAN_BLOCK_RE,
    GONOGO_TRIAL_RE,
    HILBIG_TRIAL_RE,
    FLESCH_TREE_TRIAL_RE,
    FREY_CCT_ROUND_RE,
    GERSHMAN_TRIAL_RE,
    HILBIG_TRIAL_RE,
    KOOL_COST_EXP1_DAY_RE,
    KOOL_COST_EXP2_DAY_RE,
    LEFEBVRE_TRIAL_RE,
    PETERSON_BLOCK_RE,
    PLONSKY_OPTION_BLOCK_RE,
    SCHULZ_TRIAL_RE,
    SPEEKENBRINK_TRIAL_RE,
    STEINGROEVER_IGT_TRIAL_RE,
    TOMOV_CASTLE_STEP_RE,
    TOMOV_STATION_STEP_RE,
    WULFF_LOTTERY_PRESS_RE,
    parse_badham_category_trials,
    parse_bahrami_four_arm_trials,
    parse_collsi_judgment_trials,
    parse_collsi_response_actions,
    parse_cox_pair_recognition_trials,
    parse_digit_span_recall_trials,
    parse_flesch_tree_trials,
    parse_frey_cct_rounds,
    parse_gershman_bandit_trials,
    parse_gershman_mapping_trials,
    parse_gershman_response_actions,
    parse_gonogo_trials,
    parse_hilbig_product_trials,
    parse_instruction_prefix,
    parse_kool_cost_exp1_days,
    parse_kool_cost_exp2_days,
    parse_lefebvre_casino_trials,
    parse_peterson_gamble_blocks,
    parse_plonsky_gamble_trials,
    parse_recent_probe_actions,
    parse_recent_probe_trials,
    parse_schulz_finding_trials,
    parse_speekenbrink_weather_actions,
    parse_speekenbrink_weather_trials,
    parse_steingroever_igt_trials,
    parse_tomov_castle_trials,
    parse_tomov_subway_trials,
    parse_wilson_slot_trials,
    parse_wulff_description_trials,
    parse_wulff_sampling_problems,
)
from psycenvir.sim.category import BadhamCategoryEnv
from psycenvir.sim.cct import FreyCCTRecordedPathEnv
from psycenvir.sim.cox_pair import CoxPairRecognitionRecordedEnv
from psycenvir.sim.digit_span import EnkaviDigitSpanRecordedEnv
from psycenvir.sim.gonogo import EnkaviGonogoRecordedEnv
from psycenvir.sim.hilbig import HilbigProductRecordedEnv
from psycenvir.sim.kool_cost_exp1 import KoolCostExp1RecordedEnv
from psycenvir.sim.kool_cost_exp2 import KoolCostExp2RecordedEnv
from psycenvir.sim.mapping import GershmanMappingEnv
from psycenvir.sim.steingroever_igt import SteingroeverIGTRecordedEnv
from psycenvir.sim.tomov_castle import TomovCastleRecordedEnv
from psycenvir.sim.tomov_subway import TomovSubwayRecordedEnv
from psycenvir.sim.wulff_lottery import WulffDescriptionRecordedEnv
from psycenvir.sim.wulff_sampling import WulffSamplingRecordedEnv

ResetFn = Callable[[Optional[int]], Tuple[str, Dict[str, Any]]]
LoaderFn = Callable[[Any, str], Tuple[str, Dict[str, Any]]]


def _flesch_trial_from_parsed(trial: FleschTreeTrial) -> _FleschTrial:
    assert trial.outcomes_by_action is not None
    return _FleschTrial(
        observation=trial.observation,
        valid_actions=trial.valid_actions,
        outcomes_by_action=dict(trial.outcomes_by_action),
        has_feedback=trial.has_feedback,
    )


def _plonsky_trial_from_parsed(trial: PlonskyGambleTrial) -> _PlonskyTrial:
    if trial.outcomes_by_action is not None:
        outcomes = dict(trial.outcomes_by_action)
    else:
        outcomes = {trial.valid_actions[0]: 0.0, trial.valid_actions[1]: 0.0}
    return _PlonskyTrial(
        observation=trial.observation,
        valid_actions=trial.valid_actions,
        outcomes_by_action=outcomes,
        has_feedback=trial.has_feedback,
    )


def _patch_reset(env: Any, loader: LoaderFn, text: str) -> None:
    def reset(seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        return loader(env, text)

    env.reset = reset  # type: ignore[method-assign]


def _wrap_generative(env: Any, experiment_id: str) -> GroundedGenerativeEnv:
    profile = get_grounding_profile(experiment_id)
    return GroundedGenerativeEnv(env, profile)


def _make_generative_instance(experiment_id: str, include_human_ref: bool) -> Any:
    builder = _GENERATIVE_BUILDERS.get(experiment_id)
    if builder is None:
        raise EnvironmentNotReadyError(
            "No generative simulator registered for {!r}.".format(experiment_id)
        )
    profile = get_grounding_profile(experiment_id)
    config = apply_generative_defaults(experiment_id, {}, profile)
    if include_human_ref:
        config["include_human_ref"] = True
    return builder(**config)


# --- loaders that populate generative env state from parse ---


def _initial_observation(env: Any, body: str) -> Tuple[str, Dict[str, Any]]:
    return render_initial_observation(env.instruction, body), env._info(None, None)


def _load_schulz(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    parsed = parse_schulz_finding_trials(text)
    env._trials = [
        _SchulzTrial(
            t.round_number,
            t.valid_actions,
            dict(t.outcomes_by_action),
            t.show_round_header,
        )
        for t in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    trial = env._trials[0]
    return render_initial_observation(env.instruction, env._render_trial(trial)), env._info(
        trial, None
    )


def _load_wilson(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    parsed = parse_wilson_slot_trials(text)
    trials: List[_SlotTrial] = []
    for trial in parsed:
        if trial.trial_type == "instructed":
            arm = trial.valid_actions[0]
            trials.append(
                _SlotTrial(
                    observation=trial.observation,
                    valid_actions=(arm,),
                    outcomes_by_action={arm: int(trial.outcomes_by_action[arm])},
                    instructed=True,
                )
            )
        else:
            trials.append(
                _SlotTrial(
                    observation=trial.observation,
                    valid_actions=trial.valid_actions,
                    outcomes_by_action={
                        key: int(value) for key, value in trial.outcomes_by_action.items()
                    },
                )
            )
    env._trials = trials
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return _initial_observation(env, env._trials[0].observation)


def _load_recent_probes(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.recent_probes import _ProbeTrial

    actions = parse_recent_probe_actions(text)
    env.present_key = actions[0]
    env.absent_key = actions[1]
    parsed = parse_recent_probe_trials(text)
    env._trials = [
        _ProbeTrial(
            letters=trial.letters,
            probe=trial.probe,
            valid_actions=(env.present_key, env.absent_key),
            correct_action=trial.correct_action,
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._n_correct = 0
    env._done = False
    trial = env._trials[0]
    return (
        render_initial_observation(env.instruction, env._render_stimulus(trial)),
        env._info(None, None),
    )


def _load_collsi(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.collsi_judgment import _JudgmentTrial

    valid_actions = tuple(parse_collsi_response_actions(text))
    if env.experiment_id == COLLSI_EXP1_ID:
        parsed = []
        trial_re = re.compile(
            r"Progladine:\s*(?P<progladine>.*?)\.\s*Amalydine:\s*(?P<amalydine>.*?)\.\s*"
            r"You say that the Caldionine concentration is <<(?P<human_action>[^<>]+)>>"
            r"(?:\.\s*That is (?P<correctness>correct|incorrect)\.\s*The correct concentration "
            r"of Caldionine is(?: indeed)? (?P<correct_action>.*?))?\.",
            re.IGNORECASE,
        )
        for match in trial_re.finditer(text):
            human_action = match.group("human_action").strip()
            correct_action = match.group("correct_action")
            parsed.append(
                type(
                    "_ParsedCollsi",
                    (),
                    {
                        "progladine": match.group("progladine").strip(),
                        "amalydine": match.group("amalydine").strip(),
                        "correct_action": (correct_action.strip() if correct_action else human_action),
                        "human_action": human_action,
                        "has_feedback": correct_action is not None,
                    },
                )()
            )
        if not parsed:
            raise EnvironmentNotReadyError("No Collsi exp1 trials found.")
    else:
        parsed = parse_collsi_judgment_trials(text)
    env._trials = [
        _JudgmentTrial(
            progladine=trial.progladine,
            amalydine=trial.amalydine,
            correct_action=trial.correct_action,
            valid_actions=valid_actions,
            has_feedback=getattr(trial, "has_feedback", True),
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return (
        render_initial_observation(env.instruction, env._render_stimulus(env._trials[0])),
        env._info(None, None),
    )


def _load_frey_risk(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    balloon_re = re.compile(
        r"You press (?P<actions>(?:<<[A-Z]>>\s*)+)\.\s*(?P<feedback>The balloon was inflated too much and explodes|You stop inflating the balloon and get (?P<points>\d+) points)\.",
        re.IGNORECASE,
    )
    thresholds: List[int] = []
    pump_candidates: List[str] = []
    collect_candidates: List[str] = []
    for match in balloon_re.finditer(text):
        actions = re.findall(r"<<([^<>]+)>>", match.group("actions"))
        upper_actions = [action.upper() for action in actions]
        if match.group("points") is None:
            pump_candidates.extend(upper_actions)
        else:
            collect_candidates.append(upper_actions[-1])
            pump_candidates.extend(upper_actions[:-1])
        if pump_candidates:
            env.pump_key = max(set(pump_candidates), key=pump_candidates.count)
        if collect_candidates:
            env.collect_key = max(set(collect_candidates), key=collect_candidates.count)
        pumps = sum(1 for action in upper_actions if action == env.pump_key)
        if match.group("points") is None:
            thresholds.append(max(1, pumps))
        else:
            thresholds.append(pumps + 1)
    if not thresholds:
        raise EnvironmentNotReadyError("No Frey risk balloons found.")
    env.n_balloons = len(thresholds)
    env._thresholds = thresholds
    env._balloon_idx = 0
    env._pumps = 0
    env._accumulated = 0
    env._total_points = 0.0
    env._done = False
    return render_initial_observation(env.instruction, env._balloon_header()), env._info()


def _load_garcia(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    trial_re = re.compile(
        r"You can choose between option (?P<a>\w+) and option (?P<b>\w+)\.\s*"
        r"You press <<(?P<action>[^<>]+)>> and get (?P<reward>-?\d+(?:\.\d+)?) points\.\s*"
        r"You would have gotten (?P<alt_reward>-?\d+(?:\.\d+)?) points had you chosen option (?P<alt>\w+) instead\.",
        re.IGNORECASE,
    )
    trials: List[_GarciaTrial] = []
    for match in trial_re.finditer(text):
        a, b = match.group("a"), match.group("b")
        action = match.group("action").strip()
        alt = match.group("alt").strip()
        observation = "You can choose between option {} and option {}.".format(a, b)
        trials.append(
            _GarciaTrial(
                observation=observation,
                option_a=a,
                option_b=b,
                valid_actions=(a, b),
                outcomes_by_action={
                    action: float(match.group("reward")),
                    alt: float(match.group("alt_reward")),
                },
                part_number=1,
            )
        )
    if not trials:
        raise EnvironmentNotReadyError("No Garcia trials found.")
    env._trials = trials
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return render_initial_observation(env.instruction, trials[0].observation), env._info(None, None)


def _load_gershman_volatile(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    parsed = parse_gershman_bandit_trials(text)
    env._trials = [
        _VolatileTrial(
            game_number=trial.game_number,
            valid_actions=trial.valid_actions,
            outcomes_by_action=dict(trial.outcomes_by_action),
            show_game_header=trial.show_game_header,
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return render_initial_observation(env.instruction, env._render_trial(env._trials[0])), env._info(
        env._trials[0], None
    )


def _load_kool_exp1(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    trial_re = re.compile(
        r"You are presented with spaceships (?P<s1>\w+) and (?P<s2>\w+)\.\s*"
        r"You press <<(?P<action>[^<>]+)>>\.\s*You end up on planet (?P<planet>\w+)\.\s*"
        r"You find (?:(?P<amount>\d+) pieces of (?P<kind>space treasure|antimatter)|nothing)\.",
        re.IGNORECASE,
    )
    days: List[_KoolDay] = []
    for match in trial_re.finditer(text):
        action = match.group("action").strip()
        amount = int(match.group("amount") or 0)
        kind = "antimatter" if (match.group("kind") or "").lower() == "antimatter" else "treasure"
        days.append(
            _KoolDay(
                pair_key="transcript",
                ships=(match.group("s1"), match.group("s2")),
                planet_map={action: match.group("planet")},
                outcomes_by_action={action: (kind, amount)},
            )
        )
    if not days:
        raise EnvironmentNotReadyError("No Kool exp1 trials found.")
    env._days = days
    env._day_idx = 0
    env._treasure_total = 0
    env._done = False
    return render_initial_observation(env.instruction, env._render_day(days[0])), env._info(None, None)


def _load_kool_exp2(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    trial_re = re.compile(
        r"You are presented with spaceships (?P<s1>\w+) and (?P<s2>\w+)\.\s*"
        r"You press <<(?P<ship>[^<>]+)>>\.\s*You end up on planet (?P<planet>\w+)\.\s*"
        r"You see alien (?P<a1>\w+) and alien (?P<a2>\w+)\.\s*"
        r"You press <<(?P<alien>[^<>]+)>>\.\s*You find (?P<reward>\d+) pieces of space treasure\.",
        re.IGNORECASE,
    )
    events: List[Tuple[str, float, str]] = []
    for match in trial_re.finditer(text):
        ship_obs = (
            "You are presented with spaceships {} and {}. You press <<{}>>. "
            "You end up on planet {}."
        ).format(
            match.group("s1"), match.group("s2"), match.group("ship"), match.group("planet")
        )
        events.append((match.group("ship").strip().upper(), 0.0, ship_obs))
        events.append((match.group("alien").strip().upper(), float(match.group("reward")), match.group(0)))
    if not events:
        raise EnvironmentNotReadyError("No Kool exp2 trials found.")
    _install_one_step_schedule(env, events)
    return render_initial_observation(env.instruction, events[0][2]), env._info(None, None)


def _load_krueger(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    event_re = re.compile(
        r"You press <<(?P<gamble>[^<>]+)>> and then type <<(?P<follow>[^<>]+)>>\.\s*"
        r"(?:(?:The payoff for this combination would be (?P<check>-?\d+) points)|"
        r"(?:A (?P<color>\w+) ball is chosen, and you earn (?P<earn>-?\d+) points))\.",
        re.IGNORECASE,
    )
    matches = list(event_re.finditer(text))
    gambles_seen: List[str] = []
    colors_seen: List[str] = []
    for match in matches:
        gamble = match.group("gamble").strip().upper()
        follow = match.group("follow").strip().lower()
        if gamble not in gambles_seen:
            gambles_seen.append(gamble)
        if follow != "stop" and follow not in colors_seen:
            colors_seen.append(follow)
        if match.group("color") is not None:
            color = match.group("color").strip().lower()
            if color not in colors_seen:
                colors_seen.append(color)
    if gambles_seen:
        env.gambles = tuple(gambles_seen)
    if colors_seen:
        env.colors = tuple(colors_seen)
    rounds: List[_KruegerRound] = []
    payoffs: Dict[str, Dict[str, float]] = {gamble: {} for gamble in env.gambles}
    for match in matches:
        gamble = match.group("gamble").strip().upper()
        follow = match.group("follow").strip().lower()
        if match.group("check") is not None:
            payoffs.setdefault(gamble, {})[follow] = float(match.group("check"))
            continue
        color = match.group("color").strip().lower()
        payoffs.setdefault(gamble, {})[color] = float(match.group("earn"))
        rounds.append(
            _KruegerRound(
                observation="A new round begins.",
                valid_actions=env.gambles,
                outcomes_by_action={key: dict(value) for key, value in payoffs.items()},
                chosen_color=color,
            )
        )
        payoffs = {gamble_key: {} for gamble_key in env.gambles}
    if not rounds:
        raise EnvironmentNotReadyError("No Krueger rounds found.")
    env._rounds = rounds
    env._trial_idx = 0
    env._pending_gamble = None
    env._points = 0.0
    env._done = False
    return render_initial_observation(env.instruction, rounds[0].observation), env._info(None, None)


def _install_one_step_schedule(env: Any, events: List[Tuple[str, float, str]]) -> None:
    if not events:
        raise EnvironmentNotReadyError("No transcript events found.")
    env._transcript_events = events
    env._trial_idx = 0
    env._points = 0.0
    env._done = False

    def info_for(submitted: str) -> Dict[str, Any]:
        if not hasattr(env, "_info"):
            return {"selected_action": submitted}
        for args in ((None, submitted), (None,), (submitted,)):
            try:
                return env._info(*args)
            except TypeError:
                continue
        return {"selected_action": submitted}

    def step(action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if env._done:
            raise RuntimeError("Cannot step a completed transcript-bound env; call reset().")
        submitted = str(action).strip().upper()
        expected_action, reward, observation = env._transcript_events[env._trial_idx]
        if submitted != expected_action.upper():
            env._done = True
            info = info_for(submitted)
            info["invalid_action"] = submitted
            return "Invalid transcript action <<{}>>.".format(submitted), 0.0, False, True, info
        env._points = getattr(env, "_points", 0.0) + float(reward)
        env._trial_idx += 1
        env._done = env._trial_idx >= len(env._transcript_events)
        info = info_for(submitted)
        return observation, float(reward), env._done, False, info

    env.step = step  # type: ignore[method-assign]


def _load_ludwig(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    event_re = re.compile(
        r"You press <<(?P<action>[^<>]+)>> and find the (?P<fruit>.*?) which has the vitamins "
        r"\[(?P<vitamins>[^\]]+)\]\.\s*You get (?P<reward>-?\d+) points\.",
        re.IGNORECASE,
    )
    events = [
        (
            match.group("action").strip().upper(),
            float(match.group("reward")),
            match.group(0),
        )
        for match in event_re.finditer(text)
    ]
    _install_one_step_schedule(env, events)
    return render_initial_observation(env.instruction, events[0][2]), env._info(None, None)


def _load_wu_bandit(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    event_re = re.compile(
        r"You press <<(?P<action>\d+)>> and receive (?P<reward>-?\d+) points\.",
        re.IGNORECASE,
    )
    events = [
        (match.group("action").strip(), float(match.group("reward")), match.group(0))
        for match in event_re.finditer(text)
    ]
    _install_one_step_schedule(env, events)
    return render_initial_observation(env.instruction, events[0][2]), env._info(None)


def _load_xiong(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    event_re = re.compile(
        r"You press <<(?P<action>[^<>]+)>> and get (?P<reward>-?\d+) points\.",
        re.IGNORECASE,
    )
    matches = list(event_re.finditer(text))
    actions_seen: List[str] = []
    for match in matches:
        action = match.group("action").strip().upper()
        if action not in actions_seen:
            actions_seen.append(action)
    if len(actions_seen) >= 2:
        env.arms = tuple(actions_seen[:2])
    trials: List[_HazardTrial] = []
    for match in matches:
        action = match.group("action").strip().upper()
        reward = int(match.group("reward"))
        other = env.arms[1] if action == env.arms[0] else env.arms[0]
        trials.append(_HazardTrial(1, 0.0, {action: reward, other: reward}))
    if not trials:
        raise EnvironmentNotReadyError("No Xiong trials found.")
    env._trials = trials
    env.payoff_sd = 0.0
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return render_initial_observation(env.instruction, env._render_trial(trials[0])), env._info(None, None)


def _load_zorowitz(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    event_re = re.compile(
        r"You are presented with two spaceships called (?P<s1>\w+) and (?P<s2>\w+)\.\s*"
        r"You press <<(?P<ship>[^<>]+)>>\.\s*You end up on the (?P<planet>\w+) planet\.\s*"
        r"You see a (?P=planet) alien named (?P<a1>\w+) and a (?P=planet) alien named (?P<a2>\w+)\.\s*"
        r"You press <<(?P<alien>[^<>]+)>>\.\s*You find (?P<outcome>treasure|junk)\.",
        re.IGNORECASE,
    )
    events: List[Tuple[str, float, str]] = []
    for match in event_re.finditer(text):
        ship_obs = (
            "You are presented with two spaceships called {} and {}. You press <<{}>>. "
            "You end up on the {} planet."
        ).format(match.group("s1"), match.group("s2"), match.group("ship"), match.group("planet"))
        events.append((match.group("ship").strip().upper(), 0.0, ship_obs))
        reward = 1.0 if match.group("outcome").lower() == "treasure" else 0.0
        events.append((match.group("alien").strip().upper(), reward, match.group(0)))
    _install_one_step_schedule(env, events)
    return render_initial_observation(env.instruction, events[0][2]), env._info(None, None)


def _load_bahrami(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.four_arm_bandit import _FourArmTrial

    parsed = parse_bahrami_four_arm_trials(text)
    env._trials = [
        _FourArmTrial(
            valid_actions=trial.valid_actions,
            outcomes_by_action=dict(trial.outcomes_by_action),
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    labels = env._trials[0].valid_actions
    return (
        render_initial_observation(env.instruction, "You press <<{}>>.".format(labels[0])),
        env._info(env._trials[0], None),
    )


def _load_lefebvre(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.casino_bandit import _CasinoTrial

    parsed = parse_lefebvre_casino_trials(text)
    env._trials = [
        _CasinoTrial(
            casino_id=trial.casino_id,
            valid_actions=trial.valid_actions,
            outcomes_by_action=dict(trial.outcomes_by_action),
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    first = env._trials[0]
    body = "Casino {}. You press <<{}>>.".format(first.casino_id, first.valid_actions[0])
    return render_initial_observation(env.instruction, body), env._info(first, None)


def _load_speekenbrink(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.weather_cards import _WeatherTrial

    parsed = parse_speekenbrink_weather_trials(text)
    rainy_action, fine_action = parse_speekenbrink_weather_actions(text)
    env._trials = [
        _WeatherTrial(
            cards_display=trial.cards_display,
            weather=trial.weather,
            correct_action=rainy_action if trial.weather == "rainy" else fine_action,
            valid_actions=(rainy_action, fine_action),
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    return (
        render_initial_observation(env.instruction, env._render_trial(env._trials[0])),
        env._info(None, None),
    )


def _load_gershman_bandit(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    from psycenvir.generative.competitive_bandit import _CompetitiveTrial

    parsed = parse_gershman_bandit_trials(text)
    env._trials = [
        _CompetitiveTrial(
            game_number=trial.game_number,
            valid_actions=trial.valid_actions,
            outcomes_by_action=dict(trial.outcomes_by_action),
            show_game_header=trial.show_game_header,
        )
        for trial in parsed
    ]
    env._trial_idx = 0
    env._points = 0.0
    env._done = False
    first = env._trials[0]
    header = "Game {}:".format(first.game_number) if first.show_game_header else ""
    return render_initial_observation(env.instruction, header), env._info(first, None)


def _load_chunking(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    trials = [
        _ChunkingTrial(match.group("target").strip(), int(match.group("rt")))
        for match in re.finditer(
            r"The instruction is to press (?P<target>\w+), you press <<(?P<action>[^<>]+)>> in (?P<rt>\d+) ms\. That is (?P<correctness>correct|incorrect)\.",
            text,
            re.IGNORECASE,
        )
    ]
    if not trials:
        raise EnvironmentNotReadyError("No chunking trials found.")
    env._trials = trials
    env._trial_idx = 0
    env._done = False
    return (
        render_initial_observation(env.instruction, env._render_trial(env._trials[0])),
        env._info(None, None),
    )


def _load_ruggeri(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    matches = list(
        re.finditer(
            r"You have the choice between (?P<left>.*?) \(press (?P<left_key>\w+)\) or (?P<right>.*?) \(press (?P<right_key>\w+)\)\. You press <<(?P<action>[^<>]+)>>\.",
            text,
            re.IGNORECASE,
        )
    )
    choices = [_RuggeriChoice(match.group("left").strip(), match.group("right").strip()) for match in matches]
    if not choices:
        raise EnvironmentNotReadyError("No Ruggeri choices found.")
    env.action_keys = (matches[0].group("left_key").upper(), matches[0].group("right_key").upper())
    env.valid_actions = env.action_keys
    env.choices = choices
    env._trial_idx = 0
    env._done = False
    return (
        render_initial_observation(env.instruction, env._render_choice(env.choices[0])),
        env._info(None, None),
    )


def _load_adaptive_nback(env: Any, text: str) -> Tuple[str, Dict[str, Any]]:
    key_match = re.search(
        r"matches\s+the\s+letter\s+N\s+trials\s+ago,\s+press\s+(?P<match>\w+),\s+otherwise\s+press\s+(?P<nonmatch>\w+)",
        text,
        re.IGNORECASE,
    )
    if key_match:
        env.match_key = key_match.group("match").upper()
        env.nonmatch_key = key_match.group("nonmatch").upper()
        env.valid_actions = (env.match_key, env.nonmatch_key)
    trials: List[_NBackTrial] = []
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
            correct = env.match_key if len(letters) >= n_back and letter == letters[-n_back] else env.nonmatch_key
            trials.append(_NBackTrial(block_index, n_back, letter, correct))
        letters.append(letter)
    if not trials:
        raise EnvironmentNotReadyError("No adaptive n-back action trials found.")
    env._trials = trials
    env._trial_idx = 0
    env._block_errors = {trial.block_index: 0 for trial in trials}
    env._done = False
    return (
        render_initial_observation(env.instruction, env._render_trial(env._trials[0])),
        env._info(None, None),
    )


_PATCH_LOADERS: Dict[str, LoaderFn] = {}


def _register_patch_loaders() -> None:
    for experiment_id in SCHULZ_FINDING_EXPERIMENT_IDS:
        _PATCH_LOADERS[experiment_id] = _load_schulz
    for experiment_id in WILSON_EXPERIMENT_CONFIGS:
        _PATCH_LOADERS[experiment_id] = _load_wilson
    _PATCH_LOADERS[BAHRAMI_ID] = _load_bahrami
    _PATCH_LOADERS[LEFEBVRE_EXP1_ID] = _load_lefebvre
    _PATCH_LOADERS[LEFEBVRE_EXP2_ID] = _load_lefebvre
    _PATCH_LOADERS[SPEEKENBRINK_WEATHER_ID] = _load_speekenbrink
    _PATCH_LOADERS[GERSHMAN_DECONSTRUCT_EXP2_ID] = _load_gershman_bandit
    _PATCH_LOADERS[GERSHMAN_DECONSTRUCT_EXP1_ID] = _load_gershman_volatile
    _PATCH_LOADERS[ENKAVI_RECENT_PROBES_ID] = _load_recent_probes
    _PATCH_LOADERS[COLLSI_EXP1_ID] = _load_collsi
    _PATCH_LOADERS[COLLSI_EXP3_ID] = _load_collsi
    _PATCH_LOADERS[FREY_RISK_EXPERIMENT_ID] = _load_frey_risk
    for experiment_id in GARCIA_EXPERIENTIAL_IDS:
        _PATCH_LOADERS[experiment_id] = _load_garcia
    _PATCH_LOADERS[KOOL_WHEN_EXP1_ID] = _load_kool_exp1
    _PATCH_LOADERS[KOOL_WHEN_EXP2_ID] = _load_kool_exp2
    _PATCH_LOADERS[KRUEGER_IDENTIFYING_EXP1_ID] = _load_krueger
    for experiment_id in LUDWIG_HUMAN_EXPERIMENT_IDS:
        _PATCH_LOADERS[experiment_id] = _load_ludwig
    _PATCH_LOADERS[WU_EXPERIMENT_ID] = _load_wu_bandit
    _PATCH_LOADERS[XIONG_NEURAL_EXP1_ID] = _load_xiong
    _PATCH_LOADERS[ZOROWITZ_DATA_EXP1_ID] = _load_zorowitz
    _PATCH_LOADERS[ENKAVI_ADAPTIVE_NBACK_EXP1_ID] = _load_adaptive_nback
    _PATCH_LOADERS[RUGGERI_GLOBALIZABILITY_EXP1_ID] = _load_ruggeri
    _PATCH_LOADERS[WU_CHUNKING_EXP1_ID] = _load_chunking
    _PATCH_LOADERS[WU_CHUNKING_EXP2_ID] = _load_chunking


_register_patch_loaders()

# Tasks whose generative ``reset()`` resamples structure; audit uses exact ``sim`` path.
_RECORDED_TRANSCRIPT_EXPERIMENTS = frozenset(
    {
        BADHAM_ID,
        FREY_CCT_ID,
        KOOL_COST_EXP1_ID,
        KOOL_COST_EXP2_ID,
        WULFF_SAMPLING_ID,
        WULFF_DESCRIPTION_ID,
        HILBIG_ID,
        ENKAVI_DIGIT_SPAN_EXP1_ID,
        ENKAVI_GONOGO_EXP1_ID,
        COX_PAIR_EXP1_ID,
        GERSHMAN_MAPPING_ID,
        STEINGROEVER_IGT_EXP1_ID,
        STEINGROEVER_IGT_EXP2_ID,
        STEINGROEVER_IGT_EXP3_ID,
    }
    | set(TOMOV_SUBWAY_EXPERIMENT_IDS)
    | set(TOMOV_CASTLE_EXPERIMENT_IDS)
)


def _recorded_exact_env(experiment_id: str, text: str, include_human_ref: bool) -> Any:
    """Exact Psych-101 path via sim env; wrapped for generative audit metadata."""
    if experiment_id == BADHAM_ID:
        inner = BadhamCategoryEnv(
            parse_badham_category_trials(text),
            instruction=parse_instruction_prefix(text, BADHAM_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == FREY_CCT_ID:
        inner = FreyCCTRecordedPathEnv(
            parse_frey_cct_rounds(text),
            instruction=parse_instruction_prefix(text, FREY_CCT_ROUND_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == KOOL_COST_EXP1_ID:
        inner = KoolCostExp1RecordedEnv(
            experiment_id,
            parse_kool_cost_exp1_days(text),
            instruction=parse_instruction_prefix(text, KOOL_COST_EXP1_DAY_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == KOOL_COST_EXP2_ID:
        inner = KoolCostExp2RecordedEnv(
            experiment_id,
            parse_kool_cost_exp2_days(text),
            instruction=parse_instruction_prefix(text, KOOL_COST_EXP2_DAY_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == WULFF_SAMPLING_ID:
        inner = WulffSamplingRecordedEnv(
            parse_wulff_sampling_problems(text),
            include_human_ref=include_human_ref,
        )
    elif experiment_id in TOMOV_SUBWAY_EXPERIMENT_IDS:
        inner = TomovSubwayRecordedEnv(
            experiment_id,
            parse_tomov_subway_trials(text),
            instruction=parse_instruction_prefix(text, TOMOV_STATION_STEP_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == WULFF_DESCRIPTION_ID:
        inner = WulffDescriptionRecordedEnv(
            parse_wulff_description_trials(text),
            instruction=parse_instruction_prefix(text, WULFF_LOTTERY_PRESS_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == HILBIG_ID:
        inner = HilbigProductRecordedEnv(
            parse_hilbig_product_trials(text),
            instruction=parse_instruction_prefix(text, HILBIG_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == ENKAVI_DIGIT_SPAN_EXP1_ID:
        inner = EnkaviDigitSpanRecordedEnv(
            parse_digit_span_recall_trials(text),
            instruction=parse_instruction_prefix(text, DIGIT_SPAN_BLOCK_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == ENKAVI_GONOGO_EXP1_ID:
        inner = EnkaviGonogoRecordedEnv(
            parse_gonogo_trials(text),
            instruction=parse_instruction_prefix(text, GONOGO_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == COX_PAIR_EXP1_ID:
        inner = CoxPairRecognitionRecordedEnv(
            experiment_id,
            parse_cox_pair_recognition_trials(text),
            instruction=parse_instruction_prefix(text, COX_PAIR_TEST_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id == GERSHMAN_MAPPING_ID:
        inner = GershmanMappingEnv(
            parse_gershman_mapping_trials(text),
            valid_actions=parse_gershman_response_actions(text),
            instruction=parse_instruction_prefix(text, GERSHMAN_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id in (STEINGROEVER_IGT_EXP1_ID, STEINGROEVER_IGT_EXP2_ID, STEINGROEVER_IGT_EXP3_ID):
        inner = SteingroeverIGTRecordedEnv(
            experiment_id,
            parse_steingroever_igt_trials(text),
            instruction=parse_instruction_prefix(text, STEINGROEVER_IGT_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    elif experiment_id in TOMOV_CASTLE_EXPERIMENT_IDS:
        inner = TomovCastleRecordedEnv(
            experiment_id,
            parse_tomov_castle_trials(text),
            instruction=parse_instruction_prefix(text, TOMOV_CASTLE_STEP_RE),
            include_human_ref=include_human_ref,
        )
    else:
        raise EnvironmentNotReadyError(experiment_id)
    return _wrap_generative(TranscriptBoundAuditEnv(inner, experiment_id), experiment_id)


def make_generative_env_from_transcript(
    experiment_id: str,
    text: str,
    include_human_ref: bool = False,
) -> Any:
    """Generative audit env: ``step()`` from generative code, schedule from *text*.

    Most experiments patch a ``make_generative_env`` instance so ``reset()``
    loads parsed trials instead of sampling a new episode. A few tasks (subway
    graph, Wulff sampling arms, Badham problem boundaries, CCT rounds, Kool
    days) use the exact-transition ``sim`` engine wrapped as
    ``TranscriptBoundAuditEnv`` because the transcript encodes that path rather
    than a fresh generative draw.
    """
    if experiment_id == PETERSON_ID:
        blocks = parse_peterson_gamble_blocks(text)
        instruction = parse_instruction_prefix(text, PETERSON_BLOCK_RE)
        env = PetersonGenerativeEnv(
            blocks=blocks,
            instruction=instruction,
            include_human_ref=include_human_ref,
            transcript_bound=True,
        )
        return _wrap_generative(env, experiment_id)

    if experiment_id == PLONSKY_ID:
        trials = [_plonsky_trial_from_parsed(t) for t in parse_plonsky_gamble_trials(text)]
        env = PlonskyGambleGenerativeEnv(
            schedule_trials=trials,
            include_human_ref=include_human_ref,
            transcript_bound=True,
        )
        return _wrap_generative(env, experiment_id)

    if experiment_id == FLESCH_TREE_EXP1_ID:
        trials = [_flesch_trial_from_parsed(t) for t in parse_flesch_tree_trials(text)]
        env = FleschTreeGenerativeEnv(
            schedule_trials=trials,
            include_human_ref=include_human_ref,
            transcript_bound=True,
        )
        return _wrap_generative(env, experiment_id)

    if experiment_id in _RECORDED_TRANSCRIPT_EXPERIMENTS:
        return _recorded_exact_env(experiment_id, text, include_human_ref)

    if experiment_id not in _PATCH_LOADERS:
        raise EnvironmentNotReadyError(
            "No generative-from-transcript binding for {!r}.".format(experiment_id)
        )
    env = _make_generative_instance(experiment_id, include_human_ref)
    _patch_reset(env, _PATCH_LOADERS[experiment_id], text)
    return _wrap_generative(env, experiment_id)


def generative_transcript_audit_experiments() -> Tuple[str, ...]:
    """Experiment ids supported by :func:`make_generative_env_from_transcript`."""
    supported = set(_PATCH_LOADERS)
    supported.update(
        {PETERSON_ID, PLONSKY_ID, FLESCH_TREE_EXP1_ID},
    )
    supported.update(_RECORDED_TRANSCRIPT_EXPERIMENTS)
    return tuple(sorted(supported & set(list_generative_experiments())))
