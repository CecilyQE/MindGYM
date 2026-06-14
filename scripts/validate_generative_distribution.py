#!/usr/bin/env python3
"""Validate fresh generative env rollouts and write distribution summaries.

This is intentionally a first-pass distributional validation, not a transcript
exact-match audit. It samples fresh ``make_generative_env`` episodes over many
seeds and checks generic invariants: reset succeeds, legal actions can drive an
episode, rewards are numeric, termination/truncation occurs within a step
budget, grounding metadata is present, and different seeds usually produce
non-identical trajectories for stochastic tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:  # pragma: no cover - test environments may not have PyYAML.
    yaml = None

from psycenvir.core.generative_registry import (  # noqa: E402
    list_generative_experiments,
    make_generative_env,
)
from psycenvir.generative.grounding import get_grounding_profile  # noqa: E402
from psycenvir.models import GONOGO_NO_PRESS  # noqa: E402

DEFAULT_TIERS = ROOT / "data" / "generated" / "generative_setting_tiers.yaml"
DEFAULT_JSON = ROOT / "results" / "generative_distribution_validation.json"
DEFAULT_MD = ROOT / "results" / "generative_distribution_validation.md"

ACTION_RE = re.compile(r"<<([^<>]+)>>")
INSTRUCTED_RE = re.compile(r"instructed to press\s+([^\s.]+)", re.IGNORECASE)
MACHINES_RE = re.compile(r"machines\s+([^\s.]+)\s+and\s+([^\s.]+)", re.IGNORECASE)
SPACESHIPS_RE = re.compile(r"spaceships\s+([^\s.]+)\s+and\s+([^\s.]+)", re.IGNORECASE)
ALIENS_RE = re.compile(r"alien\s+([^\s.]+)\s+and\s+alien\s+([^\s.]+)", re.IGNORECASE)
DOORS_RE = re.compile(r"doors?\s+([A-Z0-9])(?:,|\s+and|\s+or)?\s+([A-Z0-9])?(?:,|\s+and|\s+or)?\s*([A-Z0-9])?", re.IGNORECASE)


def _load_tier_a(path: Path) -> List[str]:
    if yaml is not None:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return list(data["tier_A"])
    ids: List[str] = []
    in_tier_a = False
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip()
            if line.startswith("tier_A:"):
                in_tier_a = True
                continue
            if line.startswith("tier_C:"):
                break
            if in_tier_a and line.strip().startswith("- "):
                ids.append(line.strip()[2:].strip())
    return ids


def _compact_config(experiment_id: str) -> Dict[str, Any]:
    """Small episodes for validation speed while preserving task structure."""
    if experiment_id.startswith("badham"):
        return {"n_problems": 2, "trials_per_problem": 6}
    if experiment_id.startswith("bahrami"):
        return {"n_trials": 8}
    if experiment_id.startswith("collsi"):
        return {"n_feedback_trials": 8, "n_silent_trials": 2}
    if experiment_id.startswith("cox2017"):
        return {"n_test_trials": 8, "n_studied_pairs": 8}
    if experiment_id.startswith("enkavi2019adaptivenback"):
        return {"n_blocks": 2, "block_base_trials": 6}
    if experiment_id.startswith("enkavi2019digitspan"):
        return {"n_spans": 3, "min_length": 2, "max_length": 4}
    if experiment_id.startswith("enkavi2019gonogo"):
        return {"n_practice_trials": 4, "n_test_trials": 8}
    if experiment_id.startswith("enkavi2019recentprobes"):
        return {"n_trials": 8}
    if experiment_id.startswith("flesch"):
        return {"n_training_trials": 6, "n_test_trials": 4}
    if experiment_id.startswith("frey2017cct"):
        return {"n_rounds": 4}
    if experiment_id.startswith("frey2017risk"):
        return {"n_balloons": 4, "min_threshold": 3, "max_threshold": 6}
    if experiment_id.startswith("garcia"):
        return {"part_trial_counts": (3, 3, 2)}
    if experiment_id.startswith("gershman2018"):
        return {"n_games": 2, "trials_per_game": 5}
    if experiment_id.startswith("gershman2020"):
        return {"n_games": 2, "trials_per_game": 5}
    if experiment_id.startswith("hilbig"):
        return {"n_trials": 8}
    if experiment_id.startswith("kool2016when/exp1"):
        return {"n_days": 4, "timeout_probability": 0.0}
    if experiment_id.startswith("kool2016when/exp2"):
        return {"n_days": 4}
    if experiment_id == "kool2017cost/exp2.csv":
        return {"n_days": 4}
    if experiment_id.startswith("kool2017cost"):
        return {"n_days": 4}
    if experiment_id.startswith("krueger"):
        return {"n_rounds": 4}
    if experiment_id.startswith("lefebvre"):
        return {"n_casinos": 2, "visits_per_casino": 4}
    if experiment_id.startswith("ludwig"):
        return {"n_blocks": 1, "trials_per_block": 3}
    if experiment_id.startswith("peterson"):
        return {"n_blocks": 2, "trials_per_block": 3}
    if experiment_id.startswith("plonsky"):
        return {"n_problems": 2, "trials_per_problem": 6, "no_feedback_trials": 2}
    if experiment_id.startswith("ruggeri"):
        return {}
    if experiment_id.startswith("schulz"):
        return {"n_rounds": 2, "trials_per_round": 5, "n_arms": 8}
    if experiment_id.startswith("speekenbrink"):
        return {"n_trials": 8}
    if experiment_id.startswith("steingroever"):
        return {"n_trials": 8}
    if experiment_id.startswith("wilson2014humans"):
        return {"n_games": 2, "instructed_trials": 2, "free_trials_choices": (2, 2)}
    if experiment_id.startswith("wu2018"):
        return {"n_environments": 2, "choices_short": 2, "choices_long": 2}
    if experiment_id.startswith("wu2023chunking"):
        return {"n_trials": 8}
    if experiment_id.startswith("wulff2018description"):
        return {"n_problems": 3}
    if experiment_id.startswith("wulff2018sampling"):
        return {"n_problems": 2, "max_samples_before_stop": 4}
    if experiment_id.startswith("xiong"):
        return {"hazard_rates": (0.1, 0.3), "games_per_hazard": 1, "trials_per_game": 5}
    if experiment_id.startswith("zorowitz"):
        return {"n_trials": 8}
    return {}


def _info_action_candidates(info: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in (
        "correct_action",
        "selected_action",
        "selected_arm",
        "selected_ship",
        "present_key",
        "absent_key",
        "match_key",
        "nonmatch_key",
        "pump_key",
        "collect_key",
    ):
        value = info.get(key)
        if isinstance(value, str):
            candidates.append(value)
    for key in ("valid_actions", "action_space", "actions", "decks", "arms", "gambles"):
        value = info.get(key)
        if isinstance(value, (list, tuple, set)):
            candidates.extend(str(item) for item in value)
    outcomes = info.get("outcomes_by_action")
    if isinstance(outcomes, dict):
        candidates.extend(str(key) for key in outcomes.keys())
    return _dedupe_actions(candidates)


def _regex_action_candidates(observation: str) -> List[str]:
    candidates = ACTION_RE.findall(observation)
    for regex in (INSTRUCTED_RE, MACHINES_RE, SPACESHIPS_RE, ALIENS_RE, DOORS_RE):
        for match in regex.finditer(observation):
            candidates.extend(group for group in match.groups() if group)
    return _dedupe_actions(candidates)


def _fallback_actions(experiment_id: str) -> List[str]:
    if experiment_id.startswith("badham"):
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if experiment_id.startswith("bahrami"):
        return ["L", "G", "O", "U", "A", "B", "C", "D"]
    if experiment_id.startswith("collsi"):
        return [
            "extremely low",
            "very low",
            "low",
            "somewhat low",
            "normal",
            "somewhat high",
            "high",
            "very high",
            "extremely high",
        ]
    if experiment_id.startswith("cox2017"):
        return ["D", "N"]
    if experiment_id.startswith("enkavi2019digitspan"):
        return [str(i) for i in range(10)] + ["S"]
    if experiment_id.startswith("enkavi2019gonogo"):
        return ["X", GONOGO_NO_PRESS]
    if experiment_id.startswith("enkavi2019recentprobes"):
        return ["K", "D"]
    if experiment_id.startswith("frey2017cct"):
        return ["E", "C"]
    if experiment_id.startswith("frey2017risk"):
        return ["H", "W"]
    if experiment_id.startswith("gershman2020"):
        return ["S", "K"]
    if experiment_id.startswith("hilbig"):
        return ["A", "R"]
    if experiment_id.startswith("kool2016when/exp2") or experiment_id == "kool2017cost/exp2.csv":
        return ["R", "F", "W", "Q", "P", "V"]
    if experiment_id.startswith("kool"):
        return ["P", "F", "R", "V"]
    if experiment_id.startswith("krueger"):
        return ["A", "B", "C", "D"]
    if experiment_id.startswith("lefebvre"):
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if experiment_id.startswith("peterson"):
        return ["Z", "L", "B", "H", "F"]
    if experiment_id.startswith("plonsky"):
        return ["F", "J", "B", "S"]
    if experiment_id.startswith("schulz"):
        return [str(i) for i in range(1, 9)]
    if experiment_id.startswith("speekenbrink"):
        return ["E", "J"]
    if experiment_id == "steingroever2015data/exp3.csv":
        return ["U", "F", "I", "S"]
    if experiment_id.startswith("steingroever"):
        return ["H", "V", "J", "D", "A", "I", "K"]
    if experiment_id.startswith("wilson2014humans"):
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if experiment_id.startswith("wu"):
        return [str(i) for i in range(10)]
    if experiment_id.startswith("wulff"):
        return ["W", "H", "K", "D", "C", "F", "X", "N"]
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]


def _dedupe_actions(actions: Iterable[str]) -> List[str]:
    seen = set()
    clean: List[str] = []
    for action in actions:
        value = str(action).strip().strip(".,;:")
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
    return clean


def _candidate_actions(experiment_id: str, observation: str, info: Dict[str, Any]) -> List[str]:
    state_actions = _state_action_candidates(info.get("_env_ref"))
    if state_actions:
        return state_actions
    return _dedupe_actions(
        _info_action_candidates(info)
        + _regex_action_candidates(observation)
        + _fallback_actions(experiment_id)
    )


def _state_action_candidates(env: Any) -> List[str]:
    if env is None:
        return []
    # GroundedGenerativeEnv wrapper.
    env = getattr(env, "_inner", env)
    candidates: List[str] = []

    def add_many(value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            candidates.extend(str(item) for item in value)

    # Current trial-like objects.
    for list_attr, index_attr in (
        ("_flat_trials", "_trial_idx"),
        ("_trials", "_trial_idx"),
        ("_steps", "_step_idx"),
        ("_problems", "_problem_idx"),
        ("_days", "_day_idx"),
        ("_rounds", "_trial_idx"),
    ):
        seq = getattr(env, list_attr, None)
        idx = getattr(env, index_attr, None)
        if isinstance(seq, list) and isinstance(idx, int) and 0 <= idx < len(seq):
            current = seq[idx]
            if isinstance(current, tuple):
                # Badham-style tuples are (prefix, stimulus, correct_label).
                if len(current) >= 3 and isinstance(current[2], str):
                    candidates.append(str(current[2]))
                else:
                    candidates.extend(str(item) for item in current if isinstance(item, str))
            else:
                add_many(getattr(current, "valid_actions", None))
                value = getattr(current, "valid_action", None)
                if isinstance(value, str):
                    candidates.append(value)
                value = getattr(current, "correct_action", None)
                if isinstance(value, str):
                    candidates.append(value)
                value = getattr(current, "instructed_key", None)
                if isinstance(value, str):
                    candidates.append(value)
                outcomes = getattr(current, "outcomes_by_action", None)
                if isinstance(outcomes, dict):
                    candidates.extend(str(key) for key in outcomes.keys())
                treasure_by_action = getattr(current, "treasure_by_action", None)
                if isinstance(treasure_by_action, dict):
                    candidates.extend(str(key) for key in treasure_by_action.keys())
                planet_map = getattr(current, "planet_map", None)
                if isinstance(planet_map, dict):
                    candidates.extend(str(key) for key in planet_map.keys())
                lotteries = getattr(current, "lotteries", None)
                if isinstance(lotteries, dict):
                    candidates.extend(str(key) for key in lotteries.keys())
                sample_pools = getattr(current, "sample_pools", None)
                if isinstance(sample_pools, dict):
                    candidates.extend(str(key) for key in sample_pools.keys())
                stop_action = getattr(current, "stop_action", None)
                if isinstance(stop_action, str):
                    candidates.append(stop_action)
                spaceship_phase = getattr(current, "spaceship_phase", None)
                if spaceship_phase is not None:
                    planet_by_ship = getattr(spaceship_phase, "planet_by_ship", None)
                    if isinstance(planet_by_ship, dict):
                        candidates.extend(str(key) for key in planet_by_ship.keys())
                    treasure_by_alien = getattr(spaceship_phase, "treasure_by_alien", None)
                    if isinstance(treasure_by_alien, dict):
                        candidates.extend(str(key) for key in treasure_by_alien.keys())
                common_planet_by_ship = getattr(current, "common_planet_by_ship", None)
                if isinstance(common_planet_by_ship, dict):
                    candidates.extend(str(key) for key in common_planet_by_ship.keys())
                treasure_probs = getattr(current, "treasure_probs", None)
                if isinstance(treasure_probs, dict):
                    candidates.extend(str(key) for key in treasure_probs.keys())
                market = getattr(current, "market", None)
                if isinstance(market, dict):
                    for path in market.keys():
                        if isinstance(path, tuple):
                            candidates.extend(str(item) for item in path)

    # State-dependent phase choices for two-step and query-choice tasks.
    phase_candidates: List[str] = []
    days = getattr(env, "_days", None)
    day_idx = getattr(env, "_day_idx", None)
    if isinstance(days, list) and isinstance(day_idx, int) and 0 <= day_idx < len(days):
        day = days[day_idx]
        if getattr(env, "_awaiting_alien", False):
            selected_planet = getattr(day, "selected_planet", None)
            for mapping_attr in ("spaceship_phase",):
                phase = getattr(day, mapping_attr, None)
                aliens_by_planet = getattr(phase, "aliens_by_planet", None)
                if isinstance(aliens_by_planet, dict) and isinstance(selected_planet, str):
                    phase_candidates.extend(str(key) for key in aliens_by_planet.get(selected_planet, ()))
                treasure_by_alien = getattr(phase, "treasure_by_alien", None)
                if isinstance(treasure_by_alien, dict) and not isinstance(selected_planet, str):
                    # Kool exp2 has planet-specific alien sets in module globals,
                    # but all alien keys are legal candidates only for their planet.
                    phase_candidates.extend(str(key) for key in treasure_by_alien.keys())
            if isinstance(selected_planet, str):
                if selected_planet == "J":
                    phase_candidates.extend(["W", "K"])
                elif selected_planet == "T":
                    phase_candidates.extend(["I", "G"])
                elif selected_planet == "blue":
                    phase_candidates.extend(["D", "R"])
                elif selected_planet == "red":
                    phase_candidates.extend(["G", "V"])
        else:
            phase = getattr(day, "spaceship_phase", None)
            planet_by_ship = getattr(phase, "planet_by_ship", None)
            if isinstance(planet_by_ship, dict):
                phase_candidates.extend(str(key) for key in planet_by_ship.keys())

    trials = getattr(env, "_trials", None)
    trial_idx = getattr(env, "_trial_idx", None)
    if isinstance(trials, list) and isinstance(trial_idx, int) and 0 <= trial_idx < len(trials):
        trial = trials[trial_idx]
        if getattr(env, "_awaiting_alien", False):
            selected_planet = getattr(trial, "selected_planet", None)
            if selected_planet == "blue":
                phase_candidates.extend(["D", "R"])
            elif selected_planet == "red":
                phase_candidates.extend(["G", "V"])
        else:
            common_planet_by_ship = getattr(trial, "common_planet_by_ship", None)
            if isinstance(common_planet_by_ship, dict):
                phase_candidates.extend(str(key) for key in common_planet_by_ship.keys())

    rounds = getattr(env, "_rounds", None)
    trial_idx = getattr(env, "_trial_idx", None)
    if isinstance(rounds, list) and isinstance(trial_idx, int) and 0 <= trial_idx < len(rounds):
        round_state = rounds[trial_idx]
        pending = getattr(env, "_pending_gamble", None)
        if pending is None:
            value = getattr(round_state, "valid_actions", None)
            if isinstance(value, (list, tuple, set)):
                phase_candidates.extend(str(item) for item in value)
        else:
            # Querying colors is optional; choosing STOP completes the round.
            phase_candidates.append("STOP")

    if phase_candidates:
        return _dedupe_actions(phase_candidates)

    # Peterson parsed/generated blocks.
    parsed_blocks = getattr(env, "_parsed_blocks", None)
    block_idx = getattr(env, "_block_idx", None)
    if isinstance(parsed_blocks, list) and isinstance(block_idx, int) and 0 <= block_idx < len(parsed_blocks):
        add_many(getattr(parsed_blocks[block_idx], "valid_actions", None))
    blocks = getattr(env, "_blocks", None)
    if isinstance(blocks, list) and isinstance(block_idx, int) and 0 <= block_idx < len(blocks):
        trial_idx = getattr(env, "_block_trial_idx", 0)
        block = blocks[block_idx]
        if isinstance(block, list) and isinstance(trial_idx, int) and 0 <= trial_idx < len(block):
            add_many(getattr(block[trial_idx], "valid_actions", None))

    # Tomov subway exposes a method for the current graph position.
    valid_fn = getattr(env, "_valid_actions", None)
    if callable(valid_fn):
        try:
            add_many(valid_fn())
        except Exception:
            pass

    if candidates:
        return _dedupe_actions(candidates)

    # Common simple attributes used when there is no current trial object.
    for attr in (
        "valid_actions",
        "action_keys",
        "arms",
        "decks",
        "gambles",
        "door_keys",
        "valid_action",
    ):
        if attr in env.__dict__:
            value = getattr(env, attr, None)
            if isinstance(value, str):
                candidates.append(value)
            else:
                add_many(value)

    for attr in (
        "pump_key",
        "collect_key",
        "match_key",
        "nonmatch_key",
        "present_key",
        "absent_key",
        "end_key",
        "_go_key",
    ):
        value = getattr(env, attr, None)
        if isinstance(value, str):
            candidates.append(value)

    return _dedupe_actions(candidates)


def _step_with_candidates(
    env: Any, experiment_id: str, observation: str, info: Dict[str, Any]
) -> Tuple[str, float, bool, bool, Dict[str, Any], str, bool]:
    candidates = _candidate_actions(experiment_id, observation, info)
    action = candidates[0] if candidates else "A"
    next_observation, reward, terminated, truncated, next_info = env.step(action)
    return next_observation, reward, terminated, truncated, next_info, action, False


def _run_episode(experiment_id: str, seed: int, max_steps: int) -> Dict[str, Any]:
    config = _compact_config(experiment_id)
    env = make_generative_env(experiment_id, seed=seed, include_human_ref=True, **config)
    observation, info = env.reset(seed=seed)
    info = dict(info)
    info["_env_ref"] = env
    signature_parts = [observation[:300]]
    rewards: List[float] = []
    actions: List[str] = []
    fallback_used = False
    terminated = False
    truncated = False
    issues: List[str] = []
    final_info: Dict[str, Any] = dict(info)

    if not isinstance(observation, str):
        issues.append("reset observation is not a string")
    if "generative_grounding" not in info:
        issues.append("reset info missing generative_grounding")
    if "fidelity_level" not in info:
        issues.append("reset info missing fidelity_level")

    for _ in range(max_steps):
        try:
            final_info["_env_ref"] = env
            observation, reward, terminated, truncated, final_info, action, used_fallback = (
                _step_with_candidates(env, experiment_id, observation, final_info)
            )
            final_info["_env_ref"] = env
        except Exception as exc:  # noqa: BLE001 - report validation failure.
            issues.append("step raised {}: {}".format(type(exc).__name__, exc))
            break
        fallback_used = fallback_used or used_fallback
        actions.append(action)
        if not isinstance(observation, str):
            issues.append("step observation is not a string")
            break
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            issues.append("reward is not a finite number: {!r}".format(reward))
            break
        rewards.append(float(reward))
        signature_parts.append("{}:{:.3f}:{}".format(action, float(reward), observation[:120]))
        if final_info.get("invalid_action") is not None:
            issues.append("invalid action accepted by candidate policy: {}".format(action))
            break
        if terminated or truncated:
            break

    exhausted_budget = not (terminated or truncated)
    if exhausted_budget:
        issues.append("step budget exhausted before episode end")
    if truncated and final_info.get("invalid_action") is not None:
        issues.append("episode truncated after invalid action")
    if "generative_grounding" not in final_info:
        issues.append("final info missing generative_grounding")
    if "fidelity_level" not in final_info:
        issues.append("final info missing fidelity_level")

    return {
        "seed": seed,
        "ok": not issues or (issues == ["step budget exhausted before episode end"]),
        "issues": issues,
        "warnings": ["step budget exhausted before episode end"] if exhausted_budget else [],
        "steps": len(actions),
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": sum(rewards),
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "actions": actions[:20],
        "fallback_used": fallback_used,
        "signature": hash(tuple(signature_parts)),
    }


def _summarize_experiment(
    experiment_id: str, seeds: Sequence[int], max_steps: int
) -> Dict[str, Any]:
    profile = get_grounding_profile(experiment_id)
    episodes: List[Dict[str, Any]] = []
    build_issue: Optional[str] = None
    for seed in seeds:
        try:
            episodes.append(_run_episode(experiment_id, seed, max_steps))
        except Exception as exc:  # noqa: BLE001 - report validation failure.
            build_issue = "{}: {}".format(type(exc).__name__, exc)
            episodes.append(
                {
                    "seed": seed,
                    "ok": False,
                    "issues": [build_issue],
                    "steps": 0,
                    "terminated": False,
                    "truncated": False,
                    "total_reward": 0.0,
                    "mean_reward": 0.0,
                    "min_reward": 0.0,
                    "max_reward": 0.0,
                    "actions": [],
                    "fallback_used": False,
                    "signature": None,
                }
            )
    ok_count = sum(1 for item in episodes if item["ok"])
    issue_counts = Counter(issue for item in episodes for issue in item.get("issues", []))
    warning_counts = Counter(warn for item in episodes for warn in item.get("warnings", []))
    total_rewards = [item["total_reward"] for item in episodes]
    step_counts = [item["steps"] for item in episodes]
    unique_signatures = len({item["signature"] for item in episodes if item["signature"] is not None})
    status = "pass"
    if ok_count < len(episodes):
        status = "fail"
    elif any(item.get("warnings") for item in episodes):
        status = "warning"
    elif any(item.get("fallback_used") for item in episodes):
        status = "warning"
    elif unique_signatures <= 1 and len(episodes) > 1:
        status = "warning"

    return {
        "experiment_id": experiment_id,
        "status": status,
        "grounding": profile.kind.value,
        "sources": list(profile.sources),
        "caveats": list(profile.caveats),
        "seeds": list(seeds),
        "episodes": len(episodes),
        "ok_episodes": ok_count,
        "unique_signatures": unique_signatures,
        "total_reward_mean": statistics.mean(total_rewards) if total_rewards else 0.0,
        "total_reward_stdev": statistics.pstdev(total_rewards) if len(total_rewards) > 1 else 0.0,
        "steps_mean": statistics.mean(step_counts) if step_counts else 0.0,
        "steps_min": min(step_counts) if step_counts else 0,
        "steps_max": max(step_counts) if step_counts else 0,
        "issue_counts": dict(issue_counts),
        "warning_counts": dict(warning_counts),
        "episodes_sample": episodes[:3],
    }


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# Generative Distribution Validation",
        "",
        "Fresh `make_generative_env` episodes sampled over multiple seeds. This validates generic rollout and distribution-health invariants, not transcript exact matching.",
        "",
        "| status | count |",
        "|--------|-------|",
    ]
    for status in ("pass", "warning", "fail"):
        lines.append("| {} | {} |".format(status, counts.get(status, 0)))
    lines.extend(
        [
            "",
            "| experiment_id | status | grounding | episodes | unique signatures | steps mean | reward mean | top issue |",
            "|---------------|--------|-----------|----------|-------------------|------------|-------------|-----------|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["status"], item["experiment_id"])):
        top_issue = "—"
        if row["issue_counts"]:
            top_issue = max(row["issue_counts"].items(), key=lambda item: item[1])[0]
        elif row.get("warning_counts"):
            top_issue = "warning: " + max(row["warning_counts"].items(), key=lambda item: item[1])[0]
        lines.append(
            "| `{}` | {} | {} | {}/{} | {} | {:.1f} | {:.2f} | {} |".format(
                row["experiment_id"],
                row["status"],
                row["grounding"],
                row["ok_episodes"],
                row["episodes"],
                row["unique_signatures"],
                row["steps_mean"],
                row["total_reward_mean"],
                top_issue.replace("|", "\\|"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", type=Path, default=DEFAULT_TIERS)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment ids. Default: tier-A registered generative ids.",
    )
    args = parser.parse_args()

    registered = set(list_generative_experiments())
    if args.experiments:
        experiments = list(args.experiments)
    else:
        experiments = [eid for eid in _load_tier_a(args.tiers) if eid in registered]
    seeds = tuple(range(args.seeds))
    rows = []
    for index, experiment_id in enumerate(experiments, start=1):
        print("[{}/{}] {}".format(index, len(experiments), experiment_id), flush=True)
        rows.append(_summarize_experiment(experiment_id, seeds, args.max_steps))

    payload = {
        "experiments": len(rows),
        "seeds": list(seeds),
        "max_steps": args.max_steps,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "results": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(rows, args.md)
    print("Wrote {} and {}".format(args.json, args.md))


if __name__ == "__main__":
    main()
