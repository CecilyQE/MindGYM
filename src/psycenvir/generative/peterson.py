"""Generative gamble environment with choices13k Corr/Amb problem schema."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import PETERSON_INSTRUCTION
from psycenvir.models import PetersonGambleBlock, PetersonGambleTrial
from psycenvir.psych101.parse import parse_peterson_gamble_blocks

PETERSON_EXPERIMENT_ID = "peterson2021using/exp1.csv"
DEFAULT_PSYCH101_JSONL = Path(__file__).resolve().parents[3] / "data" / "raw" / "prompts_training.jsonl"
DEFAULT_CHOICES13K_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "choices13k"


@dataclass(frozen=True)
class _GeneratedGambleTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    problem_id: Optional[int] = None
    corr: Optional[int] = None
    amb: Optional[bool] = None
    has_feedback: bool = True


@dataclass(frozen=True)
class _ChoiceProblem:
    problem_id: int
    distributions: Dict[str, Tuple[Tuple[float, float], ...]]
    corr: int
    amb: bool
    feedback: bool


def _format_probability(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return "{:.4f}".format(value).rstrip("0").rstrip(".")


def _format_percent(probability: float) -> str:
    return "{:.1f}%".format(probability * 100.0)


def _format_distribution(
    action: str, distribution: Sequence[Tuple[float, float]], hide_probabilities: bool = False
) -> str:
    parts = []
    for probability, outcome in distribution:
        if hide_probabilities:
            parts.append("{} points with unknown chance".format(outcome))
        else:
            parts.append("{} points with {} chance".format(outcome, _format_percent(probability)))
    if len(parts) == 1:
        return "Option {} delivers {}.".format(action, parts[0])
    return "Option {} delivers {}.".format(action, ", or ".join(parts))


def _build_observation(
    action_a: str,
    dist_a: Sequence[Tuple[float, float]],
    action_b: str,
    dist_b: Sequence[Tuple[float, float]],
    amb: bool,
) -> str:
    return "{}\n{}".format(
        _format_distribution(action_a, dist_a, hide_probabilities=False),
        _format_distribution(action_b, dist_b, hide_probabilities=amb),
    )


def _sample_distribution(rng: random.Random, distribution: Sequence[Tuple[float, float]], u: float) -> float:
    cumulative = 0.0
    for probability, outcome in distribution:
        cumulative += probability
        if u <= cumulative:
            return float(outcome)
    return float(distribution[-1][1])


def _sample_joint_outcomes(
    rng: random.Random,
    dist_a: Sequence[Tuple[float, float]],
    dist_b: Sequence[Tuple[float, float]],
    corr: int,
) -> Tuple[float, float]:
    u_a = rng.random()
    if corr > 0:
        u_b = u_a
    elif corr < 0:
        u_b = 1.0 - u_a
    else:
        u_b = rng.random()
    return _sample_distribution(rng, dist_a, u_a), _sample_distribution(rng, dist_b, u_b)


def load_choices13k_problems(
    data_dir: Path = DEFAULT_CHOICES13K_DIR,
) -> List[_ChoiceProblem]:
    problems_path = data_dir / "c13k_problems.json"
    selections_path = data_dir / "c13k_selections.csv"
    if not problems_path.exists() or not selections_path.exists():
        return []
    raw_problems = json.loads(problems_path.read_text(encoding="utf-8"))
    loaded: List[_ChoiceProblem] = []
    with selections_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            problem_number = int(row["Problem"])
            problem_key = str(problem_number - 1)
            raw = raw_problems.get(problem_key)
            if raw is None:
                continue
            loaded.append(
                _ChoiceProblem(
                    problem_id=problem_number,
                    distributions={
                        "A": tuple((float(prob), float(outcome)) for prob, outcome in raw["A"]),
                        "B": tuple((float(prob), float(outcome)) for prob, outcome in raw["B"]),
                    },
                    corr=int(row["Corr"]),
                    amb=row["Amb"].strip().lower() == "true",
                    feedback=row["Feedback"].strip().lower() == "true",
                )
            )
    return loaded


def load_peterson_outcome_pairs(
    jsonl_path: Path = DEFAULT_PSYCH101_JSONL,
) -> List[Tuple[float, float]]:
    """Collect ordered payoff pairs from recorded full-feedback Peterson blocks."""
    if not jsonl_path.exists():
        return []
    pairs: List[Tuple[float, float]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("experiment") != PETERSON_EXPERIMENT_ID:
                continue
            for block in parse_peterson_gamble_blocks(row["text"]):
                if not block.has_feedback:
                    continue
                for trial in block.trials:
                    assert trial.outcomes_by_action is not None
                    values = list(trial.outcomes_by_action.values())
                    if len(values) == 2:
                        pairs.append((float(values[0]), float(values[1])))
    return pairs


class PetersonGenerativeEnv:
    """Fresh gamble blocks sampled from choices13k problems with Corr/Amb outcomes."""

    def __init__(
        self,
        n_blocks: int = 20,
        trials_per_block: int = 5,
        action_pairs: Sequence[Tuple[str, str]] = (("Z", "L"), ("A", "B")),
        outcome_pairs: Optional[Sequence[Tuple[float, float]]] = None,
        jsonl_path: Path = DEFAULT_PSYCH101_JSONL,
        choices13k_dir: Path = DEFAULT_CHOICES13K_DIR,
        choice_problems: Optional[Sequence[_ChoiceProblem]] = None,
        instruction: str = PETERSON_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
        blocks: Optional[Iterable[PetersonGambleBlock]] = None,
        transcript_bound: bool = False,
    ) -> None:
        self._parsed_blocks: Optional[List[PetersonGambleBlock]] = (
            list(blocks) if blocks is not None else None
        )
        self.transcript_bound = transcript_bound
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._block_idx = 0
        self._trial_idx = 0
        self._block_trial_idx = 0
        self._points = 0.0
        self._done = False
        self._blocks: List[List[_GeneratedGambleTrial]] = []
        if self._parsed_blocks is not None:
            if not self._parsed_blocks:
                raise ValueError("PetersonGenerativeEnv requires at least one parsed block.")
            self.n_blocks = len(self._parsed_blocks)
            self.trials_per_block = len(self._parsed_blocks[0].trials)
            self.action_pairs = tuple()
            self.outcome_pairs = []
            self._rng = random.Random(0)
            return
        if n_blocks <= 0 or trials_per_block <= 0:
            raise ValueError("n_blocks and trials_per_block must be positive.")
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.action_pairs = tuple(action_pairs)
        self.outcome_pairs = list(outcome_pairs) if outcome_pairs is not None else []
        self.choice_problems = (
            list(choice_problems)
            if choice_problems is not None
            else ([] if outcome_pairs is not None else load_choices13k_problems(choices13k_dir))
        )
        if not self.choice_problems and not self.outcome_pairs:
            self.outcome_pairs = load_peterson_outcome_pairs(jsonl_path)
        if not self.choice_problems and not self.outcome_pairs:
            raise ValueError(
                "PetersonGenerativeEnv requires choices13k data or explicit outcome_pairs."
            )
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if self._parsed_blocks is not None:
            del seed
            self._block_idx = 0
            self._block_trial_idx = 0
            self._trial_idx = 0
            self._points = 0.0
            self._done = False
            return (
                render_initial_observation(
                    self.instruction, self._parsed_blocks[0].observation
                ),
                self._info_parsed(None, None),
            )
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._blocks = []
        for _ in range(self.n_blocks):
            action_a, action_b = self._rng.choice(self.action_pairs)
            block_trials = []
            problem = self._rng.choice(self.choice_problems) if self.choice_problems else None
            for _trial in range(self.trials_per_block):
                if problem is None:
                    outcome_a, outcome_b = self._rng.choice(self.outcome_pairs)
                    observation = _build_observation(
                        action_a,
                        ((0.5, outcome_a), (0.5, outcome_b)),
                        action_b,
                        ((0.5, outcome_b), (0.5, outcome_a)),
                        False,
                    )
                    corr = 0
                    amb = False
                    has_feedback = True
                else:
                    outcome_a, outcome_b = _sample_joint_outcomes(
                        self._rng,
                        problem.distributions["A"],
                        problem.distributions["B"],
                        problem.corr,
                    )
                    observation = _build_observation(
                        action_a,
                        problem.distributions["A"],
                        action_b,
                        problem.distributions["B"],
                        problem.amb,
                    )
                    corr = problem.corr
                    amb = problem.amb
                    has_feedback = problem.feedback
                block_trials.append(
                    _GeneratedGambleTrial(
                        observation=observation,
                        valid_actions=(action_a, action_b),
                        outcomes_by_action={action_a: outcome_a, action_b: outcome_b},
                        problem_id=None if problem is None else problem.problem_id,
                        corr=corr,
                        amb=amb,
                        has_feedback=has_feedback,
                    )
                )
            self._blocks.append(block_trials)
        self._block_idx = 0
        self._trial_idx = 0
        self._block_trial_idx = 0
        self._points = 0.0
        self._done = False
        first = self._blocks[0][0]
        return (
            render_initial_observation(self.instruction, first.observation),
            self._info(first, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed PetersonGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        if self._parsed_blocks is not None:
            return self._step_parsed(submitted)
        trial = self._blocks[self._block_idx][self._block_trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info

        alternative = (
            trial.valid_actions[1]
            if submitted == trial.valid_actions[0]
            else trial.valid_actions[0]
        )
        selected_outcome = trial.outcomes_by_action[submitted]
        forgone_outcome = trial.outcomes_by_action[alternative]
        if trial.has_feedback:
            reward = float(selected_outcome)
            self._points += reward
            feedback = (
                "You press <<{}>>. You receive {} points by selecting this option. "
                "You would have received {} points had you chosen the other option."
            ).format(submitted, selected_outcome, forgone_outcome)
        else:
            reward = 0.0
            feedback = "You press <<{}>>.".format(submitted)

        self._trial_idx += 1
        self._block_trial_idx += 1
        if self._block_trial_idx >= len(self._blocks[self._block_idx]):
            self._block_idx += 1
            self._block_trial_idx = 0
        self._done = self._block_idx >= len(self._blocks)
        if not self._done and self._block_trial_idx == 0:
            next_observation = self._blocks[self._block_idx][0].observation
            observation = "{}\n\n{}".format(feedback, next_observation)
        else:
            observation = feedback
        return observation, reward, self._done, False, self._info(trial, selected_outcome)

    def _step_parsed(self, submitted: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        block = self._parsed_blocks[self._block_idx]
        trial = block.trials[self._block_trial_idx]
        if submitted not in block.valid_actions:
            self._done = True
            info = self._info_parsed(block, trial)
            info["invalid_action"] = submitted
            return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info

        if not block.has_feedback:
            if submitted != trial.human_action:
                self._done = True
                info = self._info_parsed(block, trial)
                info["unsupported_counterfactual"] = "no_feedback_block"
                return (
                    "No counterfactual outcomes are available for this no-feedback block.",
                    0.0,
                    False,
                    True,
                    info,
                )
            feedback = "You press <<{}>>.".format(submitted)
            reward = 0.0
        else:
            assert trial.outcomes_by_action is not None
            alternative = (
                block.valid_actions[1]
                if submitted == block.valid_actions[0]
                else block.valid_actions[0]
            )
            selected_outcome = trial.outcomes_by_action[submitted]
            forgone_outcome = trial.outcomes_by_action[alternative]
            reward = float(selected_outcome)
            self._points += reward
            feedback = (
                "You press <<{}>>. You receive {} points by selecting this option. "
                "You would have received {} points had you chosen the other option."
            ).format(submitted, selected_outcome, forgone_outcome)

        self._trial_idx += 1
        self._block_trial_idx += 1
        if self._block_trial_idx >= len(block.trials):
            self._block_idx += 1
            self._block_trial_idx = 0
        self._done = self._block_idx >= len(self._parsed_blocks)
        if not self._done and self._block_trial_idx == 0:
            feedback = "{}\n\n{}".format(
                feedback, self._parsed_blocks[self._block_idx].observation
            )
        return feedback, reward, self._done, False, self._info_parsed(block, trial)

    def _info_parsed(
        self,
        block: Optional[PetersonGambleBlock],
        trial: Optional[PetersonGambleTrial],
    ) -> Dict[str, Any]:
        info = self._info(None, None)
        info["fidelity_level"] = (
            "generative_transcript_calibrated"
            if self.transcript_bound
            else "generative_partial"
        )
        info["source_feedback_only"] = False
        if block is not None:
            info["has_feedback"] = block.has_feedback
            info["source_block_idx"] = block.source_block_idx
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
        return info

    def _info(
        self, trial: Optional[_GeneratedGambleTrial], selected_outcome: Optional[float]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": PETERSON_EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": (
                "generative_transcript_calibrated"
                if self.transcript_bound
                else "generative_exact"
            ),
            "episode_generative": True,
            "source_feedback_only": False,
            "instruction_shown": bool(self.instruction),
            "selected_outcome": selected_outcome,
        }
        if trial is not None:
            info.update(
                {
                    "problem_id": trial.problem_id,
                    "corr": trial.corr,
                    "amb": trial.amb,
                    "has_feedback": trial.has_feedback,
                    "counterfactual_mode": "choices13k_joint_sample",
                }
            )
            if self.include_human_ref:
                info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
