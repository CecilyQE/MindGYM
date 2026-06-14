"""Generative category-learning environment modeled on Badham et al. (2017)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import BADHAM_INSTRUCTION

BADHAM_EXPERIMENT_ID = "badham2017deficits/exp1.csv"
BADHAM_STIMULI: Tuple[str, ...] = (
    "a big black square",
    "a small black square",
    "a big white square",
    "a small white square",
    "a big black triangle",
    "a small black triangle",
    "a big white triangle",
    "a small white triangle",
)
RULE_DIMENSIONS: Tuple[str, ...] = ("size", "color", "shape")
PROBLEM_RESET_CUE = "You encounter a new problem with a new rule determining which objects belong to each category:"


@dataclass(frozen=True)
class _BadhamProblem:
    category_a: str
    category_b: str
    dimension: str
    positive_value: str
    trials: Tuple[Tuple[str, str], ...]


def _stimulus_features(stimulus: str) -> Dict[str, str]:
    parts = stimulus.split()
  # a big black square
    return {"size": parts[1], "color": parts[2], "shape": parts[3]}


def _label_for_stimulus(stimulus: str, problem: _BadhamProblem) -> str:
    features = _stimulus_features(stimulus)
    value = features[problem.dimension]
    return problem.category_a if value == problem.positive_value else problem.category_b


def _sample_problem(
    rng: random.Random,
    trials_per_problem: int,
    label_pool: Sequence[str],
) -> _BadhamProblem:
    category_a, category_b = rng.sample(list(label_pool), 2)
    dimension = rng.choice(RULE_DIMENSIONS)
    if dimension == "size":
        positive_value = rng.choice(["big", "small"])
    elif dimension == "color":
        positive_value = rng.choice(["black", "white"])
    else:
        positive_value = rng.choice(["square", "triangle"])
    schedule = list(BADHAM_STIMULI) * max(1, trials_per_problem // len(BADHAM_STIMULI))
    rng.shuffle(schedule)
    schedule = schedule[:trials_per_problem]
    problem = _BadhamProblem(
        category_a=category_a,
        category_b=category_b,
        dimension=dimension,
        positive_value=positive_value,
        trials=(),
    )
    trials = tuple((stimulus, _label_for_stimulus(stimulus, problem)) for stimulus in schedule)
    return _BadhamProblem(
        category_a=category_a,
        category_b=category_b,
        dimension=dimension,
        positive_value=positive_value,
        trials=trials,
    )


class BadhamGenerativeEnv:
    """Sample a fresh Badham-style session with causal category feedback."""

    def __init__(
        self,
        n_problems: int = 4,
        trials_per_problem: int = 88,
        label_pool: Sequence[str] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        instruction: str = BADHAM_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if n_problems <= 0 or trials_per_problem <= 0:
            raise ValueError("n_problems and trials_per_problem must be positive.")
        if len(label_pool) < 2:
            raise ValueError("label_pool must contain at least two category labels.")
        self.n_problems = n_problems
        self.trials_per_problem = trials_per_problem
        self.label_pool = tuple(label_pool)
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._problems: List[_BadhamProblem] = []
        self._flat_trials: List[Tuple[str, str, str]] = []
        self._problem_starts: List[int] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        self._valid_actions: Optional[set] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._problems = [
            _sample_problem(self._rng, self.trials_per_problem, self.label_pool)
            for _ in range(self.n_problems)
        ]
        self._flat_trials = []
        self._problem_starts = []
        for problem in self._problems:
            self._problem_starts.append(len(self._flat_trials))
            for stimulus, label in problem.trials:
                prefix = PROBLEM_RESET_CUE if len(self._flat_trials) == self._problem_starts[-1] else ""
                self._flat_trials.append((prefix, stimulus, label))
        self._valid_actions = {
            problem.category_a for problem in self._problems
        } | {problem.category_b for problem in self._problems}
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_stimulus(0)),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed BadhamGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        if self._valid_actions is not None and submitted not in self._valid_actions:
            self._done = True
            info = self._info(None, False)
            info["invalid_action"] = submitted
            return "Invalid category action <<{}>>.".format(submitted), 0.0, False, True, info

        prefix, stimulus, correct = self._flat_trials[self._trial_idx]
        is_correct = submitted == correct
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        feedback = "You press <<{}>>. The correct category is {}.".format(submitted, correct)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._flat_trials)
        if not self._done:
            next_prefix, next_stimulus, _ = self._flat_trials[self._trial_idx]
            observation = "{} {}".format(feedback, self._compose_observation(next_prefix, next_stimulus))
        else:
            observation = feedback
        return observation, reward, self._done, False, self._info(correct, is_correct)

    def _render_stimulus(self, trial_idx: int) -> str:
        prefix, stimulus, _ = self._flat_trials[trial_idx]
        return self._compose_observation(prefix, stimulus)

    @staticmethod
    def _compose_observation(prefix: str, stimulus: str) -> str:
        core = "You see {}.".format(stimulus)
        if prefix:
            return "{} {}".format(prefix, core)
        return core

    def _info(
        self, correct_action: Optional[str], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        return {
            "experiment_id": BADHAM_EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "n_problems": self.n_problems,
            "trials_per_problem": self.trials_per_problem,
            "action_space_validated": self._valid_actions is not None,
            "instruction_shown": bool(self.instruction),
            "correct_action": correct_action,
            "is_correct": is_correct,
        }
