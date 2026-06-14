"""Generative Wulff description lottery choice (two explicit lotteries)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.dynamics import sample_discrete_outcome
from psycenvir.generative.instructions import WULFF_DESCRIPTION_INSTRUCTION

WULFF_DESCRIPTION_EXPERIMENT_ID = "wulff2018description/exp1.csv"


@dataclass(frozen=True)
class _LotteryOutcome:
    value: float
    probability: float


@dataclass(frozen=True)
class _LotteryProblem:
    observation: str
    valid_actions: Tuple[str, str]
    lotteries: Dict[str, Tuple[_LotteryOutcome, ...]]


class WulffDescriptionGenerativeEnv:
    """Fresh two-lottery problems with stated outcome probabilities."""

    def __init__(
        self,
        experiment_id: str = WULFF_DESCRIPTION_EXPERIMENT_ID,
        n_problems: int = 20,
        instruction: str = WULFF_DESCRIPTION_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_problems = n_problems
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._problems: List[_LotteryProblem] = []
        self._problem_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._problems = [self._sample_problem() for _ in range(self.n_problems)]
        self._problem_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._problems[0].observation),
            self._info(self._problems[0], None),
        )

    def _sample_problem(self) -> _LotteryProblem:
        labels = ("W", "H")
        lotteries: Dict[str, Tuple[_LotteryOutcome, ...]] = {}
        lines = []
        for label in labels:
            high = round(self._rng.uniform(1.0, 8.0), 1)
            low = round(self._rng.uniform(-5.0, 2.0), 1)
            probability = self._rng.uniform(0.1, 0.9)
            lotteries[label] = (
                _LotteryOutcome(high, probability),
                _LotteryOutcome(low, 1.0 - probability),
            )
            lines.append(
                "Lottery {} offers {} points with {:.1f}% probability or {} points with {:.1f}% probability.".format(
                    label,
                    high,
                    probability * 100.0,
                    low,
                    (1.0 - probability) * 100.0,
                )
            )
        return _LotteryProblem(
            observation="\n".join(lines),
            valid_actions=labels,
            lotteries=lotteries,
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WulffDescriptionGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        problem = self._problems[self._problem_idx]
        if submitted not in problem.valid_actions:
            self._done = True
            info = self._info(problem, None)
            info["invalid_action"] = submitted
            return "Invalid lottery action <<{}>>.".format(submitted), 0.0, False, True, info

        lottery = problem.lotteries[submitted]
        reward = sample_discrete_outcome(
            self._rng,
            [outcome.value for outcome in lottery],
            [outcome.probability for outcome in lottery],
        )
        self._points += reward
        feedback = "{}\nYou press <<{}>>.".format(problem.observation, submitted)
        self._problem_idx += 1
        self._done = self._problem_idx >= len(self._problems)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._problems[self._problem_idx].observation)
        return feedback, reward, self._done, False, self._info(problem, submitted)

    def _info(self, problem: Optional[_LotteryProblem], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "problem_idx": self._problem_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        return info
