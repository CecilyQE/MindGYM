"""Generative Wulff sampling task: explore K/D lotteries then stop with X and choose."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.dynamics import sample_discrete_outcome
from psycenvir.generative.instructions import WULFF_SAMPLING_INSTRUCTION

WULFF_SAMPLING_EXPERIMENT_ID = "wulff2018sampling/exp1.csv"
SAMPLING_ARMS = ("K", "D")
STOP_ACTION = "X"


@dataclass
class _SamplingProblem:
    problem_number: int
    lotteries: Dict[str, List[Tuple[float, float]]]
    sample_pools: Dict[str, List[float]] = field(default_factory=dict)


class WulffSamplingGenerativeEnv:
    """Fresh choice problems with free sampling then a paid final lottery draw."""

    def __init__(
        self,
        experiment_id: str = WULFF_SAMPLING_EXPERIMENT_ID,
        n_problems: int = 3,
        max_samples_before_stop: int = 120,
        instruction: str = WULFF_SAMPLING_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_problems = n_problems
        self.max_samples_before_stop = max_samples_before_stop
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._problems: List[_SamplingProblem] = []
        self._problem_idx = 0
        self._phase = "sampling"
        self._sample_counts = {arm: 0 for arm in SAMPLING_ARMS}
        self._bonus_total = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._problems = []
        for problem_number in range(1, self.n_problems + 1):
            lotteries = {
                "K": self._sample_lottery(),
                "D": self._sample_lottery(),
            }
            self._problems.append(_SamplingProblem(problem_number=problem_number, lotteries=lotteries))
        self._problem_idx = 0
        self._phase = "sampling"
        self._sample_counts = {arm: 0 for arm in SAMPLING_ARMS}
        self._bonus_total = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_problem_header()),
            self._info(),
        )

    def _sample_lottery(self) -> List[Tuple[float, float]]:
        high = round(self._rng.uniform(0.0, 8.0), 1)
        low = round(self._rng.uniform(-35.0, 2.0), 1)
        probability = self._rng.uniform(0.1, 0.9)
        return [(high, probability), (low, 1.0 - probability)]

    def _current_problem(self) -> _SamplingProblem:
        return self._problems[self._problem_idx]

    def _draw_sample(self, arm: str) -> float:
        problem = self._current_problem()
        if arm not in problem.sample_pools:
            problem.sample_pools[arm] = []
        lottery = problem.lotteries[arm]
        values = [value for value, _ in lottery]
        probabilities = [probability for _, probability in lottery]
        outcome = sample_discrete_outcome(self._rng, values, probabilities)
        problem.sample_pools[arm].append(outcome)
        return outcome

    def _draw_final(self, arm: str) -> float:
        problem = self._current_problem()
        lottery = problem.lotteries[arm]
        values = [value for value, _ in lottery]
        probabilities = [probability for _, probability in lottery]
        return sample_discrete_outcome(self._rng, values, probabilities)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WulffSamplingGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        problem = self._current_problem()

        if self._phase == "sampling":
            if submitted == STOP_ACTION:
                self._phase = "final"
                feedback = (
                    "You encounter a new choice problem:\n"
                    "You press <<X>> to stop sampling and then press <<K>> or <<D>> for your bonus draw."
                )
                return feedback, 0.0, False, False, self._info()

            if submitted not in SAMPLING_ARMS:
                self._done = True
                info = self._info()
                info["invalid_action"] = submitted
                return "Invalid sampling action <<{}>>.".format(submitted), 0.0, False, True, info

            if sum(self._sample_counts.values()) >= self.max_samples_before_stop:
                self._done = True
                info = self._info()
                info["unsupported_counterfactual"] = "sample_budget_exhausted"
                return "No more sampling draws are available in this problem.", 0.0, False, True, info

            outcome = self._draw_sample(submitted)
            self._sample_counts[submitted] += 1
            feedback = "You press <<{}>> and observe {} points.".format(submitted, outcome)
            return feedback, 0.0, False, False, self._info(selected_arm=submitted, sample_outcome=outcome)

        if submitted not in SAMPLING_ARMS:
            self._done = True
            info = self._info()
            info["invalid_action"] = submitted
            return "Invalid final lottery action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = self._draw_final(submitted)
        self._bonus_total += reward
        feedback = (
            "You encounter a new choice problem:\n"
            "You press <<X>> to stop sampling and then press <<{}>>."
        ).format(submitted)
        self._problem_idx += 1
        self._phase = "sampling"
        self._sample_counts = {arm: 0 for arm in SAMPLING_ARMS}
        self._done = self._problem_idx >= len(self._problems)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_problem_header())
        return feedback, reward, self._done, False, self._info(selected_arm=submitted, final_outcome=reward)

    def _render_problem_header(self) -> str:
        return "You encounter a new choice problem:"

    def _info(
        self,
        selected_arm: Optional[str] = None,
        sample_outcome: Optional[float] = None,
        final_outcome: Optional[float] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "problem_idx": self._problem_idx,
            "phase": self._phase,
            "bonus_total": self._bonus_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_arm": selected_arm,
        }
        if sample_outcome is not None:
            info["sample_outcome"] = sample_outcome
        if final_outcome is not None:
            info["final_outcome"] = final_outcome
        if self.include_human_ref and self._problems:
            problem = self._current_problem()
            info["lotteries"] = {
                arm: list(distribution) for arm, distribution in problem.lotteries.items()
            }
        return info
