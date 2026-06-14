"""Recorded-path Wulff sampling task."""

from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import WULFF_SAMPLING_INSTRUCTION
from psycenvir.models import WulffSamplingProblem


class WulffSamplingRecordedEnv:
    """Sampling with pooled arm outcomes and a final bonus draw per problem."""

    EXPERIMENT_ID = "wulff2018sampling/exp1.csv"

    def __init__(
        self,
        problems: List[WulffSamplingProblem],
        include_human_ref: bool = False,
        instruction: str = WULFF_SAMPLING_INSTRUCTION,
    ) -> None:
        self.problems: List[WulffSamplingProblem] = list(problems)
        if not self.problems:
            raise ValueError("WulffSamplingRecordedEnv requires at least one problem.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._problem_idx = 0
        self._phase = "sampling"
        self._pool_indices: Dict[str, int] = {}
        self._sample_steps = 0
        self._bonus_total = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._problem_idx = 0
        self._phase = "sampling"
        self._pool_indices = {}
        self._sample_steps = 0
        self._bonus_total = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, "You encounter a new choice problem:"),
            self._info(),
        )

    def _current(self) -> WulffSamplingProblem:
        return self.problems[self._problem_idx]

    def _reset_pool_indices(self) -> None:
        self._pool_indices = {arm: 0 for arm in self._current().sampling_arms}

    def _next_pool_outcome(self, arm: str) -> float:
        problem = self._current()
        pool = list(problem.sample_pools.get(arm, []))
        if not pool:
            return 0.0
        index = self._pool_indices[arm] % len(pool)
        self._pool_indices[arm] += 1
        return pool[index]

    def _final_outcome(self, arm: str) -> float:
        problem = self._current()
        return float(problem.final_outcomes_by_action.get(arm, 0.0))

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WulffSamplingRecordedEnv; call reset().")
        submitted = normalize_action(action)

        problem = self._current()
        if not self._pool_indices:
            self._reset_pool_indices()

        if self._phase == "sampling":
            if (
                problem.fixed_sample_count is not None
                and self._sample_steps
                >= (len(problem.sample_sequence) if problem.sample_sequence else problem.fixed_sample_count)
            ):
                self._phase = "final"
            elif problem.stop_action is not None and submitted == problem.stop_action:
                self._phase = "final"
                arm0, arm1 = problem.sampling_arms
                return (
                    "You press <<{}>> to stop sampling and then press <<{}>> or <<{}>>.".format(
                        problem.stop_action, arm0, arm1
                    ),
                    0.0,
                    False,
                    False,
                    self._info(),
                )
            elif submitted not in problem.sampling_arms:
                self._done = True
                info = self._info()
                info["invalid_action"] = submitted
                return "Invalid sampling action <<{}>>.".format(submitted), 0.0, False, True, info
            elif self._phase == "sampling":
                if problem.sample_sequence:
                    expected_arm, outcome = problem.sample_sequence[self._sample_steps]
                    if submitted != expected_arm:
                        self._done = True
                        info = self._info()
                        info["invalid_action"] = submitted
                        info["expected_action"] = expected_arm
                        return (
                            "Invalid recorded sampling action <<{}>>; expected <<{}>>.".format(
                                submitted, expected_arm
                            ),
                            0.0,
                            False,
                            True,
                            info,
                        )
                else:
                    outcome = self._next_pool_outcome(submitted)
                self._sample_steps += 1
                feedback = "You press <<{}>> and observe {} points.".format(submitted, outcome)
                return feedback, 0.0, False, False, self._info(
                    selected_arm=submitted, sample_outcome=outcome
                )

        if submitted not in problem.sampling_arms:
            self._done = True
            info = self._info()
            info["invalid_action"] = submitted
            return "Invalid final lottery action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = self._final_outcome(submitted)
        self._bonus_total += reward
        if problem.stop_action is not None:
            feedback = "You press <<{}>> to stop sampling and then press <<{}>>.".format(
                problem.stop_action, submitted
            )
        else:
            feedback = "You are asked to choose one lottery for real and you press <<{}>>.".format(
                submitted
            )
        self._problem_idx += 1
        self._phase = "sampling"
        self._pool_indices = {}
        self._sample_steps = 0
        self._done = self._problem_idx >= len(self.problems)
        if not self._done:
            feedback = "{}\n\nYou encounter a new choice problem:".format(feedback)
        return feedback, reward, self._done, False, self._info(selected_arm=submitted, final_outcome=reward)

    def _info(
        self,
        selected_arm: Optional[str] = None,
        sample_outcome: Optional[float] = None,
        final_outcome: Optional[float] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "problem_idx": self._problem_idx,
            "phase": self._phase,
            "bonus_total": self._bonus_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_arm": selected_arm,
        }
        if self.include_human_ref:
            problem = self._current()
            info["human_final_action"] = problem.human_final_action
            info["sample_pools"] = {arm: list(pool) for arm, pool in problem.sample_pools.items()}
        if sample_outcome is not None:
            info["sample_outcome"] = sample_outcome
        if final_outcome is not None:
            info["final_outcome"] = final_outcome
        return info
