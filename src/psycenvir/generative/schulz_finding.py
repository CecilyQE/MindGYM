"""Generative Schulz finding task (multi-arm bandit with volatile rounds)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import SCHULZ_FINDING_INSTRUCTION

SCHULZ_FINDING_EXP1_ID = "schulz2020finding/exp1.csv"
SCHULZ_FINDING_EXPERIMENT_IDS = tuple(
    "schulz2020finding/exp{}.csv".format(index) for index in range(1, 6)
)
DEFAULT_SCHULZ_ARMS = tuple(str(index) for index in range(1, 9))


@dataclass(frozen=True)
class _SchulzTrial:
    round_number: int
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, float]
    show_round_header: bool


class SchulzFindingGenerativeEnv:
    """Fresh rounds with latent arm payoffs that reset between rounds."""

    def __init__(
        self,
        experiment_id: str = SCHULZ_FINDING_EXP1_ID,
        n_rounds: int = 30,
        trials_per_round: int = 10,
        n_arms: int = 8,
        payoff_min: float = 0.0,
        payoff_max: float = 50.0,
        payoff_noise: float = 0.2,
        instruction: str = SCHULZ_FINDING_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("SchulzFindingGenerativeEnv requires at least two arms.")
        self.experiment_id = experiment_id
        self.n_rounds = n_rounds
        self.trials_per_round = trials_per_round
        self.n_arms = n_arms
        self.payoff_min = payoff_min
        self.payoff_max = payoff_max
        self.payoff_noise = payoff_noise
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_SchulzTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def _arm_labels(self) -> Tuple[str, ...]:
        return tuple(str(index) for index in range(1, self.n_arms + 1))

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        arms = self._arm_labels()
        self._trials = []
        for round_number in range(1, self.n_rounds + 1):
            latents = {
                arm: self._rng.uniform(self.payoff_min, self.payoff_max) for arm in arms
            }
            for trial_number in range(self.trials_per_round):
                outcomes = {
                    arm: latents[arm] + self._rng.gauss(0.0, self.payoff_noise) for arm in arms
                }
                self._trials.append(
                    _SchulzTrial(
                        round_number=round_number,
                        valid_actions=arms,
                        outcomes_by_action=outcomes,
                        show_round_header=trial_number == 0,
                    )
                )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(self._trials[0], None),
        )

    def _render_trial(self, trial: _SchulzTrial) -> str:
        if trial.show_round_header:
            return "You are playing round {}:".format(trial.round_number)
        return "You are playing round {}:".format(trial.round_number)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed SchulzFindingGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid option <<{}>>.".format(submitted), 0.0, False, True, info

        reward = trial.outcomes_by_action[submitted]
        self._points += reward
        feedback = "You press <<{}>> and get {} points.".format(submitted, reward)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            next_trial = self._trials[self._trial_idx]
            if next_trial.show_round_header:
                feedback = "{}\n\n{}".format(feedback, self._render_trial(next_trial))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[_SchulzTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None:
            info["round_number"] = trial.round_number
        return info
