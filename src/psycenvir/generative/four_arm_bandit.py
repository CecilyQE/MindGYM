"""Generative four-arm bandit (Bahrami et al., 2020 style)."""

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import BAHRAMI_FOUR_ARM_INSTRUCTION

BAHRAMI_EXPERIMENT_ID = "bahrami2020four/exp.csv"
DEFAULT_ARMS = ("L", "G", "O", "U")


@dataclass(frozen=True)
class _FourArmTrial:
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, float]


class BahramiFourArmGenerativeEnv:
    """Fresh four-option bandit with independent Gaussian-like payoffs."""

    def __init__(
        self,
        experiment_id: str = BAHRAMI_EXPERIMENT_ID,
        n_trials: int = 148,
        n_arms: int = 4,
        outcome_mean: float = 60.0,
        outcome_spread: float = 25.0,
        instruction: str = BAHRAMI_FOUR_ARM_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("BahramiFourArmGenerativeEnv requires at least two arms.")
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.n_arms = n_arms
        self.outcome_mean = outcome_mean
        self.outcome_spread = outcome_spread
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_FourArmTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        labels = tuple(self._rng.sample(list(string.ascii_uppercase), self.n_arms))
        arm_means = {
            label: self._rng.uniform(self.outcome_mean - self.outcome_spread, self.outcome_mean + self.outcome_spread)
            for label in labels
        }
        self._trials = []
        for _ in range(self.n_trials):
            outcomes = {
                label: round(self._rng.gauss(arm_means[label], self.outcome_spread / 3.0), 1)
                for label in labels
            }
            self._trials.append(_FourArmTrial(valid_actions=labels, outcomes_by_action=outcomes))
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        prompt = "You press <<{}>>.".format(labels[0])
        return (
            render_initial_observation(self.instruction, prompt),
            self._info(self._trials[0], None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed BahramiFourArmGenerativeEnv; call reset().")
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
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[_FourArmTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
