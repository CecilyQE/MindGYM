"""Generative Hilbig generalized weighted-additive product choice."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import HILBIG_PRODUCT_INSTRUCTION

HILBIG_EXPERIMENT_ID = "hilbig2014generalized/exp1.csv"
EXPERT_WEIGHTS = (0.9, 0.8, 0.7, 0.6)
DEFAULT_ACTIONS = ("A", "R")


@dataclass(frozen=True)
class _ProductTrial:
    observation: str
    valid_actions: Tuple[str, str]
    correct_action: str
    ratings_a: Tuple[int, ...]
    ratings_b: Tuple[int, ...]


def hilbig_correct_action(ratings_a: Tuple[int, ...], ratings_b: Tuple[int, ...]) -> str:
    score_a = sum(rating * weight for rating, weight in zip(ratings_a, EXPERT_WEIGHTS))
    score_b = sum(rating * weight for rating, weight in zip(ratings_b, EXPERT_WEIGHTS))
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "R"
    return "A"


class HilbigProductGenerativeEnv:
    """Fresh expert-rating product pairs with normative A/R choice."""

    def __init__(
        self,
        experiment_id: str = HILBIG_EXPERIMENT_ID,
        n_trials: int = 96,
        valid_actions: Tuple[str, str] = DEFAULT_ACTIONS,
        instruction: str = HILBIG_PRODUCT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.valid_actions = valid_actions
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_ProductTrial] = []
        self._trial_idx = 0
        self._correct_count = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        for _ in range(self.n_trials):
            ratings_a = tuple(self._rng.randint(0, 1) for _ in range(4))
            ratings_b = tuple(self._rng.randint(0, 1) for _ in range(4))
            correct = hilbig_correct_action(ratings_a, ratings_b)
            if correct not in self.valid_actions:
                correct = self.valid_actions[0]
            self._trials.append(
                _ProductTrial(
                    observation=self._format_observation(ratings_a, ratings_b),
                    valid_actions=self.valid_actions,
                    correct_action=correct,
                    ratings_a=ratings_a,
                    ratings_b=ratings_b,
                )
            )
        self._trial_idx = 0
        self._correct_count = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._trials[0].observation),
            self._info(self._trials[0], None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed HilbigProductGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid product action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = 1.0 if submitted == trial.correct_action else 0.0
        self._correct_count += reward
        feedback = "{}\nYou press <<{}>>.".format(trial.observation, submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _format_observation(ratings_a: Tuple[int, ...], ratings_b: Tuple[int, ...]) -> str:
        return (
            "Product A ratings: {}. Product R ratings: {}."
        ).format(list(ratings_a), list(ratings_b))

    def _info(self, trial: Optional[_ProductTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "correct_count": self._correct_count,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["correct_action"] = trial.correct_action
        return info
