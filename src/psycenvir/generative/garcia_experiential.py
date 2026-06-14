"""Generative Garcia experiential learning task (three-part bandit)."""

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import GARCIA_EXPERIENTIAL_INSTRUCTION

GARCIA_EXPERIENTIAL_IDS = tuple(
    "garcia2023experiential/exp{}.csv".format(index) for index in range(1, 5)
)
GARCIA_EXPERIENTIAL_EXP1_ID = GARCIA_EXPERIENTIAL_IDS[0]
DEFAULT_PART_TRIALS = (70, 70, 68)


@dataclass(frozen=True)
class _GarciaTrial:
    observation: str
    option_a: str
    option_b: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    part_number: int
    described_option: Optional[str] = None


class GarciaExperientialGenerativeEnv:
    """Fresh three-part episodes with +1/-1 bandit feedback and counterfactual sentences."""

    def __init__(
        self,
        experiment_id: str = GARCIA_EXPERIENTIAL_EXP1_ID,
        part_trial_counts: Tuple[int, int, int] = DEFAULT_PART_TRIALS,
        instruction: str = GARCIA_EXPERIENTIAL_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        counts = calibration.get("part_trial_counts", part_trial_counts)
        self.part_trial_counts = tuple(int(value) for value in counts)
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_GarciaTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def _sample_pair(self, part_number: int) -> _GarciaTrial:
        letters = self._rng.sample(string.ascii_uppercase, 2)
        option_a, option_b = letters[0], letters[1]
        better = self._rng.choice([option_a, option_b])
        outcomes = {
            option_a: 1.0 if better == option_a else -1.0,
            option_b: 1.0 if better == option_b else -1.0,
        }
        if part_number == 2 and self._rng.random() < 0.35:
            described = (
                "Option {}: 70% chance to win 1.0 points and 30% chance to lose 1.0 points."
            ).format(option_b)
            observation = (
                "You can choose between option {} and option {}. {}"
            ).format(option_a, option_b, described)
            return _GarciaTrial(
                observation, option_a, option_b, (option_a, option_b), outcomes, part_number, described
            )
        observation = "You can choose between option {} and option {}.".format(option_a, option_b)
        return _GarciaTrial(
            observation, option_a, option_b, (option_a, option_b), outcomes, part_number
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        for part_number, count in enumerate(self.part_trial_counts, start=1):
            for _ in range(count):
                self._trials.append(self._sample_pair(part_number))
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed GarciaExperientialGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid Garcia action <<{}>>.".format(submitted), 0.0, False, True, info
        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        alt = trial.option_b if submitted == trial.option_a else trial.option_a
        alt_reward = float(trial.outcomes_by_action[alt])
        feedback = (
            "{} You press <<{}>> and get {} points. You would have gotten {} points had you "
            "chosen option {} instead."
        ).format(
            trial.observation,
            submitted,
            reward,
            alt_reward,
            alt,
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[_GarciaTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None:
            info["part_number"] = trial.part_number
        return info
