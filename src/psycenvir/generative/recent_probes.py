"""Generative Enkavi recent-probes (old/new) memory task."""

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import ENKAVI_RECENT_PROBE_INSTRUCTION

ENKAVI_RECENT_PROBES_EXP1_ID = "enkavi2019recentprobes/exp1.csv"
DEFAULT_PRESENT_KEY = "K"
DEFAULT_ABSENT_KEY = "D"


@dataclass(frozen=True)
class _ProbeTrial:
    letters: Tuple[str, ...]
    probe: str
    valid_actions: Tuple[str, str]
    correct_action: str


class EnkaviRecentProbesGenerativeEnv:
    """Fresh probe episodes: six-letter display then old/new judgment (no trial feedback)."""

    def __init__(
        self,
        experiment_id: str = ENKAVI_RECENT_PROBES_EXP1_ID,
        n_trials: int = 75,
        n_letters: int = 6,
        present_key: Optional[str] = None,
        absent_key: Optional[str] = None,
        instruction: str = ENKAVI_RECENT_PROBE_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_trials = int(calibration.get("n_trials", n_trials))
        self.n_letters = n_letters
        self.present_key = present_key or calibration.get("present_key", DEFAULT_PRESENT_KEY)
        self.absent_key = absent_key or calibration.get("absent_key", DEFAULT_ABSENT_KEY)
        if self.present_key == self.absent_key:
            raise ValueError("Recent-probes present and absent keys must differ.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_ProbeTrial] = []
        self._trial_idx = 0
        self._n_correct = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        alphabet = list(string.ascii_uppercase)
        self._trials = []
        for _ in range(self.n_trials):
            letters = tuple(self._rng.sample(alphabet, self.n_letters))
            probe = self._rng.choice(alphabet)
            correct = (
                self.present_key
                if probe in letters
                else self.absent_key
            )
            self._trials.append(
                _ProbeTrial(
                    letters=letters,
                    probe=probe,
                    valid_actions=(self.present_key, self.absent_key),
                    correct_action=correct,
                )
            )
        self._trial_idx = 0
        self._n_correct = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_stimulus(self._trials[0])),
            self._info(None, None),
        )

    @staticmethod
    def _render_stimulus(trial: _ProbeTrial) -> str:
        return "You are shown the letters {}. You see the letter {}.".format(
            list(trial.letters), trial.probe
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviRecentProbesGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid recent-probe action <<{}>>.".format(submitted), 0.0, False, True, info
        is_correct = submitted == trial.correct_action
        if is_correct:
            self._n_correct += 1
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        observation = "" if self._done else self._render_stimulus(self._trials[self._trial_idx])
        return observation, 0.0, self._done, False, self._info(trial, is_correct)

    def _info(
        self, trial: Optional[_ProbeTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "n_correct": self._n_correct,
            "feedback_causal": True,
            "feedback_present": False,
            "reward_defined": False,
            "objective_accuracy_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["correct_action"] = trial.correct_action
            if is_correct is not None:
                info["is_correct"] = is_correct
        return info
