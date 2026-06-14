"""Generative Collsio et al. Caldionine judgment task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import COLLSI_JUDGMENT_INSTRUCTION

COLLSI_EXP1_ID = "collsiöö2023MCPL/exp1.csv"
COLLSI_EXP3_ID = "collsiöö2023MCPL/exp3.csv"
DEFAULT_CONCENTRATIONS = (
    "extremely low",
    "very low",
    "low",
    "somewhat low",
    "normal",
    "somewhat high",
    "high",
    "very high",
    "extremely high",
)
DEFAULT_PROGLADINE_LEVELS = ("very little", "a little", "average", "a lot", "very much")
DEFAULT_AMALYDINE_LEVELS = DEFAULT_PROGLADINE_LEVELS


@dataclass(frozen=True)
class _JudgmentTrial:
    progladine: str
    amalydine: str
    correct_action: str
    valid_actions: Tuple[str, ...]
    has_feedback: bool


class CollsiJudgmentGenerativeEnv:
    """Fresh judgment episodes with label feedback and optional silent test trials."""

    def __init__(
        self,
        experiment_id: str = COLLSI_EXP3_ID,
        n_feedback_trials: int = 120,
        n_silent_trials: int = 0,
        concentrations: Tuple[str, ...] = DEFAULT_CONCENTRATIONS,
        progladine_levels: Tuple[str, ...] = DEFAULT_PROGLADINE_LEVELS,
        amalydine_levels: Tuple[str, ...] = DEFAULT_AMALYDINE_LEVELS,
        instruction: str = COLLSI_JUDGMENT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_feedback_trials = int(calibration.get("n_feedback_trials", n_feedback_trials))
        self.n_silent_trials = int(calibration.get("n_silent_trials", n_silent_trials))
        self.concentrations = concentrations
        self.progladine_levels = progladine_levels
        self.amalydine_levels = amalydine_levels
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_JudgmentTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def _sample_lookup(self) -> Dict[Tuple[str, str], str]:
        lookup: Dict[Tuple[str, str], str] = {}
        for pro in self.progladine_levels:
            for ama in self.amalydine_levels:
                lookup[(pro, ama)] = self._rng.choice(self.concentrations)
        return lookup

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        lookup = self._sample_lookup()
        self._trials = []
        for _ in range(self.n_feedback_trials):
            pro = self._rng.choice(self.progladine_levels)
            ama = self._rng.choice(self.amalydine_levels)
            correct = lookup[(pro, ama)]
            self._trials.append(
                _JudgmentTrial(pro, ama, correct, self.concentrations, True)
            )
        for _ in range(self.n_silent_trials):
            pro = self._rng.choice(self.progladine_levels)
            ama = self._rng.choice(self.amalydine_levels)
            correct = lookup[(pro, ama)]
            self._trials.append(
                _JudgmentTrial(pro, ama, correct, self.concentrations, False)
            )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_stimulus(self._trials[0])),
            self._info(None, None),
        )

    @staticmethod
    def _render_stimulus(trial: _JudgmentTrial) -> str:
        return "Progladine: {}. Amalydine: {}.".format(trial.progladine, trial.amalydine)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed CollsiJudgmentGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid Caldionine response <<{}>>.".format(submitted), 0.0, False, True, info
        is_correct = submitted == trial.correct_action
        latent_reward = 1.0 if is_correct else 0.0
        if trial.has_feedback:
            reward = latent_reward
            self._points += reward
            evaluation = "correct" if is_correct else "incorrect"
            feedback = (
                "You say that the Caldionine concentration is <<{}>>. That is {}. "
                "The correct concentration of Caldionine is {}."
            ).format(submitted, evaluation, trial.correct_action)
        else:
            reward = 0.0
            feedback = "You say that the Caldionine concentration is <<{}>>.".format(submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{} {}".format(feedback, self._render_stimulus(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, is_correct, latent_reward)

    def _info(
        self,
        trial: Optional[_JudgmentTrial],
        is_correct: Optional[bool],
        latent_reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["has_feedback"] = trial.has_feedback
            info["correct_action"] = trial.correct_action
            if is_correct is not None:
                info["is_correct"] = is_correct
            if latent_reward is not None:
                info["latent_reward"] = latent_reward
        return info
