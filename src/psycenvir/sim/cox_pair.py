"""Recorded-path Cox pair-recognition task."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import CoxPairRecognitionTrial


class CoxPairRecognitionRecordedEnv:
    """Exact-transition studied/new pair recognition with correctness reward."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[CoxPairRecognitionTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[CoxPairRecognitionTrial] = list(trials)
        if not self.trials:
            raise ValueError("CoxPairRecognitionRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed CoxPairRecognitionRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid pair action <<{}>>.".format(submitted), 0.0, False, True, info

        is_correct = submitted == trial.correct_action
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        feedback = "{} You press <<{}>>.".format(trial.observation, submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, is_correct)

    def _info(
        self, trial: Optional[CoxPairRecognitionTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["correct_action"] = trial.correct_action
            if is_correct is not None:
                info["is_correct"] = is_correct
            if self.include_human_ref:
                info["human_ref"] = trial.human_action
        return info
