"""Recorded-path Wilson two-arm slot machine."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import WilsonSlotTrial


class WilsonSlotRecordedEnv:
    """Exact-transition Wilson trials on a recorded game schedule."""

    EXPERIMENT_ID = "wilson2014humans/exp1.csv"

    def __init__(
        self,
        trials: Iterable[WilsonSlotTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[WilsonSlotTrial] = list(trials)
        if not self.trials:
            raise ValueError("WilsonSlotRecordedEnv requires at least one trial.")
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
            render_initial_observation(self.instruction, self._initial_observation()),
            self._info(None, None),
        )

    def _initial_observation(self) -> str:
        trial = self.trials[0]
        if trial.trial_type == "instructed":
            return trial.observation
        return trial.observation

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WilsonSlotRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid slot action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        if trial.trial_type == "instructed":
            feedback = "You are instructed to press {} and get {} points.".format(
                submitted, int(reward)
            )
        else:
            feedback = "You press <<{}>> and get {} points.".format(submitted, int(reward))
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            next_trial = self.trials[self._trial_idx]
            if next_trial.trial_type == "instructed":
                feedback = "{}\n\n{}".format(feedback, next_trial.observation)
            else:
                feedback = "{}\n\n{}".format(feedback, next_trial.observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[WilsonSlotTrial], selected_arm: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_arm": selected_arm,
        }
        if trial is not None:
            info["trial_type"] = trial.trial_type
            if self.include_human_ref:
                info["human_ref"] = trial.human_action
                info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
