"""Recorded-path four-arm bandit (Bahrami et al.)."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import BahramiFourArmTrial


class BahramiFourArmRecordedEnv:
    """Exact-transition four-arm bandit with pooled arm outcomes."""

    EXPERIMENT_ID = "bahrami2020four/exp.csv"

    def __init__(
        self,
        trials: Iterable[BahramiFourArmTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[BahramiFourArmTrial] = list(trials)
        if not self.trials:
            raise ValueError("BahramiFourArmRecordedEnv requires at least one trial.")
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
            render_initial_observation(self.instruction, "You press <<{}>>.".format(self.trials[0].valid_actions[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed BahramiFourArmRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid option <<{}>>.".format(submitted), 0.0, False, True, info

        reward = trial.outcomes_by_action[submitted]
        self._points += reward
        feedback = "You press <<{}>> and get {} points.".format(submitted, reward)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[BahramiFourArmTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
