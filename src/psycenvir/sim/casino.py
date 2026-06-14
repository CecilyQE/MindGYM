"""Recorded-path casino bandit (Lefebvre et al.)."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import LefebvreCasinoTrial


class LefebvreCasinoRecordedEnv:
    """Exact-transition casino visits with pooled arm outcomes per casino."""

    EXPERIMENT_ID = "lefebvre2017behavioural/exp1.csv"

    def __init__(
        self,
        trials: Iterable[LefebvreCasinoTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[LefebvreCasinoTrial] = list(trials)
        if not self.trials:
            raise ValueError("LefebvreCasinoRecordedEnv requires at least one trial.")
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
            render_initial_observation(self.instruction, self._render_trial(self.trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed LefebvreCasinoRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid casino action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = trial.outcomes_by_action[submitted]
        self._points += reward
        feedback = (
            "You go to casino {}. You can choose between machines {} and {}. "
            "You press <<{}>> and receive {} points."
        ).format(
            trial.casino_id,
            trial.valid_actions[0],
            trial.valid_actions[1],
            submitted,
            reward,
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self.trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _render_trial(trial: LefebvreCasinoTrial) -> str:
        return (
            "You go to casino {}. You can choose between machines {} and {}."
        ).format(trial.casino_id, trial.valid_actions[0], trial.valid_actions[1])

    def _info(
        self, trial: Optional[LefebvreCasinoTrial], selected_arm: Optional[str]
    ) -> Dict[str, Any]:
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
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
