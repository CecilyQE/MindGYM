"""Recorded-path Steingroever Iowa Gambling Task."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import SteingroeverIGTTrial


class SteingroeverIGTRecordedEnv:
    """Exact-transition IGT with pooled per-deck win/loss outcomes."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[SteingroeverIGTTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[SteingroeverIGTTrial] = list(trials)
        if not self.trials:
            raise ValueError("SteingroeverIGTRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._balance = 2000.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._balance = 2000.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed SteingroeverIGTRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid deck <<{}>>.".format(submitted), 0.0, False, True, info

        win, loss = trial.outcomes_by_action[submitted]
        reward = win - loss
        self._balance += reward
        feedback = "You press <<{}>>. You win {}$ and lose {}$.".format(
            submitted, win, loss
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[SteingroeverIGTTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "balance": self._balance,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
        return info
