"""Recorded-path Schulz finding multi-arm bandit."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import SchulzFindingTrial


class SchulzFindingRecordedEnv:
    """Exact-transition multi-arm bandit with pooled per-round arm outcomes."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[SchulzFindingTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[SchulzFindingTrial] = list(trials)
        if not self.trials:
            raise ValueError("SchulzFindingRecordedEnv requires at least one trial.")
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
            render_initial_observation(self.instruction, self._trial_observation(self.trials[0])),
            self._info(None, None),
        )

    @staticmethod
    def _trial_observation(trial: SchulzFindingTrial) -> str:
        if trial.show_round_header:
            return "You are playing round {}:".format(trial.round_number)
        return "You are playing round {}:".format(trial.round_number)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed SchulzFindingRecordedEnv; call reset().")
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
        if not self._done:
            next_trial = self.trials[self._trial_idx]
            if next_trial.show_round_header:
                feedback = "{}\n\n{}".format(feedback, self._trial_observation(next_trial))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[SchulzFindingTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None:
            info["round_number"] = trial.round_number
            if self.include_human_ref:
                info["human_ref"] = trial.human_action
        return info
