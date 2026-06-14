"""Recorded-path Tomov castle multitask exploration."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import TomovCastleTrial


class TomovCastleRecordedEnv:
    """Exact-transition castle navigation on a recorded participant path."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[TomovCastleTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[TomovCastleTrial] = list(trials)
        if not self.trials:
            raise ValueError("TomovCastleRecordedEnv requires at least one trial.")
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

    @staticmethod
    def _format_feedback(
        trial: TomovCastleTrial, action: str, reward: float, use_recorded_resources: bool
    ) -> str:
        if use_recorded_resources:
            wood, stone, iron = trial.resource_amounts
            return (
                "You are in room {}. You press <<{}>> and you find {} wood, {} stone, "
                "and {} iron. You get {} points."
            ).format(trial.room_number, action, wood, stone, iron, reward)
        return "You press <<{}>> and you get {} points.".format(action, reward)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed TomovCastleRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid door action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = trial.outcomes_by_action[submitted]
        use_recorded = submitted == trial.human_action
        feedback = self._format_feedback(trial, submitted, reward, use_recorded)
        self._points += reward
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[TomovCastleTrial], selected_action: Optional[str]
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
            info["room_number"] = trial.room_number
            if self.include_human_ref:
                info["human_ref"] = trial.human_action
        return info
