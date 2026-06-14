"""Recorded-path Tomov subway navigation."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import TomovSubwayTrial


class TomovSubwayRecordedEnv:
    """Exact-transition subway navigation on a recorded participant path."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[TomovSubwayTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[TomovSubwayTrial] = list(trials)
        if not self.trials:
            raise ValueError("TomovSubwayRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._successful_rounds = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._successful_rounds = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed TomovSubwayRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid subway action <<{}>>.".format(submitted), 0.0, False, True, info

        if submitted != trial.human_action:
            self._done = True
            info = self._info(trial, None)
            info["unsupported_counterfactual"] = "recorded_path_only"
            return (
                "Counterfactual subway moves are not supported on the recorded path.",
                0.0,
                False,
                True,
                info,
            )

        reward = 1.0 if trial.completes_round else 0.0
        if trial.completes_round:
            self._successful_rounds += 1
        feedback = "You press <<{}>>.".format(submitted)
        if trial.completes_round:
            feedback = "{}\nYou are successful.".format(feedback)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[TomovSubwayTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "successful_rounds": self._successful_rounds,
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
