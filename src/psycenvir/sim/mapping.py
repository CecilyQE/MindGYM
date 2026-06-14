"""Stimulus-response mapping simulation for Gershman et al. transcripts."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import GershmanMappingTrial


class GershmanMappingEnv:
    """Exact-transition mapping feedback on a recorded stimulus schedule."""

    EXPERIMENT_ID = "gershman2020reward/exp1.csv"

    def __init__(
        self,
        trials: Iterable[GershmanMappingTrial],
        valid_actions: Iterable[str],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[GershmanMappingTrial] = list(trials)
        if not self.trials:
            raise ValueError("GershmanMappingEnv requires at least one trial.")
        self.valid_actions = tuple(valid_actions)
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
            raise RuntimeError("Cannot step a completed GershmanMappingEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid response <<{}>>.".format(submitted), 0.0, False, True, info

        reward = 1.0 if submitted == trial.correct_action else 0.0
        self._points += reward
        feedback = "You see stimulus {}. You press <<{}>> and get {} points.".format(
            trial.stimulus_id,
            submitted,
            int(reward),
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self.trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: GershmanMappingTrial) -> str:
        if trial.show_game_header:
            return (
                "Game {}:\nThere are 6 different stimuli.\nYou see stimulus {}."
            ).format(trial.game_number, trial.stimulus_id)
        return "You see stimulus {}.".format(trial.stimulus_id)

    def _info(
        self, trial: Optional[GershmanMappingTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
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
            info["correct_action"] = trial.correct_action
        return info
