"""Deterministic no-feedback memory-probe environment."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import RecentProbeTrial


class EnkaviRecentProbeEnv:
    """Old/new probe task where the source participant receives no trial feedback."""

    EXPERIMENT_ID = "enkavi2019recentprobes/exp1.csv"

    def __init__(
        self,
        trials: Iterable[RecentProbeTrial],
        valid_actions: Iterable[str],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[RecentProbeTrial] = list(trials)
        if not self.trials:
            raise ValueError("EnkaviRecentProbeEnv requires at least one trial.")
        self.valid_actions = set(valid_actions)
        if len(self.valid_actions) != 2:
            raise ValueError("EnkaviRecentProbeEnv requires two declared response keys.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_stimulus(self.trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviRecentProbeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid recent-probe action <<{}>>.".format(submitted), 0.0, False, True, info
        is_correct = submitted == trial.correct_action
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        observation = "" if self._done else self._render_stimulus(self.trials[self._trial_idx])
        return observation, 0.0, self._done, False, self._info(trial, is_correct)

    @staticmethod
    def _render_stimulus(trial: RecentProbeTrial) -> str:
        return "You are shown the letters {}. You see the letter {}.".format(
            list(trial.letters), trial.probe
        )

    def _info(
        self, trial: Optional[RecentProbeTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "feedback_causal": True,
            "feedback_present": False,
            "reward_defined": False,
            "objective_accuracy_defined": True,
            "fidelity_level": "exact_transition_no_feedback",
            "action_space_validated": True,
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None and self.include_human_ref:
            info["correct_action"] = trial.correct_action
            info["is_correct"] = is_correct
            info["human_ref"] = trial.human_action
        return info
