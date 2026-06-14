"""Recorded-path Gershman competitive volatile bandit (exp2)."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION
from psycenvir.models import GershmanBanditTrial


class GershmanCompetitiveBanditRecordedEnv:
    """Recorded game schedule with pooled per-arm outcomes for counterfactuals."""

    EXPERIMENT_ID = "gershman2018deconstructing/exp2.csv"

    def __init__(
        self,
        trials: Iterable[GershmanBanditTrial],
        include_human_ref: bool = False,
        instruction: str = GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION,
    ) -> None:
        self.trials: List[GershmanBanditTrial] = list(trials)
        if not self.trials:
            raise ValueError("GershmanCompetitiveBanditRecordedEnv requires at least one trial.")
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
        first = self.trials[0]
        prefix = "Game {}:\n".format(first.game_number) if first.show_game_header else ""
        return (
            render_initial_observation(self.instruction, prefix),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "Cannot step a completed GershmanCompetitiveBanditRecordedEnv; call reset()."
            )
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid slot action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        feedback = "You press <<{}>> and get {} points.".format(submitted, int(reward))
        if trial.show_game_header:
            feedback = "Game {}:\n{}".format(trial.game_number, feedback)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done and self.trials[self._trial_idx].show_game_header:
            feedback = "{}\nGame {}:".format(feedback, self.trials[self._trial_idx].game_number)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[GershmanBanditTrial], selected_action: Optional[str]) -> Dict[str, Any]:
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
