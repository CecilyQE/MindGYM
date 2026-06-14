"""Recorded-path Hilbig product choice."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import HilbigProductTrial


class HilbigProductRecordedEnv:
    """Exact-transition normative product choice on recorded rating pairs."""

    EXPERIMENT_ID = "hilbig2014generalized/exp1.csv"

    def __init__(
        self,
        trials: Iterable[HilbigProductTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[HilbigProductTrial] = list(trials)
        if not self.trials:
            raise ValueError("HilbigProductRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._correct_count = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._correct_count = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed HilbigProductRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid product action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = 1.0 if submitted == trial.correct_action else 0.0
        self._correct_count += reward
        feedback = "{}\nYou press <<{}>>.".format(trial.observation, submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[HilbigProductTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "correct_count": self._correct_count,
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
