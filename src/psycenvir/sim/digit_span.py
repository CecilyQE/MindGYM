"""Recorded-path Enkavi digit-span recall."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import DigitSpanRecallTrial


class EnkaviDigitSpanRecordedEnv:
    """Exact key-by-key digit recall with per-press correctness scoring."""

    EXPERIMENT_ID = "enkavi2019digitspan/exp1.csv"

    def __init__(
        self,
        trials: Iterable[DigitSpanRecallTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[DigitSpanRecallTrial] = list(trials)
        if not self.trials:
            raise ValueError("EnkaviDigitSpanRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._done = False
        first = self.trials[0]
        observation = render_initial_observation(self.instruction, first.observation)
        return observation, self._info(None, None)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviDigitSpanRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid digit-span key <<{}>>.".format(submitted), 0.0, False, True, info

        is_correct = submitted == trial.correct_action
        reward = 1.0 if is_correct else 0.0
        feedback = "You press <<{}>>.".format(submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        observation = ""
        if not self._done:
            next_trial = self.trials[self._trial_idx]
            if next_trial.observation:
                observation = next_trial.observation
        return observation, reward, self._done, False, self._info(trial, is_correct)

    def _info(
        self, trial: Optional[DigitSpanRecallTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "feedback_causal": True,
            "feedback_present": True,
            "reward_defined": True,
            "objective_accuracy_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["span_index"] = trial.span_index
            info["span_length"] = trial.span_length
            if self.include_human_ref:
                info["correct_action"] = trial.correct_action
                info["is_correct"] = is_correct
                info["human_ref"] = trial.human_action
        return info
