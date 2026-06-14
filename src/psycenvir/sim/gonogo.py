"""Recorded-path Enkavi go/no-go task."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.errors import InvalidActionError
from psycenvir.models import GONOGO_NO_PRESS, GonogoTrial


class EnkaviGonogoRecordedEnv:
    """Exact colour1/colour2 trials with go-key or withhold responses."""

    def __init__(
        self,
        trials: Iterable[GonogoTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[GonogoTrial] = list(trials)
        if not self.trials:
            raise ValueError("EnkaviGonogoRecordedEnv requires at least one trial.")
        self.go_key = self.trials[0].go_key
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self.trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviGonogoRecordedEnv; call reset().")
        submitted = self._coerce_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in (trial.go_key, None):
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = action
            return "Invalid go/no-go action <<{}>>.".format(action), 0.0, False, True, info

        is_correct = self._is_correct(trial, submitted)
        reward = 1.0 if is_correct else 0.0
        feedback = self._format_feedback(trial.stimulus, submitted, trial.human_rt_ms)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        observation = ""
        if not self._done:
            observation = self._render_trial(self.trials[self._trial_idx])
        return observation, reward, self._done, False, self._info(trial, is_correct)

    def _coerce_action(self, action: str) -> Optional[str]:
        if isinstance(action, str) and action.strip().upper() == GONOGO_NO_PRESS:
            return None
        try:
            return normalize_action(action)
        except InvalidActionError:
            if not action.strip():
                return None
            raise

    @staticmethod
    def _is_correct(trial: GonogoTrial, submitted: Optional[str]) -> bool:
        if trial.stimulus == "colour1":
            return submitted == trial.go_key
        return submitted is None

    @staticmethod
    def _format_feedback(
        stimulus: str, submitted: Optional[str], rt_ms: Optional[float]
    ) -> str:
        if submitted is None:
            return "You see {} and press nothing.".format(stimulus)
        rt = rt_ms if rt_ms is not None else 400.0
        return "You see {} and press <<{}>> in {}ms.".format(stimulus, submitted, rt)

    @staticmethod
    def _render_trial(trial: GonogoTrial) -> str:
        return "You see {}.".format(trial.stimulus)

    def _info(
        self, trial: Optional[GonogoTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": "enkavi2019gonogo/exp1.csv",
            "trial_idx": self._trial_idx,
            "go_key": self.go_key,
            "feedback_causal": True,
            "feedback_present": True,
            "reward_defined": True,
            "objective_accuracy_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["stimulus"] = trial.stimulus
            info["is_practice"] = trial.is_practice
            if self.include_human_ref:
                info["human_ref"] = trial.human_key or GONOGO_NO_PRESS
                info["is_correct"] = is_correct
        return info
