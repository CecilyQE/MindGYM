"""Weather-forecast simulation for Speekenbrink et al. transcripts."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import SpeekenbrinkWeatherTrial


class SpeekenbrinkWeatherEnv:
    """Exact-transition correctness feedback on recorded card draws."""

    EXPERIMENT_ID = "speekenbrink2008learning/exp1.csv"
    VALID_ACTIONS = ("E", "J")

    def __init__(
        self,
        trials: Iterable[SpeekenbrinkWeatherTrial],
        valid_actions: Tuple[str, str] = VALID_ACTIONS,
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[SpeekenbrinkWeatherTrial] = list(trials)
        if not self.trials:
            raise ValueError("SpeekenbrinkWeatherEnv requires at least one trial.")
        if len(valid_actions) != 2 or len(set(valid_actions)) != 2:
            raise ValueError("SpeekenbrinkWeatherEnv requires rainy and fine action keys.")
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
            raise RuntimeError("Cannot step a completed SpeekenbrinkWeatherEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid forecast action <<{}>>.".format(submitted), 0.0, False, True, info

        correct_action = self.valid_actions[0] if trial.weather == "rainy" else self.valid_actions[1]
        is_correct = submitted == correct_action
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        if is_correct:
            correctness = "correct, the weather is indeed {}".format(trial.weather)
        else:
            correctness = "wrong, the weather is {}".format(trial.weather)
        feedback = (
            "You are seeing the following: {}. You press <<{}>>. You are {}."
        ).format(trial.cards_display, submitted, correctness)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self.trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _render_trial(trial: SpeekenbrinkWeatherTrial) -> str:
        return "You are seeing the following: {}.".format(trial.cards_display)

    def _info(
        self, trial: Optional[SpeekenbrinkWeatherTrial], selected_action: Optional[str]
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
            info["weather"] = trial.weather
        return info
