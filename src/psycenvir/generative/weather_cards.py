"""Generative weather-forecasting card task (Speekenbrink et al. style)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import SPEEKENBRINK_WEATHER_INSTRUCTION

SPEEKENBRINK_EXPERIMENT_ID = "speekenbrink2008learning/exp1.csv"
DEFAULT_WEATHER_ACTIONS = ("E", "J")


@dataclass(frozen=True)
class _WeatherTrial:
    cards_display: str
    weather: str
    correct_action: str
    valid_actions: Tuple[str, str] = DEFAULT_WEATHER_ACTIONS


class SpeekenbrinkWeatherGenerativeEnv:
    """Fresh card draws with latent rainy/fine weather."""

    def __init__(
        self,
        experiment_id: str = SPEEKENBRINK_EXPERIMENT_ID,
        n_trials: int = 200,
        card_ids: Tuple[int, ...] = (1, 2, 3, 4),
        instruction: str = SPEEKENBRINK_WEATHER_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.card_ids = card_ids
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_WeatherTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        card_probs = {card_id: self._rng.uniform(0.2, 0.8) for card_id in self.card_ids}
        self._trials = []
        for _ in range(self.n_trials):
            n_cards = self._rng.randint(1, min(3, len(self.card_ids)))
            cards = sorted(self._rng.sample(self.card_ids, n_cards))
            cards_display = ", ".join("card {}".format(card_id) for card_id in cards)
            mean_prob = sum(card_probs[card_id] for card_id in cards) / len(cards)
            weather = "rainy" if self._rng.random() < mean_prob else "fine"
            correct_action = "E" if weather == "rainy" else "J"
            self._trials.append(
                _WeatherTrial(
                    cards_display=cards_display,
                    weather=weather,
                    correct_action=correct_action,
                )
            )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed SpeekenbrinkWeatherGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid forecast action <<{}>>.".format(submitted), 0.0, False, True, info

        is_correct = submitted == trial.correct_action
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
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _render_trial(trial: _WeatherTrial) -> str:
        return "You are seeing the following: {}.".format(trial.cards_display)

    def _info(self, trial: Optional[_WeatherTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["weather"] = trial.weather
            info["correct_action"] = trial.correct_action
        return info
