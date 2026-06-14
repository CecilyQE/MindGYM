"""Generative Krueger et al. identifying gambles task (choice-only rounds)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import KRUEGER_IDENTIFYING_INSTRUCTION

KRUEGER_IDENTIFYING_EXP1_ID = "krueger2022identifying/exp1.csv"
DEFAULT_GAMBLES = ("Q", "N", "E", "S", "H", "K")
DEFAULT_COLORS = ("pink", "red", "black", "maroon")


@dataclass(frozen=True)
class _KruegerRound:
    observation: str
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, Dict[str, float]]
    chosen_color: str


class KruegerIdentifyingGenerativeEnv:
    """Fresh rounds with staged check/query then choose actions."""

    def __init__(
        self,
        experiment_id: str = KRUEGER_IDENTIFYING_EXP1_ID,
        n_rounds: int = 40,
        gambles: Tuple[str, ...] = DEFAULT_GAMBLES,
        colors: Tuple[str, ...] = DEFAULT_COLORS,
        instruction: str = KRUEGER_IDENTIFYING_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_rounds = int(calibration.get("n_rounds", n_rounds))
        self.gambles = gambles
        self.colors = colors
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._rounds: List[_KruegerRound] = []
        self._trial_idx = 0
        self._pending_gamble: Optional[str] = None
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._rounds = []
        for _ in range(self.n_rounds):
            counts = self._sample_color_counts()
            chosen_color = self._weighted_color(counts)
            payoffs = {
                gamble: {
                    color: float(self._rng.randint(-200, 200)) for color in self.colors
                }
                for gamble in self.gambles
            }
            observation = "A new round begins.\n{}".format(self._format_counts(counts))
            self._rounds.append(
                _KruegerRound(
                    observation=observation,
                    valid_actions=self.gambles,
                    outcomes_by_action=payoffs,
                    chosen_color=chosen_color,
                )
            )
        self._trial_idx = 0
        self._pending_gamble = None
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._rounds[0].observation),
            self._info(None, None),
        )

    def _sample_color_counts(self) -> Dict[str, int]:
        remaining = 100
        counts: Dict[str, int] = {}
        for index, color in enumerate(self.colors):
            if index == len(self.colors) - 1:
                counts[color] = remaining
            else:
                value = self._rng.randint(0, remaining)
                counts[color] = value
                remaining -= value
        if sum(counts.values()) == 0:
            counts[self._rng.choice(self.colors)] = 100
        return counts

    @staticmethod
    def _format_counts(counts: Dict[str, int]) -> str:
        items = list(counts.items())
        parts = ["{} {} balls".format(count, color) for color, count in items]
        if len(parts) == 1:
            return "There are {}.".format(parts[0])
        return "There are " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

    def _weighted_color(self, counts: Dict[str, int]) -> str:
        total = sum(counts.values())
        threshold = self._rng.randint(1, total)
        cumulative = 0
        for color, count in counts.items():
            cumulative += count
            if threshold <= cumulative:
                return color
        return self.colors[-1]

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KruegerIdentifyingGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        submitted_key = submitted.upper()
        round_state = self._rounds[self._trial_idx]
        if self._pending_gamble is None:
            if submitted_key not in round_state.valid_actions:
                self._done = True
                info = self._info(round_state, None)
                info["invalid_action"] = submitted
                return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info
            self._pending_gamble = submitted_key
            feedback = (
                "You press <<{}>>. Type a ball color to check this gamble, or type <<stop>> "
                "to choose it."
            ).format(submitted_key)
            return feedback, 0.0, False, False, self._info(round_state, submitted_key)

        pending = self._pending_gamble
        stop_action = "STOP"
        color_lookup = {color.upper(): color for color in self.colors}
        if submitted_key != stop_action and submitted_key not in color_lookup:
            self._done = True
            info = self._info(round_state, pending)
            info["invalid_action"] = submitted
            return "Invalid follow-up action <<{}>>.".format(submitted), 0.0, False, True, info

        if submitted_key in color_lookup:
            color = color_lookup[submitted_key]
            payoff = round_state.outcomes_by_action[pending][color]
            reward = -4.0
            self._points += reward
            self._pending_gamble = None
            feedback = (
                "You press <<{}>> and then type <<{}>>. The payoff for this combination "
                "would be {} points."
            ).format(pending, color, int(payoff))
            return feedback, reward, False, False, self._info(round_state, pending)

        reward = float(round_state.outcomes_by_action[pending][round_state.chosen_color])
        self._points += reward
        feedback = (
            "{} You press <<{}>> and then type <<stop>>. A {} ball is chosen, and you earn {} points."
        ).format(round_state.observation, pending, round_state.chosen_color, int(reward))
        self._trial_idx += 1
        self._pending_gamble = None
        self._done = self._trial_idx >= len(self._rounds)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._rounds[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(round_state, pending)

    def _info(
        self, round_state: Optional[_KruegerRound], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
            "pending_gamble": self._pending_gamble,
            "chosen_color": None if round_state is None else round_state.chosen_color,
        }
