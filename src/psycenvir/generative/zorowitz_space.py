"""Generative Zorowitz-style two-step space treasure task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

ZOROWITZ_DATA_EXP1_ID = "zorowitz2023data/exp1.csv"
ZOROWITZ_INSTRUCTION = (
    "You are participating in a space treasure game.\n"
    "You will choose between two rocket ships to visit one of two alien planets.\n"
    "Each planet has two aliens. When you trade with an alien, it will either give "
    "you treasure or junk.\n"
    "Each rocket ship has a planet it will fly to most of the time, but sometimes "
    "it will take you to the other planet.\n"
    "How likely an alien is to give treasure changes slowly over time."
)

_PLANET_ALIENS = {
    "blue": ("D", "R"),
    "red": ("G", "V"),
}


@dataclass
class _ZorowitzTrial:
    ships: Tuple[str, str]
    common_planet_by_ship: Dict[str, str]
    treasure_probs: Dict[str, float]
    selected_ship: Optional[str] = None
    selected_planet: Optional[str] = None


class ZorowitzSpaceTreasureGenerativeEnv:
    """Fresh two-step task with stochastic ship transitions and drifting alien rewards."""

    def __init__(
        self,
        n_trials: int = 201,
        ships: Tuple[str, str] = ("S", "C"),
        transition_probability: float = 0.8,
        drift_sd: float = 0.035,
        instruction: str = ZOROWITZ_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = ZOROWITZ_DATA_EXP1_ID
        self.n_trials = n_trials
        self.ships = tuple(ship.upper() for ship in ships)
        self.transition_probability = transition_probability
        self.drift_sd = drift_sd
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_ZorowitzTrial] = []
        self._trial_idx = 0
        self._awaiting_alien = False
        self._treasure_total = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        treasure_probs = {
            alien: self._rng.uniform(0.25, 0.75)
            for aliens in _PLANET_ALIENS.values()
            for alien in aliens
        }
        common = {self.ships[0]: "blue", self.ships[1]: "red"}
        self._trials = []
        for _ in range(self.n_trials):
            self._trials.append(_ZorowitzTrial(self.ships, dict(common), dict(treasure_probs)))
            for alien, probability in list(treasure_probs.items()):
                treasure_probs[alien] = min(0.95, max(0.05, probability + self._rng.gauss(0.0, self.drift_sd)))
        self._trial_idx = 0
        self._awaiting_alien = False
        self._treasure_total = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_ship_prompt(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed ZorowitzSpaceTreasureGenerativeEnv; call reset().")
        submitted = normalize_action(action).upper()
        trial = self._trials[self._trial_idx]
        if not self._awaiting_alien:
            if submitted not in trial.ships:
                self._done = True
                info = self._info(trial, None)
                info["invalid_action"] = submitted
                return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info
            common_planet = trial.common_planet_by_ship[submitted]
            other_planet = "red" if common_planet == "blue" else "blue"
            planet = common_planet if self._rng.random() < self.transition_probability else other_planet
            trial.selected_ship = submitted
            trial.selected_planet = planet
            self._awaiting_alien = True
            feedback = (
                "You are presented with two spaceships called {} and {}. You press <<{}>>. "
                "You end up on the {} planet. {}"
            ).format(trial.ships[0], trial.ships[1], submitted, planet, self._render_alien_prompt(planet))
            return feedback, 0.0, False, False, self._info(trial, submitted)

        aliens = _PLANET_ALIENS[trial.selected_planet]
        if submitted not in aliens:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid alien action <<{}>>.".format(submitted), 0.0, False, True, info
        got_treasure = self._rng.random() < trial.treasure_probs[submitted]
        reward = 1.0 if got_treasure else 0.0
        self._treasure_total += int(reward)
        feedback = (
            "You are presented with two spaceships called {} and {}. You press <<{}>>. "
            "You end up on the {} planet. You see a {} alien named {} and a {} alien named {}. "
            "You press <<{}>>. You find {}."
        ).format(
            trial.ships[0],
            trial.ships[1],
            trial.selected_ship,
            trial.selected_planet,
            trial.selected_planet,
            aliens[0],
            trial.selected_planet,
            aliens[1],
            submitted,
            "treasure" if got_treasure else "junk",
        )
        self._trial_idx += 1
        self._awaiting_alien = False
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_ship_prompt(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _render_ship_prompt(trial: _ZorowitzTrial) -> str:
        return "You are presented with two spaceships called {} and {}.".format(
            trial.ships[0], trial.ships[1]
        )

    @staticmethod
    def _render_alien_prompt(planet: str) -> str:
        aliens = _PLANET_ALIENS[planet]
        return "You see a {} alien named {} and a {} alien named {}.".format(
            planet, aliens[0], planet, aliens[1]
        )

    def _info(self, trial: Optional[_ZorowitzTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "awaiting_alien": self._awaiting_alien,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["treasure_probs"] = dict(trial.treasure_probs)
            info["common_planet_by_ship"] = dict(trial.common_planet_by_ship)
        return info
