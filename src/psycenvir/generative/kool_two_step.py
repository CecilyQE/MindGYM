"""Generative Kool et al. (2016) when-task exp2: spaceship then alien."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import KOOL_WHEN_EXP2_INSTRUCTION

KOOL_WHEN_EXP2_ID = "kool2016when/exp2.csv"

_PLANET_ALIENS = {
    "J": ("W", "K"),
    "T": ("I", "G"),
}


@dataclass(frozen=True)
class _SpaceshipPhase:
    ships: Tuple[str, str]
    planet_by_ship: Dict[str, str]
    treasure_by_alien: Dict[str, int]


@dataclass
class _KoolTwoStepDay:
    spaceship_phase: _SpaceshipPhase
    selected_ship: Optional[str] = None
    selected_planet: Optional[str] = None


class KoolWhenExp2GenerativeEnv:
    """Fresh two-step days: choose spaceship, then alien on the visited planet."""

    def __init__(
        self,
        experiment_id: str = KOOL_WHEN_EXP2_ID,
        n_days: int = 125,
        ship_planet_skew: float = 0.8,
        instruction: str = KOOL_WHEN_EXP2_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_days = n_days
        self.ship_planet_skew = ship_planet_skew
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._days: List[_KoolTwoStepDay] = []
        self._day_idx = 0
        self._treasure_total = 0
        self._done = False
        self._awaiting_alien = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        alien_means = {
            alien: self._rng.uniform(0.0, 3.0)
            for aliens in _PLANET_ALIENS.values()
            for alien in aliens
        }
        self._days = []
        for _ in range(self.n_days):
            ships = ("R", "U")
            if self._rng.random() < 0.5:
                ships = (ships[1], ships[0])
            planet_by_ship = {}
            for ship in ships:
                preferred = "J" if ship == "R" else "T"
                other = "T" if preferred == "J" else "J"
                planet = (
                    preferred
                    if self._rng.random() < self.ship_planet_skew
                    else other
                )
                planet_by_ship[ship] = planet
            treasure_by_alien = {
                alien: max(0, int(round(self._rng.gauss(alien_means[alien], 1.0))))
                for alien in alien_means
            }
            for alien in alien_means:
                alien_means[alien] += self._rng.gauss(0.0, 0.1)
            self._days.append(
                _KoolTwoStepDay(
                    spaceship_phase=_SpaceshipPhase(
                        ships=ships,
                        planet_by_ship=planet_by_ship,
                        treasure_by_alien=treasure_by_alien,
                    )
                )
            )
        self._day_idx = 0
        self._treasure_total = 0
        self._done = False
        self._awaiting_alien = False
        return (
            render_initial_observation(
                self.instruction, self._render_spaceship_prompt(self._days[0])
            ),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KoolWhenExp2GenerativeEnv; call reset().")
        submitted = normalize_action(action)
        day = self._days[self._day_idx]
        if not self._awaiting_alien:
            if submitted not in day.spaceship_phase.planet_by_ship:
                self._done = True
                info = self._info(day, None)
                info["invalid_action"] = submitted
                return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info
            day.selected_ship = submitted
            day.selected_planet = day.spaceship_phase.planet_by_ship[submitted]
            self._awaiting_alien = True
            feedback = (
                "You are presented with spaceships {} and {}. You press <<{}>>. "
                "You end up on planet {}. {}"
            ).format(
                day.spaceship_phase.ships[0],
                day.spaceship_phase.ships[1],
                submitted,
                day.selected_planet,
                self._render_alien_prompt(day),
            )
            return feedback, 0.0, False, False, self._info(day, submitted)

        aliens = _PLANET_ALIENS[day.selected_planet]
        if submitted not in aliens:
            self._done = True
            info = self._info(day, None)
            info["invalid_action"] = submitted
            return "Invalid alien action <<{}>>.".format(submitted), 0.0, False, True, info

        treasure = day.spaceship_phase.treasure_by_alien[submitted]
        self._treasure_total += treasure
        feedback = (
            "You are presented with spaceships {} and {}. You press <<{}>>. "
            "You end up on planet {}. You see alien {} and alien {}. "
            "You press <<{}>>. You find {} pieces of space treasure."
        ).format(
            day.spaceship_phase.ships[0],
            day.spaceship_phase.ships[1],
            day.selected_ship,
            day.selected_planet,
            aliens[0],
            aliens[1],
            submitted,
            treasure,
        )
        self._advance_day()
        return feedback, float(treasure), self._done, False, self._info(day, submitted)

    def _advance_day(self) -> None:
        self._day_idx += 1
        self._done = self._day_idx >= len(self._days)
        self._awaiting_alien = False

    @staticmethod
    def _render_spaceship_prompt(day: _KoolTwoStepDay) -> str:
        ships = day.spaceship_phase.ships
        return "You are presented with spaceships {} and {}.".format(ships[0], ships[1])

    @staticmethod
    def _render_alien_prompt(day: _KoolTwoStepDay) -> str:
        aliens = _PLANET_ALIENS[day.selected_planet]
        return "You see alien {} and alien {}.".format(aliens[0], aliens[1])

    def _info(self, day: Optional[_KoolTwoStepDay], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "treasure_total": self._treasure_total,
            "awaiting_alien": self._awaiting_alien,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if day is not None and self.include_human_ref and not self._awaiting_alien:
            info["planet_by_ship"] = dict(day.spaceship_phase.planet_by_ship)
            info["treasure_by_alien"] = dict(day.spaceship_phase.treasure_by_alien)
        return info
