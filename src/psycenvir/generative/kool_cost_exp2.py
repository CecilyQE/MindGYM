"""Generative Kool et al. (2017) cost-task exp2: spaceship, alien, optional multiplier."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import KOOL_COST_EXP2_INSTRUCTION

KOOL_COST_EXP2_ID = "kool2017cost/exp2.csv"


@dataclass(frozen=True)
class _SpaceshipPhase:
    ships: Tuple[str, str]
    planet_by_ship: Dict[str, str]
    aliens_by_planet: Dict[str, Tuple[str, str]]
    treasure_by_alien: Dict[str, float]
    multiplier: int


@dataclass
class _KoolCostExp2Day:
    spaceship_phase: _SpaceshipPhase
    selected_ship: Optional[str] = None
    selected_planet: Optional[str] = None


class KoolCostExp2GenerativeEnv:
    """Fresh two-step days with drifting mines and occasional 5x multipliers."""

    def __init__(
        self,
        experiment_id: str = KOOL_COST_EXP2_ID,
        n_days: int = 200,
        multiplier_value: int = 5,
        multiplier_probability: float = 0.5,
        ship_planet_skew: float = 0.8,
        instruction: Optional[str] = None,
        session_topologies: Optional[List[Dict[str, Any]]] = None,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_days = n_days
        self.multiplier_value = multiplier_value
        self.multiplier_probability = multiplier_probability
        self.ship_planet_skew = ship_planet_skew
        self._session_topologies = list(
            session_topologies or calibration.get("session_topologies") or []
        )
        if not self._session_topologies:
            raise ValueError(
                "KoolCostExp2GenerativeEnv requires transcript-derived session_topologies."
            )
        self.instruction = instruction or KOOL_COST_EXP2_INSTRUCTION
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._days: List[_KoolCostExp2Day] = []
        self._day_idx = 0
        self._treasure_total = 0.0
        self._done = False
        self._awaiting_alien = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)

        topology = self._rng.choice(self._session_topologies)
        ships = tuple(topology["ships"])
        aliens_by_planet = dict(topology["aliens_by_planet"])
        planet_by_ship = dict(topology["planet_by_ship"])
        if topology.get("instruction"):
            self.instruction = topology["instruction"]
        alien_means = {
            alien: self._rng.uniform(0.0, 3.0)
            for aliens in aliens_by_planet.values()
            for alien in aliens
        }

        self._days = []
        for _ in range(self.n_days):
            if self._rng.random() < 0.5:
                ships = (ships[1], ships[0])
            day_planet_by_ship = dict(planet_by_ship)
            multiplier = (
                self.multiplier_value
                if self._rng.random() < self.multiplier_probability
                else 1
            )
            treasure_by_alien = {
                alien: max(0.0, self._rng.gauss(alien_means[alien], 1.0))
                for alien in alien_means
            }
            for alien in alien_means:
                alien_means[alien] += self._rng.gauss(0.0, 0.1)

            self._days.append(
                _KoolCostExp2Day(
                    spaceship_phase=_SpaceshipPhase(
                        ships=ships,
                        planet_by_ship=day_planet_by_ship,
                        aliens_by_planet=aliens_by_planet,
                        treasure_by_alien=treasure_by_alien,
                        multiplier=multiplier,
                    )
                )
            )

        self._day_idx = 0
        self._treasure_total = 0.0
        self._done = False
        self._awaiting_alien = False
        return (
            render_initial_observation(
                self.instruction, self._render_spaceship_prompt(self._days[0])
            ),
            self._info(None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KoolCostExp2GenerativeEnv; call reset().")
        submitted = normalize_action(action)
        day = self._days[self._day_idx]
        phase = day.spaceship_phase

        if not self._awaiting_alien:
            if submitted not in phase.planet_by_ship:
                self._done = True
                info = self._info(day)
                info["invalid_action"] = submitted
                return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info
            day.selected_ship = submitted
            day.selected_planet = phase.planet_by_ship[submitted]
            self._awaiting_alien = True
            prefix = self._multiplier_prefix(phase)
            aliens = phase.aliens_by_planet[day.selected_planet]
            feedback = (
                "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
                "You end up on planet {planet}. You see alien {a0} and alien {a1}."
            ).format(
                prefix=prefix,
                s0=phase.ships[0],
                s1=phase.ships[1],
                ship=submitted,
                planet=day.selected_planet,
                a0=aliens[0],
                a1=aliens[1],
            )
            return feedback, 0.0, False, False, self._info(day)

        aliens = phase.aliens_by_planet[day.selected_planet]
        if submitted not in aliens:
            self._done = True
            info = self._info(day)
            info["invalid_action"] = submitted
            return "Invalid alien action <<{}>>.".format(submitted), 0.0, False, True, info

        base_treasure = phase.treasure_by_alien[submitted]
        received = base_treasure * phase.multiplier
        self._treasure_total += received
        prefix = self._multiplier_prefix(phase)
        base_display = int(base_treasure) if base_treasure == int(base_treasure) else base_treasure
        feedback = (
            "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
            "You end up on planet {planet}. You see alien {a0} and alien {a1}. "
            "You press <<{alien}>>. You find {base} pieces of space treasure."
        ).format(
            prefix=prefix,
            s0=phase.ships[0],
            s1=phase.ships[1],
            ship=day.selected_ship,
            planet=day.selected_planet,
            a0=aliens[0],
            a1=aliens[1],
            alien=submitted,
            base=base_display,
        )
        self._advance_day()
        return feedback, float(received), self._done, False, self._info(day)

    def _advance_day(self) -> None:
        self._day_idx += 1
        self._done = self._day_idx >= len(self._days)
        self._awaiting_alien = False

    @staticmethod
    def _multiplier_prefix(phase: _SpaceshipPhase) -> str:
        if phase.multiplier > 1:
            return "There is a treasure multiplier. "
        return "There is no treasure multiplier. "

    def _render_spaceship_prompt(self, day: _KoolCostExp2Day) -> str:
        phase = day.spaceship_phase
        return (
            self._multiplier_prefix(phase)
            + "You are presented with spaceships {} and {}.".format(phase.ships[0], phase.ships[1])
        )

    def _info(self, day: Optional[_KoolCostExp2Day]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "awaiting_alien": self._awaiting_alien,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
        }
        if day is not None and self.include_human_ref and not self._awaiting_alien:
            info["planet_by_ship"] = dict(day.spaceship_phase.planet_by_ship)
            info["treasure_by_alien"] = dict(day.spaceship_phase.treasure_by_alien)
            info["multiplier"] = day.spaceship_phase.multiplier
        return info
