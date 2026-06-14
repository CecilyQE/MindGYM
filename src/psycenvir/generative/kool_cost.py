"""Generative Kool et al. (2017) cost-task exp1: optional treasure multiplier."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import KOOL_COST_EXP1_INSTRUCTION

KOOL_COST_EXP1_ID = "kool2017cost/exp1.csv"


@dataclass(frozen=True)
class _KoolCostDay:
    multiplier: int
    ships: Tuple[str, str]
    planet_map: Dict[str, str]
    treasure_by_action: Dict[str, int]


class KoolCostExp1GenerativeEnv:
    """Fresh spaceship days with transcript-calibrated topology and drifting mines."""

    def __init__(
        self,
        experiment_id: str = KOOL_COST_EXP1_ID,
        n_days: int = 200,
        multiplier_value: int = 5,
        multiplier_probability: float = 0.25,
        session_topologies: Optional[List[Dict[str, Any]]] = None,
        instruction: str = KOOL_COST_EXP1_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_days = n_days
        self.multiplier_value = multiplier_value
        self.multiplier_probability = multiplier_probability
        self._session_topologies = list(
            session_topologies or calibration.get("session_topologies") or []
        )
        if not self._session_topologies:
            raise ValueError(
                "KoolCostExp1GenerativeEnv requires transcript-derived session_topologies."
            )
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._days: List[_KoolCostDay] = []
        self._day_idx = 0
        self._treasure_total = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)

        topology = self._rng.choice(self._session_topologies)
        ships = tuple(topology["ships"])
        planet_map = dict(topology["planet_by_ship"])
        ship_means = {
            ship: self._rng.uniform(0.0, 8.0) for ship in planet_map
        }

        self._days = []
        for _ in range(self.n_days):
            if self._rng.random() < 0.5:
                ships = (ships[1], ships[0])
            multiplier = (
                self.multiplier_value
                if self._rng.random() < self.multiplier_probability
                else 1
            )
            treasure_by_action = {}
            for ship in ships:
                planet = planet_map[ship]
                base = max(0, int(round(self._rng.gauss(ship_means[ship], 1.5))))
                treasure_by_action[ship] = base
                ship_means[ship] += self._rng.gauss(0.0, 0.1)
            self._days.append(
                _KoolCostDay(
                    multiplier=multiplier,
                    ships=ships,
                    planet_map=planet_map,
                    treasure_by_action=treasure_by_action,
                )
            )
        self._day_idx = 0
        self._treasure_total = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_day(self._days[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KoolCostExp1GenerativeEnv; call reset().")
        submitted = normalize_action(action)
        day = self._days[self._day_idx]
        if submitted not in day.treasure_by_action:
            self._done = True
            info = self._info(day, None)
            info["invalid_action"] = submitted
            return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info

        base_treasure = day.treasure_by_action[submitted]
        received = base_treasure * day.multiplier
        self._treasure_total += received
        planet = day.planet_map[submitted]
        if day.multiplier > 1:
            prefix = "There is a treasure multiplier. "
        else:
            prefix = "There is no treasure multiplier. "
        feedback = (
            "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
            "You end up on planet {planet}. You find {base} pieces of space treasure. "
            "You receive {received} pieces of space treasure."
        ).format(
            prefix=prefix,
            s0=day.ships[0],
            s1=day.ships[1],
            ship=submitted,
            planet=planet,
            base=base_treasure,
            received=received,
        )
        self._day_idx += 1
        self._done = self._day_idx >= len(self._days)
        return feedback, float(received), self._done, False, self._info(day, submitted)

    def _render_day(self, day: _KoolCostDay) -> str:
        if day.multiplier > 1:
            return "There is a treasure multiplier. You are presented with spaceships {} and {}.".format(
                day.ships[0], day.ships[1]
            )
        return "There is no treasure multiplier. You are presented with spaceships {} and {}.".format(
            day.ships[0], day.ships[1]
        )

    def _info(self, day: Optional[_KoolCostDay], selected_ship: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_ship": selected_ship,
        }
        if day is not None and self.include_human_ref:
            info["treasure_by_action"] = dict(day.treasure_by_action)
            info["multiplier"] = day.multiplier
        return info
