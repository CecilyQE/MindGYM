"""Generative Kool et al. (2016) when-task: spaceship -> planet -> treasure/antimatter."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import KOOL_WHEN_EXP1_INSTRUCTION

KOOL_WHEN_EXP1_ID = "kool2016when/exp1.csv"

_PAIR_CONFIGS = {
    "GS": {"ships": ("G", "S"), "planet_map": {"G": "R", "S": "Z"}},
    "TN": {"ships": ("T", "N"), "planet_map": {"T": "R", "N": "Z"}},
}


@dataclass(frozen=True)
class _KoolDay:
    pair_key: str
    ships: Tuple[str, str]
    planet_map: Dict[str, str]
    outcomes_by_action: Dict[str, Tuple[str, int]]


class KoolWhenExp1GenerativeEnv:
    """Fresh two-spaceship days with planet-specific treasure or antimatter."""

    def __init__(
        self,
        experiment_id: str = KOOL_WHEN_EXP1_ID,
        n_days: int = 125,
        timeout_probability: float = 0.02,
        instruction: str = KOOL_WHEN_EXP1_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_days = n_days
        self.timeout_probability = timeout_probability
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._days: List[_KoolDay] = []
        self._day_idx = 0
        self._treasure_total = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        planet_state = {
            "R": {"treasure_mean": self._rng.uniform(2.0, 5.0), "antimatter_prob": self._rng.uniform(0.1, 0.4)},
            "Z": {"treasure_mean": self._rng.uniform(2.0, 5.0), "antimatter_prob": self._rng.uniform(0.1, 0.4)},
        }
        self._days = []
        for _ in range(self.n_days):
            if self._rng.random() < self.timeout_probability:
                self._days.append(
                    _KoolDay(
                        pair_key="timeout",
                        ships=("?", "?"),
                        planet_map={},
                        outcomes_by_action={},
                    )
                )
                continue
            pair_key = self._rng.choice(["GS", "TN"])
            config = _PAIR_CONFIGS[pair_key]
            ships = config["ships"]
            if self._rng.random() < 0.5:
                ships = (ships[1], ships[0])
            outcomes: Dict[str, Tuple[str, int]] = {}
            for ship in ships:
                planet = config["planet_map"][ship]
                state = planet_state[planet]
                if self._rng.random() < state["antimatter_prob"]:
                    amount = max(1, int(round(self._rng.gauss(2.0, 1.0))))
                    outcomes[ship] = ("antimatter", amount)
                else:
                    amount = max(
                        0, int(round(self._rng.gauss(state["treasure_mean"], 1.5)))
                    )
                    outcomes[ship] = ("treasure", amount)
                state["treasure_mean"] += self._rng.gauss(0.0, 0.15)
                state["antimatter_prob"] = min(
                    0.8, max(0.05, state["antimatter_prob"] + self._rng.gauss(0.0, 0.03))
                )
            self._days.append(
                _KoolDay(
                    pair_key=pair_key,
                    ships=ships,
                    planet_map=dict(config["planet_map"]),
                    outcomes_by_action=outcomes,
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
            raise RuntimeError("Cannot step a completed KoolWhenExp1GenerativeEnv; call reset().")
        day = self._days[self._day_idx]
        if day.pair_key == "timeout":
            feedback = (
                "You are presented with spaceships {} and {}. "
                "You do not respond in time on this day. You do not go to any planet. You find nothing."
            ).format(day.ships[0], day.ships[1])
            self._advance_day()
            return feedback, 0.0, self._done, False, self._info(day, None)

        submitted = normalize_action(action)
        if submitted not in day.outcomes_by_action:
            self._done = True
            info = self._info(day, None)
            info["invalid_action"] = submitted
            return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info

        outcome_type, amount = day.outcomes_by_action[submitted]
        planet = day.planet_map[submitted]
        if outcome_type == "treasure":
            self._treasure_total += amount
            reward = float(amount)
            find_text = "You find {} pieces of space treasure.".format(amount)
        else:
            self._treasure_total = max(0, self._treasure_total - amount)
            reward = float(-amount)
            find_text = "You find {} pieces of antimatter.".format(amount)
        feedback = (
            "You are presented with spaceships {} and {}. You press <<{}>>. "
            "You end up on planet {}. {}"
        ).format(day.ships[0], day.ships[1], submitted, planet, find_text)
        self._advance_day()
        return feedback, reward, self._done, False, self._info(day, submitted)

    def _advance_day(self) -> None:
        self._day_idx += 1
        self._done = self._day_idx >= len(self._days)

    def _render_day(self, day: _KoolDay) -> str:
        if day.pair_key == "timeout":
            return "You are presented with spaceships {} and {}.".format(day.ships[0], day.ships[1])
        return "You are presented with spaceships {} and {}.".format(day.ships[0], day.ships[1])

    def _info(self, day: Optional[_KoolDay], selected_ship: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_ship": selected_ship,
        }
        if day is not None and self.include_human_ref:
            info["outcomes_by_action"] = dict(day.outcomes_by_action)
        return info
