"""Generative Tomov castle multitask with market prices and hidden room resources."""

import random
from typing import Any, Dict, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import TOMOV_CASTLE_INSTRUCTION

TOMOV_CASTLE_EXP1_ID = "tomov2021multitask/exp1.csv"
TOMOV_CASTLE_EXP3_ID = "tomov2021multitask/exp3.csv"
DEFAULT_CASTLE_DOORS_EXP1 = ("I", "P", "G")
DEFAULT_CASTLE_DOORS_EXP3 = ("V", "F", "Z")


class TomovCastleGenerativeEnv:
    """Fresh castle rounds: two door choices per round with resampled market prices."""

    def __init__(
        self,
        experiment_id: str = TOMOV_CASTLE_EXP1_ID,
        n_rounds: int = 20,
        door_keys: Tuple[str, ...] = DEFAULT_CASTLE_DOORS_EXP1,
        instruction: str = TOMOV_CASTLE_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_rounds = n_rounds
        self.door_keys = door_keys
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._round_number = 0
        self._step_in_round = 0
        self._current_room = 0
        self._prices = (0, 0, 0)
        self._price_line = ""
        self._room_resources: Dict[int, Dict[str, Tuple[int, int, int]]] = {}
        self._transitions: Dict[str, int] = {}
        self._points = 0.0
        self._done = False

    def _sample_prices(self) -> Tuple[int, int, int]:
        return (
            self._rng.randint(-2, 2),
            self._rng.randint(-2, 2),
            self._rng.randint(-2, 2),
        )

    def _sample_room_resources(self) -> Dict[int, Dict[str, Tuple[int, int, int]]]:
        resources: Dict[int, Dict[str, Tuple[int, int, int]]] = {}
        for room_number in range(1, 4):
            resources[room_number] = {
                key: (
                    self._rng.randint(0, 100),
                    self._rng.randint(0, 100),
                    self._rng.randint(0, 100),
                )
                for key in self.door_keys
            }
        return resources

    @staticmethod
    def _reward(
        prices: Tuple[int, int, int], resources: Tuple[int, int, int]
    ) -> float:
        wood, stone, iron = resources
        return float(wood * prices[0] + stone * prices[1] + iron * prices[2])

    def _start_round(self) -> str:
        self._prices = self._sample_prices()
        self._price_line = (
            "The current market prices are {} for wood, {} for stone, and {} for iron."
        ).format(*self._prices)
        self._room_resources = self._sample_room_resources()
        self._transitions = {key: self._rng.randint(1, 3) for key in self.door_keys}
        self._current_room = 0
        self._step_in_round = 0
        return "{}\nYou are in room 0.".format(self._price_line)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._round_number = 1
        self._points = 0.0
        self._done = False
        observation = self._start_round()
        return (
            render_initial_observation(self.instruction, observation),
            self._info(None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed TomovCastleGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        if submitted not in self.door_keys:
            self._done = True
            info = self._info(None)
            info["invalid_action"] = submitted
            return "Invalid door action <<{}>>.".format(submitted), 0.0, False, True, info

        if self._step_in_round == 0:
            destination = self._transitions[submitted]
            resources = self._room_resources[destination][submitted]
            reward = 0.0
            self._current_room = destination
            self._step_in_round = 1
            feedback = (
                "You are in room 0. You press <<{}>> and you find {} wood, {} stone, "
                "and {} iron. You get {} points."
            ).format(submitted, *resources, int(reward))
            next_observation = "You are in room {}.".format(self._current_room)
            feedback = "{}\n\n{}".format(feedback, next_observation)
            return feedback, reward, False, False, self._info(submitted)

        resources = self._room_resources[self._current_room][submitted]
        reward = self._reward(self._prices, resources)
        self._points += reward
        feedback = (
            "You are in room {}. You press <<{}>> and you find {} wood, {} stone, "
            "and {} iron. You get {} points."
        ).format(self._current_room, submitted, *resources, int(reward))
        self._round_number += 1
        self._done = self._round_number > self.n_rounds
        if not self._done:
            observation = self._start_round()
            feedback = "{}\n\n{}".format(feedback, observation)
        return feedback, reward, self._done, False, self._info(submitted)

    def _info(
        self,
        selected_action: Optional[str],
        next_observation: Optional[str] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "round_number": self._round_number,
            "step_in_round": self._step_in_round,
            "room_number": self._current_room,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if next_observation is not None:
            info["next_observation"] = next_observation
        return info
