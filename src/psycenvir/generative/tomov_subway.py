"""Generative Tomov subway discovery navigation."""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import TOMOV_SUBWAY_INSTRUCTION

TOMOV_SUBWAY_EXPERIMENT_IDS = (
    "tomov2020discovery/exp2.csv",
    "tomov2020discovery/exp4.csv",
    "tomov2020discovery/exp5.csv",
    "tomov2020discovery/exp7.csv",
)
TOMOV_SUBWAY_EXP2_ID = TOMOV_SUBWAY_EXPERIMENT_IDS[0]
DEFAULT_DIRECTION_KEYS = {
    "north": "G",
    "west": "B",
    "south": "V",
    "east": "C",
    "goal": "Z",
}


@dataclass
class _SubwayState:
    graph: Dict[str, Dict[str, Optional[str]]]
    start: str
    goal: str
    current: str
    round_number: int
    n_rounds: int


class TomovSubwayGenerativeEnv:
    """Fresh subway graphs per episode; move keys follow Psych-101 direction mapping."""

    def __init__(
        self,
        experiment_id: str = TOMOV_SUBWAY_EXP2_ID,
        n_rounds: int = 20,
        n_stations: int = 6,
        direction_keys: Optional[Dict[str, str]] = None,
        instruction: str = TOMOV_SUBWAY_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_rounds = int(calibration.get("n_rounds", n_rounds))
        self.n_stations = int(calibration.get("n_stations", n_stations))
        self.direction_keys = dict(direction_keys or DEFAULT_DIRECTION_KEYS)
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[_SubwayState] = None
        self._successful_rounds = 0
        self._done = False

    def _build_graph(self) -> Dict[str, Dict[str, Optional[str]]]:
        station_ids = [str(index) for index in range(1, self.n_stations + 1)]
        graph: Dict[str, Dict[str, Optional[str]]] = {
            station: {"north": None, "west": None, "south": None, "east": None}
            for station in station_ids
        }
        for index, station in enumerate(station_ids):
            if index + 1 < len(station_ids):
                north = station_ids[index + 1]
                graph[station]["north"] = north
                graph[north]["south"] = station
            if index + 2 < len(station_ids):
                east = station_ids[index + 2]
                graph[station]["east"] = east
                graph[east]["west"] = station
        return graph

    def _shortest_path_exists(self, graph: Dict[str, Dict[str, Optional[str]]], start: str, goal: str) -> bool:
        queue = deque([start])
        seen = {start}
        while queue:
            node = queue.popleft()
            if node == goal:
                return True
            for neighbor in graph[node].values():
                if neighbor and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return False

    def _start_round(self, round_number: int) -> None:
        graph = self._build_graph()
        stations = list(graph.keys())
        start, goal = self._rng.sample(stations, 2)
        while not self._shortest_path_exists(graph, start, goal):
            start, goal = self._rng.sample(stations, 2)
        self._state = _SubwayState(
            graph=graph,
            start=start,
            goal=goal,
            current=start,
            round_number=round_number,
            n_rounds=self.n_rounds,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._successful_rounds = 0
        self._done = False
        self._start_round(1)
        return (
            render_initial_observation(self.instruction, self._render_observation(show_header=True)),
            self._info(None),
        )

    def _neighbor_text(self, station: str) -> str:
        assert self._state is not None
        neighbors = self._state.graph[station]
        parts = []
        for direction in ("north", "west", "south", "east"):
            target = neighbors[direction]
            if target is None:
                parts.append("circle on the {}".format(direction))
            else:
                parts.append("{} on the {}".format(target, direction))
        return ", ".join(parts)

    def _render_observation(self, show_header: bool) -> str:
        assert self._state is not None
        header = ""
        if show_header:
            header = "The new starting station is {} and the goal station is {}.\n".format(
                self._state.start, self._state.goal
            )
        body = "Your station: {}. Neighboring stations: {}.".format(
            self._state.current, self._neighbor_text(self._state.current)
        )
        return header + body

    def _valid_actions(self) -> Tuple[str, ...]:
        assert self._state is not None
        keys = self.direction_keys
        actions: List[str] = []
        neighbors = self._state.graph[self._state.current]
        for direction in ("north", "west", "south", "east"):
            if neighbors[direction] is not None:
                actions.append(keys[direction])
        if self._state.current == self._state.goal:
            actions.append(keys["goal"])
        return tuple(actions)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done or self._state is None:
            raise RuntimeError("Cannot step a completed TomovSubwayGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        valid = self._valid_actions()
        if submitted not in valid:
            self._done = True
            info = self._info(submitted)
            info["invalid_action"] = submitted
            return "Invalid subway action <<{}>>.".format(submitted), 0.0, False, True, info

        keys = self.direction_keys
        completes_round = False
        reward = 0.0
        if submitted == keys["goal"]:
            completes_round = True
            reward = 1.0
            self._successful_rounds += 1
        else:
            direction = next(
                name for name, key in keys.items() if key == submitted and name != "goal"
            )
            self._state.current = self._state.graph[self._state.current][direction] or self._state.current

        feedback = "You press <<{}>>.".format(submitted)
        if completes_round:
            feedback = "{}\nYou are successful.".format(feedback)
            if self._state.round_number >= self._state.n_rounds:
                self._done = True
                return feedback, reward, True, False, self._info(submitted)
            self._start_round(self._state.round_number + 1)
            feedback = "{}\n\n{}".format(feedback, self._render_observation(show_header=True))
            return feedback, reward, False, False, self._info(submitted)

        feedback = "{}\n\n{}".format(feedback, self._render_observation(show_header=False))
        return feedback, reward, False, False, self._info(submitted)

    def _info(self, selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "successful_rounds": self._successful_rounds,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if self._state is not None:
            info["round_number"] = self._state.round_number
            info["current_station"] = self._state.current
            info["goal_station"] = self._state.goal
        return info
