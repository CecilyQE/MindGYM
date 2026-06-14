"""Generative balloon analogue risk task (Frey et al., 2017 style)."""

import random
from typing import Any, Dict, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import FREY_RISK_INSTRUCTION

FREY_RISK_EXPERIMENT_ID = "frey2017risk/exp1.csv"


class FreyRiskBalloonGenerativeEnv:
    """Fresh BART-style balloons with hidden explosion thresholds."""

    def __init__(
        self,
        n_balloons: int = 30,
        min_threshold: int = 1,
        max_threshold: int = 128,
        instruction: str = FREY_RISK_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
        pump_key: str = "H",
        collect_key: str = "W",
    ) -> None:
        if min_threshold <= 0 or max_threshold < min_threshold:
            raise ValueError("Invalid explosion threshold bounds.")
        self.n_balloons = n_balloons
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self.pump_key = pump_key.upper()
        self.collect_key = collect_key.upper()
        self._seed = seed
        self._rng = random.Random(seed)
        self._thresholds = []
        self._balloon_idx = 0
        self._pumps = 0
        self._accumulated = 0
        self._total_points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._thresholds = [
            self._rng.randint(self.min_threshold, self.max_threshold)
            for _ in range(self.n_balloons)
        ]
        self._balloon_idx = 0
        self._pumps = 0
        self._accumulated = 0
        self._total_points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._balloon_header()),
            self._info(),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed FreyRiskBalloonGenerativeEnv; call reset().")
        submitted = normalize_action(action).upper()
        if submitted not in {self.pump_key, self.collect_key}:
            self._done = True
            info = self._info()
            info["invalid_action"] = submitted
            return "Invalid balloon action <<{}>>.".format(submitted), 0.0, False, True, info

        if submitted == self.pump_key:
            self._pumps += 1
            self._accumulated += 1
            if self._pumps >= self._thresholds[self._balloon_idx]:
                feedback = (
                    "You press <<{}>>. The balloon was inflated too much and explodes."
                ).format(self.pump_key)
                reward = 0.0
                return self._advance_balloon(feedback, reward)
            return "You press <<H>>.", 0.0, False, False, self._info()

        feedback = (
            "You press <<{}>>. You stop inflating the balloon and get {} points."
        ).format(self.collect_key, self._accumulated)
        reward = float(self._accumulated)
        self._total_points += reward
        return self._advance_balloon(feedback, reward)

    def _advance_balloon(
        self, feedback: str, reward: float
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._balloon_idx += 1
        self._pumps = 0
        self._accumulated = 0
        if self._balloon_idx >= self.n_balloons:
            self._done = True
            return feedback, reward, True, False, self._info()
        observation = "{}\n\n{}".format(feedback, self._balloon_header())
        return observation, reward, False, False, self._info()

    def _balloon_header(self) -> str:
        return "Balloon {}:".format(self._balloon_idx + 1)

    def _info(self) -> Dict[str, Any]:
        return {
            "experiment_id": FREY_RISK_EXPERIMENT_ID,
            "balloon_idx": self._balloon_idx,
            "pumps": self._pumps,
            "accumulated": self._accumulated,
            "points": self._total_points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
        }
