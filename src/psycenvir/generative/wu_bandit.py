"""Generative spatially correlated multi-armed bandit (Wu et al., 2018 style)."""

import random
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.dynamics import smooth_arm_values
from psycenvir.generative.instructions import WU_BANDIT_INSTRUCTION

WU_EXPERIMENT_ID = "wu2018generalisation/exp1.csv"


class WuSpatialBanditGenerativeEnv:
    """Fresh 16-environment bandit with spatially correlated arm rewards."""

    def __init__(
        self,
        n_environments: int = 16,
        n_arms: int = 30,
        choices_short: int = 5,
        choices_long: int = 10,
        reselection_noise: float = 2.0,
        length_scale: float = 3.0,
        instruction: str = WU_BANDIT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.n_environments = n_environments
        self.n_arms = n_arms
        self.choices_short = choices_short
        self.choices_long = choices_long
        self.reselection_noise = reselection_noise
        self.length_scale = length_scale
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._env_arm_values: List[List[int]] = []
        self._choices_per_env: List[int] = []
        self._start_arms: List[int] = []
        self._env_idx = 0
        self._choices_left = 0
        self._revealed: Dict[int, int] = {}
        self._total_points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._env_arm_values = [
            smooth_arm_values(self._rng, self.n_arms, self.length_scale)
            for _ in range(self.n_environments)
        ]
        self._choices_per_env = [
            self.choices_long if self._rng.random() < 0.5 else self.choices_short
            for _ in range(self.n_environments)
        ]
        self._start_arms = [self._rng.randrange(self.n_arms) for _ in range(self.n_environments)]
        self._env_idx = 0
        self._choices_left = 0
        self._revealed = {}
        self._total_points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._environment_header()),
            self._info(None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WuSpatialBanditGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        try:
            arm = int(submitted)
        except ValueError:
            self._done = True
            info = self._info(None)
            info["invalid_action"] = submitted
            return "Invalid option <<{}>>.".format(submitted), 0.0, False, True, info
        if arm < 0 or arm >= self.n_arms:
            self._done = True
            info = self._info(None)
            info["invalid_action"] = submitted
            return "Invalid option <<{}>>.".format(submitted), 0.0, False, True, info

        if self._choices_left == 0:
            self._begin_environment()

        base_value = self._env_arm_values[self._env_idx][arm]
        if arm in self._revealed:
            realized = int(
                round(base_value + self._rng.gauss(0.0, self.reselection_noise))
            )
        else:
            realized = base_value
            self._revealed[arm] = realized
        realized = max(0, min(100, realized))
        reward = float(realized)
        self._total_points += reward
        feedback = "You press <<{}>> and receive {} points.".format(arm, realized)
        self._choices_left -= 1
        if self._choices_left <= 0:
            self._env_idx += 1
            self._revealed = {}
            if self._env_idx >= self.n_environments:
                self._done = True
                return feedback, reward, True, False, self._info(arm)
            observation = "{}\n\n{}".format(feedback, self._environment_header())
            return observation, reward, False, False, self._info(arm)
        return feedback, reward, False, False, self._info(arm)

    def _begin_environment(self) -> None:
        start_arm = self._start_arms[self._env_idx]
        start_value = self._env_arm_values[self._env_idx][start_arm]
        self._revealed = {start_arm: start_value}
        self._choices_left = self._choices_per_env[self._env_idx]

    def _environment_header(self) -> str:
        if self._env_idx >= self.n_environments:
            return ""
        env_number = self._env_idx + 1
        start_arm = self._start_arms[self._env_idx]
        start_value = self._env_arm_values[self._env_idx][start_arm]
        if self._choices_left == 0:
            self._begin_environment()
        return (
            "Environment number {}:\n"
            "The value of option {} is {}. You have {} choices to make in this environment."
        ).format(env_number, start_arm, start_value, self._choices_left)

    def _info(self, arm: Optional[int]) -> Dict[str, Any]:
        return {
            "experiment_id": WU_EXPERIMENT_ID,
            "environment_idx": self._env_idx,
            "choices_remaining": self._choices_left,
            "points": self._total_points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "selected_arm": arm,
            "instruction_shown": bool(self.instruction),
        }
