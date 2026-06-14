"""Optional Gymnasium adapter for causal text task environments."""

import string
from typing import Any, Dict, Optional, Tuple

from psycenvir.core.registry import make_task_env

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when the optional extra is absent.
    gym = None
    spaces = None


_GymBase = gym.Env if gym is not None else object


class GymnasiumTextEnv(_GymBase):
    """Expose an existing text TaskEnv through the Gymnasium API."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        task_env: Any,
        max_observation_length: int = 1_000_000,
        max_action_length: int = 64,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "GymnasiumTextEnv requires the optional dependency; install psycenvir[gym]."
            )
        self.task_env = task_env
        self.observation_space = spaces.Text(
            max_length=max_observation_length, charset=string.printable
        )
        self.action_space = spaces.Text(max_length=max_action_length, charset=string.printable)
        self._last_observation: Optional[str] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        observation, info = self.task_env.reset(seed=seed)
        self._last_observation = observation
        return observation, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.task_env.step(action)
        self._last_observation = observation
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        return self._last_observation


def make_gymnasium_env(experiment_id: str, text: str, **task_kwargs: Any) -> GymnasiumTextEnv:
    """Build a Gymnasium adapter around one registered causal TaskEnv."""
    return GymnasiumTextEnv(make_task_env(experiment_id, text, **task_kwargs))
