"""Small agent evaluation helpers for text task environments."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


TextAgent = Callable[[str, Dict[str, Any]], str]


@dataclass(frozen=True)
class EpisodeResult:
    actions: List[str]
    rewards: List[float]
    observations: List[str]
    terminated: bool
    truncated: bool
    final_info: Dict[str, Any]

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


def run_text_agent(
    env: Any,
    agent: TextAgent,
    seed: Optional[int] = None,
    max_steps: int = 100_000,
) -> EpisodeResult:
    """Run an LLM-compatible callable policy to normal termination or truncation."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    observation, info = env.reset(seed=seed)
    observations = [observation]
    actions: List[str] = []
    rewards: List[float] = []
    terminated = False
    truncated = False
    for _ in range(max_steps):
        action = agent(observation, dict(info))
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(observation)
        if terminated or truncated:
            return EpisodeResult(
                actions=actions,
                rewards=rewards,
                observations=observations,
                terminated=terminated,
                truncated=truncated,
                final_info=info,
            )
    raise RuntimeError("Text agent exceeded max_steps without terminating or truncating.")
