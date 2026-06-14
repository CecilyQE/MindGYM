"""Shared random dynamics helpers for generative environments."""

import random
from typing import List


def smooth_arm_values(
    rng: random.Random, n_arms: int, length_scale: float = 3.0, mean: float = 50.0, spread: float = 18.0
) -> List[int]:
    """Sample spatially correlated integer rewards on a 1..n_arms line."""
    raw = [rng.gauss(0.0, 1.0) for _ in range(n_arms)]
    radius = max(2, int(length_scale * 2))
    smoothed = []
    for index in range(n_arms):
        window = raw[max(0, index - radius) : min(n_arms, index + radius + 1)]
        smoothed.append(sum(window) / len(window))
    values = [mean + spread * value for value in smoothed]
    return [int(max(0, min(100, round(value)))) for value in values]


def sample_discrete_outcome(rng: random.Random, values: List[float], probabilities: List[float]) -> float:
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have the same length.")
    draw = rng.random()
    cumulative = 0.0
    for value, probability in zip(values, probabilities):
        cumulative += probability
        if draw <= cumulative:
            return value
    return values[-1]
