"""Causal text environments for selected Psych-101 experiments."""

from .agents import EpisodeResult, run_text_agent
from .benchmark import BadhamBenchmarkResult, ContextCondition, run_badham_benchmark
from .core.gymnasium import GymnasiumTextEnv, make_gymnasium_env
from .core.generative_registry import make_generative_env
from .core.registry import make_replay_env, make_task_env
from .models import ExperimentSpec, FidelityLevel, RewardMode

__all__ = [
    "ExperimentSpec",
    "EpisodeResult",
    "BadhamBenchmarkResult",
    "ContextCondition",
    "FidelityLevel",
    "GymnasiumTextEnv",
    "RewardMode",
    "make_generative_env",
    "make_gymnasium_env",
    "make_replay_env",
    "make_task_env",
    "run_text_agent",
    "run_badham_benchmark",
]
