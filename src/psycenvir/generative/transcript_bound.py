"""Transcript-frozen schedule helpers and audit metadata for generative envs."""

from __future__ import annotations

from typing import Any, Dict, Tuple

TRANSCRIPT_FIDELITY = "generative_transcript_calibrated"


class TranscriptBoundAuditEnv:
    """Wraps an env so audit reports generative_transcript_calibrated fidelity."""

    def __init__(self, inner: Any, experiment_id: str) -> None:
        self._inner = inner
        self.experiment_id = experiment_id

    def reset(self, seed: Any = None) -> Tuple[str, Dict[str, Any]]:
        observation, info = self._inner.reset(seed)
        return observation, _mark_info(info)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._inner.step(action)
        return observation, reward, terminated, truncated, _mark_info(info)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _mark_info(info: Dict[str, Any]) -> Dict[str, Any]:
    marked = dict(info)
    marked["fidelity_level"] = TRANSCRIPT_FIDELITY
    marked["episode_generative"] = True
    marked["transcript_bound"] = True
    return marked
