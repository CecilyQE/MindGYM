"""Small text-environment contract shared by replay and simulation."""

import re
from typing import Optional

from psycenvir.errors import InvalidActionError


FORMATTED_ACTION_RE = re.compile(r"^\s*<<(?P<action>[^<>]+)>>\s*$")


def normalize_action(action: str) -> str:
    """Accept either an action key or Psych-101's ``<<key>>`` formatting."""
    if not isinstance(action, str):
        raise InvalidActionError("Action must be a string.")
    match = FORMATTED_ACTION_RE.match(action)
    normalized = match.group("action").strip() if match else action.strip()
    if not normalized:
        raise InvalidActionError("Action must not be empty.")
    return normalized


def render_initial_observation(instruction: Optional[str], observation: str) -> str:
    """Include source task instructions once, before the first actionable state."""
    if not instruction or not instruction.strip():
        return observation
    return "{}\n\n{}".format(instruction.strip(), observation)
