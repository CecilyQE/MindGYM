"""Non-causal transcript replay baseline."""

from typing import Any, Dict, Optional, Tuple

from psycenvir.core.base import normalize_action
from psycenvir.models import ParsedTranscript


class ReplayEnv:
    """Play transcript continuations regardless of the submitted action.

    This environment is intentionally not suitable for causal RL. It preserves
    the baseline used to compare a simulated environment against transcript
    next-action behavior.
    """

    def __init__(self, transcript: ParsedTranscript, include_human_ref: bool = False) -> None:
        self.transcript = transcript
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._done = False
        return self.transcript.initial_observation, self._info(None)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed ReplayEnv; call reset().")
        submitted = normalize_action(action)
        event = self.transcript.events[self._trial_idx]
        observation = "{}{}".format(self._render_action(event, submitted), event.continuation)
        human_action = event.human_action
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.transcript.events)
        return observation, 0.0, self._done, False, self._info(human_action)

    @staticmethod
    def _render_action(event: Any, submitted: str) -> str:
        if not event.action_segments:
            return "You {} <<{}>>".format(event.verb, submitted)
        submitted_actions = tuple(part.strip() for part in submitted.split("||"))
        if len(submitted_actions) != len(event.human_actions):
            raise ValueError(
                "Action requires {} encoded value(s) separated by '||'; received {}.".format(
                    len(event.human_actions), len(submitted_actions)
                )
            )
        rendered = event.action_segments[0]
        for value, segment in zip(submitted_actions, event.action_segments[1:]):
            rendered += "<<{}>>{}".format(value, segment)
        return rendered

    def _info(self, human_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.transcript.experiment_id,
            "trial_idx": self._trial_idx,
            "feedback_causal": False,
            "reward_defined": False,
        }
        if self.include_human_ref and human_action is not None:
            info["human_ref"] = human_action
        return info
