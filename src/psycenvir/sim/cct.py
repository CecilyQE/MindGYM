"""Recorded-path Columbia Card Task environment."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import FreyCCTEvent, FreyCCTRound


class FreyCCTRecordedPathEnv:
    """CCT dynamics on recorded deck prefixes with exact early-stop actions.

    Turning a card is supported while the source transcript records the next
    revealed card. Stopping is always causal because it claims the current
    score. Turning after the source participant stopped is unobserved and is
    explicitly truncated rather than sampled.
    """

    EXPERIMENT_ID = "frey2017cct/exp1.csv"

    def __init__(
        self,
        rounds: Iterable[FreyCCTRound],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.rounds: List[FreyCCTRound] = list(rounds)
        if not self.rounds:
            raise ValueError("FreyCCTRecordedPathEnv requires at least one round.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._round_idx = 0
        self._event_idx = 0
        self._trial_idx = 0
        self._round_score = 0.0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._round_idx = 0
        self._event_idx = 0
        self._trial_idx = 0
        self._round_score = 0.0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_round(self.rounds[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed FreyCCTRecordedPathEnv; call reset().")
        submitted = normalize_action(action)
        round_data = self.rounds[self._round_idx]
        source_event = round_data.events[self._event_idx]
        if submitted not in {round_data.turn_action, round_data.stop_action}:
            self._done = True
            info = self._info(round_data, source_event)
            info["invalid_action"] = submitted
            return "Invalid CCT action <<{}>>.".format(submitted), 0.0, False, True, info

        if submitted == round_data.stop_action:
            observation = "You press <<{}>> and claim your payout.".format(submitted)
            return self._finish_round(observation, 0.0, round_data, source_event)

        if source_event.event_type == "stop":
            self._done = True
            info = self._info(round_data, source_event)
            info["unsupported_counterfactual"] = "turn_after_recorded_stop"
            return (
                "No recorded card outcome is available after the source participant stopped.",
                0.0,
                False,
                True,
                info,
            )

        reward = (
            float(round_data.gain_amount)
            if source_event.event_type == "gain"
            else -float(round_data.loss_amount)
        )
        self._round_score += reward
        observation = "You press <<{}>> and turn over a {} card. Your current score is {}.".format(
            submitted, source_event.event_type, self._format_score(self._round_score)
        )
        self._trial_idx += 1
        self._event_idx += 1
        if source_event.event_type == "loss":
            observation += " The round has now ended because you encountered a loss card."
            return self._finish_round(observation, reward, round_data, source_event, advanced=True)
        return observation, reward, False, False, self._info(round_data, source_event)

    def _finish_round(
        self,
        observation: str,
        reward: float,
        round_data: FreyCCTRound,
        source_event: FreyCCTEvent,
        advanced: bool = False,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not advanced:
            self._trial_idx += 1
        observation += "\nYour final score for this round is {}.".format(
            self._format_score(self._round_score)
        )
        self._points += self._round_score
        info = self._info(round_data, source_event)
        info["round_final_score"] = self._round_score
        self._round_idx += 1
        self._event_idx = 0
        self._round_score = 0.0
        self._done = self._round_idx >= len(self.rounds)
        if not self._done:
            observation = "{}\n\n{}".format(observation, self._render_round(self.rounds[self._round_idx]))
        info["points"] = self._points
        return observation, reward, self._done, False, info

    @staticmethod
    def _format_score(score: float) -> str:
        return str(int(score)) if score.is_integer() else str(score)

    @staticmethod
    def _render_round(round_data: FreyCCTRound) -> str:
        return (
            "Round {}:\n"
            "You will be awarded {} points for turning over a gain card.\n"
            "You will lose {} points for turning over a loss card.\n"
            "There are {} loss cards in this round."
        ).format(
            round_data.round_number,
            round_data.gain_amount,
            round_data.loss_amount,
            round_data.n_loss_cards,
        )

    def _info(
        self, round_data: Optional[FreyCCTRound], source_event: Optional[FreyCCTEvent]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "round_idx": self._round_idx,
            "round_score": self._round_score,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_recorded_draws_and_early_stop",
            "fresh_draw_generation": False,
            "instruction_shown": bool(self.instruction),
        }
        if round_data is not None:
            info["round_number"] = round_data.round_number
        if source_event is not None and self.include_human_ref:
            info["human_ref"] = source_event.human_action
        return info
