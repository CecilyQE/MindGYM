"""Generative Columbia Card Task (Frey et al. style)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import FREY_CCT_INSTRUCTION

FREY_CCT_EXPERIMENT_ID = "frey2017cct/exp1.csv"


@dataclass(frozen=True)
class _CCTRound:
    round_number: int
    gain_amount: int
    loss_amount: int
    n_loss_cards: int
    deck: Tuple[str, ...]


class FreyCCTGenerativeEnv:
    """Fresh CCT rounds with shuffled gain/loss decks."""

    def __init__(
        self,
        experiment_id: str = FREY_CCT_EXPERIMENT_ID,
        n_rounds: int = 84,
        n_cards: int = 32,
        gain_range: Tuple[int, int] = (10, 600),
        loss_range: Tuple[int, int] = (25, 750),
        loss_count_range: Tuple[int, int] = (1, 28),
        turn_action: str = "E",
        stop_action: str = "C",
        instruction: str = FREY_CCT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_rounds = n_rounds
        self.n_cards = n_cards
        self.gain_range = gain_range
        self.loss_range = loss_range
        self.loss_count_range = loss_count_range
        self.turn_action = turn_action
        self.stop_action = stop_action
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._rounds: List[_CCTRound] = []
        self._round_idx = 0
        self._deck_idx = 0
        self._round_score = 0.0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._rounds = []
        for round_number in range(1, self.n_rounds + 1):
            gain_amount = self._rng.randint(*self.gain_range)
            loss_amount = self._rng.randint(*self.loss_range)
            n_loss_cards = self._rng.randint(*self.loss_count_range)
            deck = ["loss"] * n_loss_cards + ["gain"] * (self.n_cards - n_loss_cards)
            self._rng.shuffle(deck)
            self._rounds.append(
                _CCTRound(
                    round_number=round_number,
                    gain_amount=gain_amount,
                    loss_amount=loss_amount,
                    n_loss_cards=n_loss_cards,
                    deck=tuple(deck),
                )
            )
        self._round_idx = 0
        self._deck_idx = 0
        self._round_score = 0.0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_round(self._rounds[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed FreyCCTGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        round_data = self._rounds[self._round_idx]
        if submitted not in {self.turn_action, self.stop_action}:
            self._done = True
            info = self._info(round_data, None)
            info["invalid_action"] = submitted
            return "Invalid CCT action <<{}>>.".format(submitted), 0.0, False, True, info

        if submitted == self.stop_action:
            observation = "You press <<{}>> and claim your payout.".format(submitted)
            return self._finish_round(observation, 0.0, round_data)

        if self._deck_idx >= len(round_data.deck):
            self._done = True
            info = self._info(round_data, None)
            info["unsupported_counterfactual"] = "turn_after_deck_exhausted"
            return "No cards remain in this round.", 0.0, False, True, info

        card_type = round_data.deck[self._deck_idx]
        reward = (
            float(round_data.gain_amount) if card_type == "gain" else -float(round_data.loss_amount)
        )
        self._round_score += reward
        self._deck_idx += 1
        observation = "You press <<{}>> and turn over a {} card. Your current score is {}.".format(
            submitted,
            card_type,
            self._format_score(self._round_score),
        )
        if card_type == "loss":
            observation += " The round has now ended because you encountered a loss card."
            return self._finish_round(observation, reward, round_data)
        return observation, reward, False, False, self._info(round_data, submitted)

    def _finish_round(
        self, observation: str, reward: float, round_data: _CCTRound
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        observation += "\nYour final score for this round is {}.".format(
            self._format_score(self._round_score)
        )
        self._points += self._round_score
        info = self._info(round_data, None)
        info["round_final_score"] = self._round_score
        self._round_idx += 1
        self._deck_idx = 0
        self._round_score = 0.0
        self._done = self._round_idx >= len(self._rounds)
        if not self._done:
            observation = "{}\n\n{}".format(observation, self._render_round(self._rounds[self._round_idx]))
        info["points"] = self._points
        return observation, reward, self._done, False, info

    @staticmethod
    def _format_score(score: float) -> str:
        return str(int(score)) if score.is_integer() else str(score)

    @staticmethod
    def _render_round(round_data: _CCTRound) -> str:
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

    def _info(self, round_data: Optional[_CCTRound], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "round_idx": self._round_idx,
            "deck_idx": self._deck_idx,
            "round_score": self._round_score,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "fresh_draw_generation": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if round_data is not None:
            info["round_number"] = round_data.round_number
        return info
