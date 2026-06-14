"""Generative Steingroever Iowa Gambling Task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import STEINGROEVER_IGT_INSTRUCTION

STEINGROEVER_IGT_EXP1_ID = "steingroever2015data/exp1.csv"
STEINGROEVER_IGT_EXP2_ID = "steingroever2015data/exp2.csv"
STEINGROEVER_IGT_EXP3_ID = "steingroever2015data/exp3.csv"
DEFAULT_IGT_DECKS = ("H", "V", "J", "D")
DEFAULT_IGT_EXP2_DECKS = ("A", "J", "I", "K")
DEFAULT_IGT_EXP3_DECKS = ("U", "F", "I", "S")


@dataclass(frozen=True)
class _IGTTrial:
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, Tuple[float, float]]


class SteingroeverIGTGenerativeEnv:
    """Fresh IGT episodes with two advantageous and two disadvantageous decks."""

    def __init__(
        self,
        experiment_id: str = STEINGROEVER_IGT_EXP1_ID,
        n_trials: int = 100,
        decks: Tuple[str, ...] = DEFAULT_IGT_DECKS,
        instruction: str = STEINGROEVER_IGT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.decks = decks
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_IGTTrial] = []
        self._trial_idx = 0
        self._balance = 2000.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        advantageous = self._rng.sample(list(self.decks), 2)
        deck_params = {}
        for deck in self.decks:
            if deck in advantageous:
                deck_params[deck] = (self._rng.uniform(50.0, 100.0), self._rng.uniform(0.0, 50.0))
            else:
                deck_params[deck] = (self._rng.uniform(50.0, 100.0), self._rng.uniform(100.0, 250.0))
        self._trials = []
        for _ in range(self.n_trials):
            outcomes = {
                deck: (
                    deck_params[deck][0] + self._rng.gauss(0.0, 5.0),
                    deck_params[deck][1] + self._rng.gauss(0.0, 5.0),
                )
                for deck in self.decks
            }
            self._trials.append(
                _IGTTrial(valid_actions=self.decks, outcomes_by_action=outcomes)
            )
        self._trial_idx = 0
        self._balance = 2000.0
        self._done = False
        return (
            render_initial_observation(
                self.instruction, "Select a card from one of the four decks."
            ),
            self._info(None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed SteingroeverIGTGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(None)
            info["invalid_action"] = submitted
            return "Invalid deck <<{}>>.".format(submitted), 0.0, False, True, info

        win, loss = trial.outcomes_by_action[submitted]
        reward = win - loss
        self._balance += reward
        feedback = "You press <<{}>>. You win {:.1f}$ and lose {:.1f}$.".format(
            submitted, win, loss
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        return feedback, reward, self._done, False, self._info(submitted)

    def _info(self, selected_action: Optional[str]) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "balance": self._balance,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
