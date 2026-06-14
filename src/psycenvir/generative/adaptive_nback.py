"""Generative adaptive n-back task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

ENKAVI_ADAPTIVE_NBACK_EXP1_ID = "enkavi2019adaptivenback/exp1.csv"
ADAPTIVE_NBACK_INSTRUCTION = (
    "You will view a stream of letters on the screen, one letter at a time.\n"
    "At the beginning of a block, you are told a number N.\n"
    "If the letter you see matches the letter N trials ago, press W, otherwise press D.\n"
    "If you make more than 5 mistakes in a block, N is decreased by 1.\n"
    "If you make fewer than 3 mistakes in a block, N is increased by 1.\n"
    "You will go through 20 blocks with 20+N trials each."
)


@dataclass(frozen=True)
class _NBackTrial:
    block_index: int
    n: int
    letter: str
    correct_action: str


class EnkaviAdaptiveNBackGenerativeEnv:
    """Fresh adaptive n-back episodes with deterministic match/non-match correctness."""

    def __init__(
        self,
        n_blocks: int = 20,
        initial_n: int = 2,
        block_base_trials: int = 20,
        letters: Tuple[str, ...] = ("B", "D", "G", "T", "V"),
        match_key: str = "W",
        nonmatch_key: str = "D",
        instruction: str = ADAPTIVE_NBACK_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = ENKAVI_ADAPTIVE_NBACK_EXP1_ID
        self.n_blocks = n_blocks
        self.initial_n = initial_n
        self.block_base_trials = block_base_trials
        self.letters = tuple(letter.upper() for letter in letters)
        self.match_key = match_key.upper()
        self.nonmatch_key = nonmatch_key.upper()
        self.valid_actions = (self.match_key, self.nonmatch_key)
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_NBackTrial] = []
        self._trial_idx = 0
        self._block_errors: Dict[int, int] = {}
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        n = self.initial_n
        for block_index in range(self.n_blocks):
            sequence = [self._rng.choice(self.letters) for _ in range(self.block_base_trials + n)]
            for index in range(n, len(sequence)):
                letter = sequence[index]
                correct = self.match_key if letter == sequence[index - n] else self.nonmatch_key
                self._trials.append(_NBackTrial(block_index, n, letter, correct))
            n = max(1, n)
        self._trial_idx = 0
        self._block_errors = {block_index: 0 for block_index in range(self.n_blocks)}
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviAdaptiveNBackGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid response <<{}>>.".format(submitted), 0.0, False, True, info
        correct = submitted == trial.correct_action
        if not correct:
            self._block_errors[trial.block_index] += 1
        reward = 1.0 if correct else 0.0
        feedback = "You see the letter {} and press <<{}>>.".format(trial.letter, submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _NBackTrial) -> str:
        return "Block {}, N = {}:\nYou see the letter {}.".format(
            trial.block_index, trial.n, trial.letter
        )

    def _info(self, trial: Optional[_NBackTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["correct_action"] = trial.correct_action
            info["n_back"] = trial.n
        return info
