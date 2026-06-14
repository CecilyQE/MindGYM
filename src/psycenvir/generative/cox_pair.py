"""Generative Cox pair-recognition task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import COX_PAIR_INSTRUCTION

COX_PAIR_EXP1_ID = "cox2017information/exp1.csv"
DEFAULT_WORDS = (
    "FILE",
    "GERMAN",
    "STANDS",
    "RISES",
    "OFFICER",
    "FUEL",
    "CLASSROOM",
    "JOURNEY",
    "TERRITORY",
    "EDUCATIONAL",
    "TRANSPORTATION",
    "AGREEMENT",
    "SIGNIFICANT",
    "SPECIALIZED",
    "TUBE",
    "ENEMY",
    "CLUB",
    "RAPID",
    "LEGAL",
    "CONCEPT",
    "TOWARDS",
    "VICTORY",
    "RARELY",
    "BAY",
    "ELECTION",
    "ROYAL",
    "BELONG",
    "SUPREME",
    "FRIGHTENED",
    "PLAINS",
    "SLAVES",
    "FILM",
    "STOMACH",
    "WHISPERED",
    "NURSE",
    "GOLDEN",
    "POLITICS",
    "WINGS",
    "BOXES",
    "TOM",
)


@dataclass(frozen=True)
class _CoxTrial:
    observation: str
    correct_action: str


class CoxPairRecognitionGenerativeEnv:
    """Fresh pair-recognition blocks with a newly sampled studied list."""

    def __init__(
        self,
        experiment_id: str = COX_PAIR_EXP1_ID,
        n_studied_pairs: int = 20,
        n_test_trials: int = 60,
        word_pool: Tuple[str, ...] = DEFAULT_WORDS,
        instruction: str = COX_PAIR_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_studied_pairs = n_studied_pairs
        self.n_test_trials = n_test_trials
        self.word_pool = word_pool
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_CoxTrial] = []
        self._studied_pairs: Set[Tuple[str, str]] = set()
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        studied_lines: List[Tuple[str, str]] = []
        self._studied_pairs = set()
        words = self._rng.sample(list(self.word_pool), self.n_studied_pairs * 2)
        for index in range(0, len(words), 2):
            left, right = words[index], words[index + 1]
            studied_lines.append((left, right))
            self._studied_pairs.add((left, right))
            self._studied_pairs.add((right, left))
        study_block = "You study the following 20 word pairs:\n" + "\n".join(
            "{}, {}".format(left, right) for left, right in studied_lines
        )
        self._trials = []
        for _ in range(self.n_test_trials):
            if self._rng.random() < 0.5 and studied_lines:
                left, right = self._rng.choice(studied_lines)
                pair_text = "{}, {}".format(left, right)
                correct_action = "D"
            else:
                left, right = self._rng.sample(list(self.word_pool), 2)
                pair_text = "{}, {}".format(left, right)
                correct_action = "D" if (left, right) in self._studied_pairs else "N"
            self._trials.append(
                _CoxTrial(
                    observation="You view the word pair {}.".format(pair_text),
                    correct_action=correct_action,
                )
            )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        first = self._trials[0].observation
        return (
            render_initial_observation(self.instruction, "{}\n\n{}".format(study_block, first)),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "Cannot step a completed CoxPairRecognitionGenerativeEnv; call reset()."
            )
        submitted = normalize_action(action)
        if submitted not in ("D", "N"):
            self._done = True
            info = self._info(None, False)
            info["invalid_action"] = submitted
            return "Invalid pair action <<{}>>.".format(submitted), 0.0, False, True, info

        trial = self._trials[self._trial_idx]
        is_correct = submitted == trial.correct_action
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        feedback = "{} You press <<{}>>.".format(trial.observation, submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, is_correct)

    def _info(self, trial: Optional[_CoxTrial], is_correct: Optional[bool]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["correct_action"] = trial.correct_action
            if is_correct is not None:
                info["is_correct"] = is_correct
        return info
