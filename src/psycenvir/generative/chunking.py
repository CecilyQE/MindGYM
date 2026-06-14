"""Generative instructed-key chunking task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

WU_CHUNKING_EXP1_ID = "wu2023chunking/exp1.csv"
WU_CHUNKING_EXP2_ID = "wu2023chunking/exp2.csv"
CHUNKING_INSTRUCTION = "Press the instructed key.\n\nAct as fast and accurately as possible."


@dataclass(frozen=True)
class _ChunkingTrial:
    instructed_key: str
    rt_ms: int


class WuChunkingGenerativeEnv:
    """Fresh deterministic key-press trials; RT is generated only for feedback format."""

    def __init__(
        self,
        experiment_id: str = WU_CHUNKING_EXP1_ID,
        n_trials: int = 1000,
        keys: Tuple[str, ...] = ("D", "F", "J", "K"),
        instruction: str = CHUNKING_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if len(set(keys)) != len(keys):
            raise ValueError("WuChunkingGenerativeEnv requires distinct keys.")
        self.experiment_id = experiment_id
        self.n_trials = n_trials
        self.keys = tuple(key.upper() for key in keys)
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_ChunkingTrial] = []
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = [
            _ChunkingTrial(self._rng.choice(self.keys), self._rng.randint(350, 2200))
            for _ in range(self.n_trials)
        ]
        self._trial_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed WuChunkingGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in self.keys:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid response <<{}>>.".format(submitted), 0.0, False, True, info
        correct = submitted == trial.instructed_key
        reward = 1.0 if correct else 0.0
        feedback = (
            "The instruction is to press {}, you press <<{}>> in {} ms. That is {}."
        ).format(trial.instructed_key, submitted, trial.rt_ms, "correct" if correct else "incorrect")
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _ChunkingTrial) -> str:
        return "The instruction is to press {}.".format(trial.instructed_key)

    def _info(self, trial: Optional[_ChunkingTrial], selected_action: Optional[str]) -> Dict[str, Any]:
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
            info["correct_action"] = trial.instructed_key
            info["rt_ms"] = trial.rt_ms
        return info
