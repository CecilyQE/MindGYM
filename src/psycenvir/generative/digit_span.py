"""Generative Enkavi digit-span recall."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import DIGIT_SPAN_INSTRUCTION

ENKAVI_DIGIT_SPAN_EXP1_ID = "enkavi2019digitspan/exp1.csv"


@dataclass(frozen=True)
class _DigitSpanStep:
    observation: str
    valid_actions: Tuple[str, ...]
    correct_action: str


class EnkaviDigitSpanGenerativeEnv:
    """Fresh digit spans with key-by-key recall feedback."""

    def __init__(
        self,
        experiment_id: str = ENKAVI_DIGIT_SPAN_EXP1_ID,
        n_spans: int = 5,
        min_length: int = 3,
        max_length: int = 5,
        end_key: Optional[str] = None,
        end_keys: Optional[List[str]] = None,
        instruction: str = DIGIT_SPAN_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        if min_length > max_length:
            raise ValueError("min_length cannot exceed max_length.")
        self.experiment_id = experiment_id
        self.n_spans = n_spans
        self.min_length = min_length
        self.max_length = max_length
        self._end_keys = list(end_keys or calibration.get("end_keys") or ["S"])
        self.end_key = end_key
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._steps: List[_DigitSpanStep] = []
        self._step_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        if self.end_key is None:
            self.end_key = self._rng.choice(self._end_keys)
        digit_actions = tuple(str(digit) for digit in range(10))
        valid_actions = digit_actions + (self.end_key,)
        self._steps = []
        for _ in range(self.n_spans):
            length = self._rng.randint(self.min_length, self.max_length)
            digits = [str(self._rng.randint(0, 9)) for _ in range(length)]
            sequence = digits + [self.end_key]
            header = "The digits are the following: [{}]".format(", ".join(digits))
            for position, digit in enumerate(sequence):
                observation = header if position == 0 else ""
                self._steps.append(
                    _DigitSpanStep(
                        observation=observation,
                        valid_actions=valid_actions,
                        correct_action=digit,
                    )
                )
        self._step_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._steps[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviDigitSpanGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        step = self._steps[self._step_idx]
        if submitted not in step.valid_actions:
            self._done = True
            info = self._info(step, False)
            info["invalid_action"] = submitted
            return "Invalid digit-span key <<{}>>.".format(submitted), 0.0, False, True, info

        is_correct = submitted == step.correct_action
        reward = 1.0 if is_correct else 0.0
        feedback = "You press <<{}>>.".format(submitted)
        self._step_idx += 1
        self._done = self._step_idx >= len(self._steps)
        observation = ""
        if not self._done and self._steps[self._step_idx].observation:
            observation = self._steps[self._step_idx].observation
        return observation, reward, self._done, False, self._info(step, is_correct)

    def _info(
        self, step: Optional[_DigitSpanStep], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._step_idx,
            "feedback_causal": True,
            "reward_defined": True,
            "objective_accuracy_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
        }
        if step is not None and is_correct is not None:
            info["is_correct"] = is_correct
            if self.include_human_ref:
                info["correct_action"] = step.correct_action
        return info
