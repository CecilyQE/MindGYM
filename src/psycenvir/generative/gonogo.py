"""Generative Enkavi go/no-go task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import render_initial_observation
from psycenvir.errors import InvalidActionError
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import GONOGO_INSTRUCTION
from psycenvir.models import GONOGO_NO_PRESS

ENKAVI_GONOGO_EXP1_ID = "enkavi2019gonogo/exp1.csv"


@dataclass(frozen=True)
class _GonogoStep:
    stimulus: str
    go_key: str


class EnkaviGonogoGenerativeEnv:
    """Fresh go/no-go blocks with transcript-calibrated keys and trial counts."""

    def __init__(
        self,
        experiment_id: str = ENKAVI_GONOGO_EXP1_ID,
        n_practice_trials: int = 10,
        n_test_trials: int = 350,
        go_keys: Optional[List[str]] = None,
        colour1_probability: Optional[float] = None,
        instruction: str = GONOGO_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_practice_trials = n_practice_trials
        self.n_test_trials = n_test_trials
        self._go_keys = list(go_keys or calibration.get("go_keys") or ["X"])
        self.colour1_probability = float(
            colour1_probability
            if colour1_probability is not None
            else calibration.get("colour1_probability", 0.5)
        )
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._steps: List[_GonogoStep] = []
        self._step_idx = 0
        self._go_key = self._go_keys[0]
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._go_key = self._rng.choice(self._go_keys)
        self._steps = []
        for is_practice in (True, False):
            count = self.n_practice_trials if is_practice else self.n_test_trials
            for _ in range(count):
                stimulus = (
                    "colour1"
                    if self._rng.random() < self.colour1_probability
                    else "colour2"
                )
                self._steps.append(_GonogoStep(stimulus=stimulus, go_key=self._go_key))
        self._step_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_step(self._steps[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed EnkaviGonogoGenerativeEnv; call reset().")
        submitted = self._coerce_action(action)
        step = self._steps[self._step_idx]
        if submitted not in (step.go_key, None):
            self._done = True
            info = self._info(step, False)
            info["invalid_action"] = action
            return "Invalid go/no-go action <<{}>>.".format(action), 0.0, False, True, info

        is_correct = (
            submitted == step.go_key if step.stimulus == "colour1" else submitted is None
        )
        reward = 1.0 if is_correct else 0.0
        rt_ms = self._rng.uniform(250.0, 900.0)
        feedback = self._format_feedback(step.stimulus, submitted, rt_ms)
        self._step_idx += 1
        self._done = self._step_idx >= len(self._steps)
        observation = ""
        if not self._done:
            observation = self._render_step(self._steps[self._step_idx])
        return observation, reward, self._done, False, self._info(step, is_correct)

    def _coerce_action(self, action: str) -> Optional[str]:
        from psycenvir.core.base import normalize_action

        if isinstance(action, str) and action.strip().upper() == GONOGO_NO_PRESS:
            return None
        try:
            return normalize_action(action)
        except InvalidActionError:
            if not action.strip():
                return None
            raise

    @staticmethod
    def _format_feedback(
        stimulus: str, submitted: Optional[str], rt_ms: float
    ) -> str:
        if submitted is None:
            return "You see {} and press nothing.".format(stimulus)
        return "You see {} and press <<{}>> in {:.1f}ms.".format(stimulus, submitted, rt_ms)

    @staticmethod
    def _render_step(step: _GonogoStep) -> str:
        return "You see {}.".format(step.stimulus)

    def _info(self, step: Optional[_GonogoStep], is_correct: Optional[bool]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._step_idx,
            "go_key": self._go_key,
            "no_press_action": GONOGO_NO_PRESS,
            "feedback_causal": True,
            "reward_defined": True,
            "objective_accuracy_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
        }
        if step is not None:
            info["stimulus"] = step.stimulus
            if self.include_human_ref:
                info["is_correct"] = is_correct
        return info
