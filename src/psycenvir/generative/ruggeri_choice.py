"""Generative no-feedback choice task from Ruggeri et al."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

RUGGERI_GLOBALIZABILITY_EXP1_ID = "ruggeri2022globalizability/exp1.csv"
RUGGERI_INSTRUCTION = (
    "In the following you will be presented with multiple choices between two options G and C.\n"
    "Please name which option you would prefer by pressing the corresponding key."
)


@dataclass(frozen=True)
class _RuggeriChoice:
    option_g: str
    option_c: str


DEFAULT_CHOICES: Tuple[_RuggeriChoice, ...] = (
    _RuggeriChoice("receiving 500$ immediately", "receiving 550$ in one year"),
    _RuggeriChoice("receiving 500$ immediately", "receiving 600$ in one year"),
    _RuggeriChoice("paying 500$ immediately", "paying 550$ in one year"),
    _RuggeriChoice("paying 500$ immediately", "paying 510$ in one year"),
    _RuggeriChoice("receiving 5000$ immediately", "receiving 5500$ in one year"),
    _RuggeriChoice("receiving 5000$ immediately", "receiving 6000$ in one year"),
    _RuggeriChoice("paying 5000$ immediately", "paying 5500$ in one year"),
    _RuggeriChoice("paying 5000$ immediately", "paying 5100$ in one year"),
    _RuggeriChoice("a sure gain of 500$", "a 50% chance to gain 1200$"),
    _RuggeriChoice("a sure loss of 500$", "a 50% chance to lose 1200$"),
)


class RuggeriGlobalizabilityGenerativeEnv:
    """Fresh preference-choice episode with no objective reward feedback."""

    def __init__(
        self,
        choices: Tuple[_RuggeriChoice, ...] = DEFAULT_CHOICES,
        action_keys: Tuple[str, str] = ("G", "C"),
        instruction: str = RUGGERI_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        del seed
        self.experiment_id = RUGGERI_GLOBALIZABILITY_EXP1_ID
        self.choices: List[_RuggeriChoice] = list(choices)
        self.action_keys = tuple(key.upper() for key in action_keys)
        self.valid_actions = self.action_keys
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_choice(self.choices[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed RuggeriGlobalizabilityGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.choices[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid response <<{}>>.".format(submitted), 0.0, False, True, info
        feedback = "{} You press <<{}>>.".format(self._render_choice(trial), submitted)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.choices)
        if not self._done:
            feedback = "{}\n{}".format(feedback, self._render_choice(self.choices[self._trial_idx]))
        return feedback, 0.0, self._done, False, self._info(trial, submitted)

    def _render_choice(self, trial: _RuggeriChoice) -> str:
        return "You have the choice between {} (press {}) or {} (press {}).".format(
            trial.option_g, self.action_keys[0], trial.option_c, self.action_keys[1]
        )

    def _info(self, trial: Optional[_RuggeriChoice], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "feedback_causal": False,
            "reward_defined": False,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["choice"] = {self.action_keys[0]: trial.option_g, self.action_keys[1]: trial.option_c}
        return info
