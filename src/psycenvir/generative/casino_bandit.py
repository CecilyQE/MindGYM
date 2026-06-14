"""Generative casino two-arm bandit (Lefebvre et al. style)."""

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.dynamics import sample_discrete_outcome
from psycenvir.generative.instructions import LEFEBVRE_INSTRUCTION

LEFEBVRE_EXP1_ID = "lefebvre2017behavioural/exp1.csv"
LEFEBVRE_EXP2_ID = "lefebvre2017behavioural/exp2.csv"


@dataclass(frozen=True)
class _CasinoTrial:
    casino_id: int
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]


class CasinoBanditGenerativeEnv:
    """Fresh casino visits with stochastic 0 / 0.5 point machines."""

    def __init__(
        self,
        experiment_id: str = LEFEBVRE_EXP1_ID,
        n_casinos: int = 4,
        visits_per_casino: int = 24,
        outcome_values: Tuple[float, float] = (0.0, 0.5),
        instruction: str = LEFEBVRE_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_casinos = n_casinos
        self.visits_per_casino = visits_per_casino
        self.outcome_values = outcome_values
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_CasinoTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        casinos: Dict[int, Tuple[Tuple[str, str], Dict[str, float]]] = {}
        for casino in range(1, self.n_casinos + 1):
            keys = tuple(sorted(self._rng.sample(list(string.ascii_uppercase), 2)))
            casinos[casino] = (
                keys,
                {
                    keys[0]: self._rng.uniform(0.05, 0.95),
                    keys[1]: self._rng.uniform(0.05, 0.95),
                },
            )
        self._trials = []
        for casino in range(1, self.n_casinos + 1):
            keys, machine_probs = casinos[casino]
            for _ in range(self.visits_per_casino):
                p_a = machine_probs[keys[0]]
                p_b = machine_probs[keys[1]]
                self._trials.append(
                    _CasinoTrial(
                        casino_id=casino,
                        valid_actions=(keys[0], keys[1]),
                        outcomes_by_action={
                            keys[0]: sample_discrete_outcome(
                                self._rng, list(self.outcome_values), [1.0 - p_a, p_a]
                            ),
                            keys[1]: sample_discrete_outcome(
                                self._rng, list(self.outcome_values), [1.0 - p_b, p_b]
                            ),
                        },
                    )
                )
        visit_order = list(self._trials)
        self._rng.shuffle(visit_order)
        self._trials = visit_order
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed CasinoBanditGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = trial.outcomes_by_action[submitted]
        self._points += reward
        feedback = (
            "You go to casino {}. You can choose between machines {} and {}. "
            "You press <<{}>> and receive {} points."
        ).format(
            trial.casino_id,
            trial.valid_actions[0],
            trial.valid_actions[1],
            submitted,
            reward,
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    @staticmethod
    def _render_trial(trial: _CasinoTrial) -> str:
        return (
            "You go to casino {}. You can choose between machines {} and {}."
        ).format(trial.casino_id, trial.valid_actions[0], trial.valid_actions[1])

    def _info(self, trial: Optional[_CasinoTrial], selected_arm: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_arm": selected_arm,
        }
        if trial is not None and self.include_human_ref:
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
