"""Generative volatile vs fixed slot machine task (Gershman 2018 deconstructing)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import GERSHMAN_DECONSTRUCT_INSTRUCTION

GERSHMAN_DECONSTRUCT_EXP1_ID = "gershman2018deconstructing/exp1.csv"
VOLATILE_OUTCOMES = (-4, -2, -1, 0, 1, 2)


@dataclass(frozen=True)
class _VolatileTrial:
    game_number: int
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, int]
    show_game_header: bool = False


class GershmanVolatileBanditGenerativeEnv:
    """Fresh games with one volatile arm and one arm that always pays 0."""

    def __init__(
        self,
        experiment_id: str = GERSHMAN_DECONSTRUCT_EXP1_ID,
        n_games: int = 20,
        trials_per_game: int = 10,
        volatile_outcomes: Tuple[int, ...] = VOLATILE_OUTCOMES,
        instruction: str = GERSHMAN_DECONSTRUCT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_games = n_games
        self.trials_per_game = trials_per_game
        self.volatile_outcomes = volatile_outcomes
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_VolatileTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        for game_number in range(1, self.n_games + 1):
            labels = self._rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXY"), 2)
            volatile_label, fixed_label = labels[0], labels[1]
            for trial_number in range(self.trials_per_game):
                volatile_outcome = self._rng.choice(self.volatile_outcomes)
                self._trials.append(
                    _VolatileTrial(
                        game_number=game_number,
                        valid_actions=(volatile_label, fixed_label),
                        outcomes_by_action={
                            volatile_label: volatile_outcome,
                            fixed_label: 0,
                        },
                        show_game_header=trial_number == 0,
                    )
                )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(self._trials[0], None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "Cannot step a completed GershmanVolatileBanditGenerativeEnv; call reset()."
            )
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid slot action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        feedback = "You press <<{}>> and get {} points.".format(submitted, int(reward))
        if trial.show_game_header:
            feedback = "Game {}:\n{}".format(trial.game_number, feedback)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            next_trial = self._trials[self._trial_idx]
            next_line = self._render_trial(next_trial)
            feedback = "{}\n{}".format(feedback, next_line)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _VolatileTrial) -> str:
        if trial.show_game_header:
            return "Game {}:".format(trial.game_number)
        return ""

    def _info(self, trial: Optional[_VolatileTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_exact",
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
