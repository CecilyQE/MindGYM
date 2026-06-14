"""Generative competitive volatile bandit (Gershman 2018 deconstructing exp2)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION
from psycenvir.generative.volatile_bandit import VOLATILE_OUTCOMES

GERSHMAN_DECONSTRUCT_EXP2_ID = "gershman2018deconstructing/exp2.csv"


@dataclass(frozen=True)
class _CompetitiveTrial:
    game_number: int
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, int]
    show_game_header: bool = False


class GershmanCompetitiveBanditGenerativeEnv:
    """Fresh games with two volatile arms; one arm is stochastically better."""

    def __init__(
        self,
        experiment_id: str = GERSHMAN_DECONSTRUCT_EXP2_ID,
        n_games: int = 20,
        trials_per_game: int = 10,
        better_shift: int = 4,
        instruction: str = GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_games = n_games
        self.trials_per_game = trials_per_game
        self.better_shift = better_shift
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_CompetitiveTrial] = []
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
            better_label, worse_label = labels[0], labels[1]
            for trial_number in range(self.trials_per_game):
                worse_outcome = self._rng.choice(VOLATILE_OUTCOMES)
                better_outcome = worse_outcome + self._rng.randint(1, self.better_shift)
                self._trials.append(
                    _CompetitiveTrial(
                        game_number=game_number,
                        valid_actions=(better_label, worse_label),
                        outcomes_by_action={
                            better_label: better_outcome,
                            worse_label: worse_outcome,
                        },
                        show_game_header=trial_number == 0,
                    )
                )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        header = "Game {}:".format(self._trials[0].game_number) if self._trials[0].show_game_header else ""
        return (
            render_initial_observation(self.instruction, header),
            self._info(self._trials[0], None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "Cannot step a completed GershmanCompetitiveBanditGenerativeEnv; call reset()."
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
            if next_trial.show_game_header:
                feedback = "{}\nGame {}:".format(feedback, next_trial.game_number)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[_CompetitiveTrial], selected_action: Optional[str]) -> Dict[str, Any]:
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
