"""Generative hazard-rate two-arm bandit task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

XIONG_NEURAL_EXP1_ID = "xiong2023neural/exp1.csv"
XIONG_HAZARD_INSTRUCTION = (
    "You are participating in multiple games involving two slot machines, labeled M and V.\n"
    "Each time you choose a slot machine, you get points.\n"
    "The expected points change randomly, abruptly, and independently with a hazard rate.\n"
    "When the points change, the new expected point value assigned to that slot machine is sampled "
    "from a uniform distribution from 1 to 99 points.\n"
    "Your goal is to choose the slot machine that will give you the most points."
)


@dataclass(frozen=True)
class _HazardTrial:
    game_number: int
    hazard_rate: float
    means: Dict[str, int]


class XiongHazardBanditGenerativeEnv:
    """Fresh restless bandit with independent hazard resets for each arm."""

    def __init__(
        self,
        arms: Tuple[str, str] = ("M", "V"),
        hazard_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
        games_per_hazard: int = 12,
        trials_per_game: int = 100,
        payoff_sd: float = 4.0,
        instruction: str = XIONG_HAZARD_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if len(arms) != 2 or len(set(arms)) != 2:
            raise ValueError("XiongHazardBanditGenerativeEnv requires two distinct arms.")
        self.experiment_id = XIONG_NEURAL_EXP1_ID
        self.arms = tuple(arm.upper() for arm in arms)
        self.hazard_rates = hazard_rates
        self.games_per_hazard = games_per_hazard
        self.trials_per_game = trials_per_game
        self.payoff_sd = payoff_sd
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_HazardTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        game_number = 1
        for hazard_rate in self.hazard_rates:
            for _ in range(self.games_per_hazard):
                means = {arm: self._rng.randint(1, 99) for arm in self.arms}
                for _ in range(self.trials_per_game):
                    means = {
                        arm: self._rng.randint(1, 99) if self._rng.random() < hazard_rate else mean
                        for arm, mean in means.items()
                    }
                    self._trials.append(_HazardTrial(game_number, hazard_rate, dict(means)))
                game_number += 1
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed XiongHazardBanditGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in self.arms:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid action <<{}>>.".format(submitted), 0.0, False, True, info
        reward = float(max(1, min(99, int(round(self._rng.gauss(trial.means[submitted], self.payoff_sd))))))
        self._points += reward
        feedback = "You press <<{}>> and get {} points.".format(submitted, int(reward))
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _HazardTrial) -> str:
        return "Game {}. The hazard rate is {}. There are {} trials in this game.".format(
            trial.game_number, trial.hazard_rate, self.trials_per_game
        )

    def _info(self, trial: Optional[_HazardTrial], selected_arm: Optional[str]) -> Dict[str, Any]:
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
        if trial is not None:
            info["hazard_rate"] = trial.hazard_rate
            if self.include_human_ref:
                info["means_by_action"] = dict(trial.means)
        return info
