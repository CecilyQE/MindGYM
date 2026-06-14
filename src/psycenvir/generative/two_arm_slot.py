"""Generative two-arm slot-machine tasks (Wilson et al. style)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 used by WILSON_EXPERIMENT_CONFIGS

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import WILSON_BANDIT_INSTRUCTION

WILSON_EXPERIMENT_ID = "wilson2014humans/exp1.csv"
DEFAULT_WILSON_ARMS = ("C", "A")
WILSON_EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "wilson2014humans/exp1.csv": {
        "arms": ("C", "A"),
        "n_games": 3,
        "instructed_trials": 20,
        "free_trials_choices": (30, 30),
    },
    "wilson2014humans/exp2.csv": {
        "arms": ("K", "W"),
        "n_games": 6,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "wilson2014humans/exp3.csv": {
        "arms": ("Z", "J"),
        "n_games": 6,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "wilson2014humans/exp4.csv": {
        "arms": ("X", "A"),
        "n_games": 6,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "wilson2014humans/exp5.csv": {
        "arms": ("F", "I"),
        "n_games": 6,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "feng2021dynamics/exp1.csv": {
        "arms": ("I", "H"),
        "n_games": 160,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "sadeghiyeh2020temporal/exp1.csv": {
        "arms": ("J", "R"),
        "n_games": 80,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "somerville2017charting/exp1.csv": {
        "arms": ("F", "N"),
        "n_games": 80,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
    "waltz2020differential/exp1.csv": {
        "arms": ("M", "U"),
        "n_games": 60,
        "instructed_trials": 4,
        "free_trials_choices": (1, 6),
    },
}


@dataclass(frozen=True)
class _SlotTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, int]
    instructed: bool = False


class TwoArmSlotGenerativeEnv:
    """Fresh two-arm bandit games with optional instructed prefixes."""

    def __init__(
        self,
        experiment_id: str = WILSON_EXPERIMENT_ID,
        arms: Tuple[str, str] = DEFAULT_WILSON_ARMS,
        n_games: int = 6,
        instructed_trials: int = 4,
        free_trials_choices: Tuple[int, int] = (1, 6),
        payoff_sd: float = 12.0,
        instruction: str = WILSON_BANDIT_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if len(arms) != 2 or len(set(arms)) != 2:
            raise ValueError("TwoArmSlotGenerativeEnv requires two distinct arm keys.")
        self.experiment_id = experiment_id
        self.arms: Tuple[str, str] = arms
        self.n_games = n_games
        self.instructed_trials = instructed_trials
        self.free_trials_choices = free_trials_choices
        self.payoff_sd = payoff_sd
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_SlotTrial] = []
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
            means = {arm: int(round(self._rng.gauss(60.0, 15.0))) for arm in self.arms}
            game_trials: List[_SlotTrial] = []
            for _ in range(self.instructed_trials):
                arm = self._rng.choice(self.arms)
                points = int(round(self._rng.gauss(means[arm], self.payoff_sd)))
                game_trials.append(
                    _SlotTrial(
                        observation=(
                            "Game {}. There are {} trials in this game.\n"
                            "You are instructed to press {} and get {} points."
                        ).format(
                            game_number,
                            self.instructed_trials + self._rng.choice(self.free_trials_choices),
                            arm,
                            points,
                        ),
                        valid_actions=(arm,),
                        outcomes_by_action={arm: points},
                        instructed=True,
                    )
                )
            free_trials = self._rng.choice(self.free_trials_choices)
            for trial_number in range(free_trials):
                game_trials.append(
                    _SlotTrial(
                        observation="Game {}. Trial {}.".format(game_number, trial_number + 1),
                        valid_actions=self.arms,
                        outcomes_by_action={
                            arm: int(round(self._rng.gauss(means[arm], self.payoff_sd)))
                            for arm in self.arms
                        },
                    )
                )
            self._trials.extend(game_trials)
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed TwoArmSlotGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        if trial.instructed:
            feedback = "You are instructed to press {} and get {} points.".format(
                submitted, int(reward)
            )
        else:
            feedback = "You press <<{}>> and get {} points.".format(submitted, int(reward))
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[_SlotTrial], selected_arm: Optional[str]) -> Dict[str, Any]:
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
