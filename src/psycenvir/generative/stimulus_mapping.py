"""Generative stimulus-to-response mapping task (Gershman et al. style)."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.instructions import GERSHMAN_MAPPING_INSTRUCTION

GERSHMAN_EXPERIMENT_ID = "gershman2020reward/exp1.csv"
DEFAULT_RESPONSES = ("S", "F", "A")


@dataclass(frozen=True)
class _MappingTrial:
    game_number: int
    stimulus_id: int
    correct_action: str
    valid_actions: Tuple[str, ...]
    show_game_header: bool = False


class GershmanMappingGenerativeEnv:
    """Fresh games with independent stimulus-response mappings."""

    def __init__(
        self,
        experiment_id: str = GERSHMAN_EXPERIMENT_ID,
        n_games: int = 13,
        n_stimuli: int = 6,
        trials_per_game: int = 30,
        valid_actions: Tuple[str, ...] = DEFAULT_RESPONSES,
        instruction: str = GERSHMAN_MAPPING_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if len(set(valid_actions)) != len(valid_actions):
            raise ValueError("GershmanMappingGenerativeEnv requires distinct response keys.")
        self.experiment_id = experiment_id
        self.n_games = n_games
        self.n_stimuli = n_stimuli
        self.trials_per_game = trials_per_game
        self.valid_actions = valid_actions
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_MappingTrial] = []
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
            mapping = {
                stimulus_id: self._rng.choice(self.valid_actions)
                for stimulus_id in range(self.n_stimuli)
            }
            for trial_number in range(self.trials_per_game):
                stimulus_id = self._rng.randrange(self.n_stimuli)
                self._trials.append(
                    _MappingTrial(
                        game_number=game_number,
                        stimulus_id=stimulus_id,
                        correct_action=mapping[stimulus_id],
                        valid_actions=self.valid_actions,
                        show_game_header=trial_number == 0,
                    )
                )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed GershmanMappingGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid response <<{}>>.".format(submitted), 0.0, False, True, info

        reward = 1.0 if submitted == trial.correct_action else 0.0
        self._points += reward
        feedback = "You see stimulus {}. You press <<{}>> and get {} points.".format(
            trial.stimulus_id,
            submitted,
            int(reward),
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _MappingTrial) -> str:
        if trial.show_game_header:
            return (
                "Game {}:\nThere are {} different stimuli.\nYou see stimulus {}."
            ).format(trial.game_number, self.n_stimuli, trial.stimulus_id)
        return "You see stimulus {}.".format(trial.stimulus_id)

    def _info(self, trial: Optional[_MappingTrial], selected_action: Optional[str]) -> Dict[str, Any]:
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
            info["correct_action"] = trial.correct_action
        return info
