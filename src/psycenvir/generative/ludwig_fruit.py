"""Generative Ludwig fruit-market task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation

LUDWIG_HUMAN_EXPERIMENT_IDS = (
    "ludwig2023human/exp0.csv",
    "ludwig2023human/exp1.csv",
    "ludwig2023human/exp2.csv",
)
LUDWIG_INSTRUCTION = (
    "You will repeatedly feed animals with fruits.\n"
    "Each fruit contains two vitamins and every animal has a two-entry preference vector.\n"
    "Your points are calculated as the dot product of the vitamin content with the current animal's preference.\n"
    "You buy fruits in a market by going left or right for two steps. Per round, you collect two fruits.\n"
    "The fruits in the market are rearranged after each block."
)

_FRUITS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("apple", (-1, -1)),
    ("orange", (0, 1)),
    ("kiwi", (1, 0)),
    ("grapes", (1, -1)),
    ("melon", (-1, 1)),
    ("pear", (0, -1)),
)
_ANIMALS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("crocodile", (1, 1)),
    ("zebra", (1, 0)),
    ("kangaroo", (-1, 0)),
    ("penguin", (-1, -1)),
    ("elephant", (-1, 1)),
    ("giraffe", (0, 1)),
    ("tiger", (1, -1)),
    ("bear", (0, -1)),
)
_KEYS_BY_EXPERIMENT = {
    "ludwig2023human/exp0.csv": ("I", "V"),
    "ludwig2023human/exp1.csv": ("L", "C"),
    "ludwig2023human/exp2.csv": ("B", "Q"),
}


@dataclass
class _LudwigTrial:
    animal: str
    preference: Tuple[int, int]
    market: Dict[Tuple[str, str], Tuple[str, Tuple[int, int]]]
    path: Tuple[str, ...] = ()
    collected: int = 0


class LudwigFruitMarketGenerativeEnv:
    """Fresh fruit-market blocks with dot-product rewards."""

    def __init__(
        self,
        experiment_id: str = LUDWIG_HUMAN_EXPERIMENT_IDS[0],
        n_blocks: int = 6,
        trials_per_block: int = 15,
        instruction: str = LUDWIG_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.n_blocks = n_blocks
        self.trials_per_block = trials_per_block
        self.left_key, self.right_key = _KEYS_BY_EXPERIMENT.get(experiment_id, ("L", "R"))
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_LudwigTrial] = []
        self._trial_idx = 0
        self._step_in_collection = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        for _ in range(self.n_blocks):
            market = self._sample_market()
            animals = [self._rng.choice(_ANIMALS) for _ in range(self.trials_per_block)]
            for animal, preference in animals:
                self._trials.append(_LudwigTrial(animal, preference, dict(market)))
        self._trial_idx = 0
        self._step_in_collection = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_trial(self._trials[0])),
            self._info(None, None),
        )

    def _sample_market(self) -> Dict[Tuple[str, str], Tuple[str, Tuple[int, int]]]:
        fruits = list(_FRUITS)
        self._rng.shuffle(fruits)
        paths = ((self.left_key, self.left_key), (self.left_key, self.right_key), (self.right_key, self.left_key), (self.right_key, self.right_key))
        return {path: fruits[index % len(fruits)] for index, path in enumerate(paths)}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed LudwigFruitMarketGenerativeEnv; call reset().")
        submitted = normalize_action(action).upper()
        if submitted not in (self.left_key, self.right_key):
            self._done = True
            info = self._info(self._trials[self._trial_idx], None)
            info["invalid_action"] = submitted
            return "Invalid market action <<{}>>.".format(submitted), 0.0, False, True, info
        trial = self._trials[self._trial_idx]
        trial.path = trial.path + (submitted,)
        if len(trial.path) < 2:
            return (
                "You press <<{}>> and continue through the market.".format(submitted),
                0.0,
                False,
                False,
                self._info(trial, submitted),
            )
        fruit, vitamins = trial.market[trial.path]
        reward = float(vitamins[0] * trial.preference[0] + vitamins[1] * trial.preference[1])
        self._points += reward
        trial.collected += 1
        feedback = (
            "You have to feed the {}. It has the preference [{} {}]. "
            "You press <<{}>> and find the {} which has the vitamins [{} {}]. "
            "You get {} points."
        ).format(
            trial.animal,
            trial.preference[0],
            trial.preference[1],
            submitted,
            fruit,
            vitamins[0],
            vitamins[1],
            int(reward),
        )
        trial.path = ()
        if trial.collected >= 2:
            self._trial_idx += 1
            self._done = self._trial_idx >= len(self._trials)
            if not self._done:
                feedback = "{}\n\n{}".format(feedback, self._render_trial(self._trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _render_trial(self, trial: _LudwigTrial) -> str:
        return "You have to feed the {}. It has the preference [{} {}].".format(
            trial.animal, trial.preference[0], trial.preference[1]
        )

    def _info(self, trial: Optional[_LudwigTrial], selected_action: Optional[str]) -> Dict[str, Any]:
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
            info["market"] = {"/".join(path): value for path, value in trial.market.items()}
            info["preference"] = trial.preference
        return info
