"""Generative Flesch tree-planting task."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.grounding import get_grounding_profile
from psycenvir.generative.instructions import FLESCH_TREE_INSTRUCTION

FLESCH_TREE_EXP1_ID = "flesch2018comparing/exp1.csv"


@dataclass(frozen=True)
class _FleschTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    has_feedback: bool


class FleschTreeGenerativeEnv:
    """Fresh North/South tree-planting blocks with training then silent test trials."""

    def __init__(
        self,
        experiment_id: str = FLESCH_TREE_EXP1_ID,
        n_training_trials: int = 20,
        n_test_trials: int = 10,
        accept_key: Optional[str] = None,
        reject_key: Optional[str] = None,
        response_key_pairs: Optional[List[Tuple[str, str]]] = None,
        instruction: str = FLESCH_TREE_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
        schedule_trials: Optional[List[_FleschTrial]] = None,
        transcript_bound: bool = False,
    ) -> None:
        self._schedule_trials = list(schedule_trials) if schedule_trials is not None else None
        self.transcript_bound = transcript_bound
        calibration = get_grounding_profile(experiment_id).default_config
        self.experiment_id = experiment_id
        self.n_training_trials = n_training_trials
        self.n_test_trials = n_test_trials
        raw_pairs = response_key_pairs or calibration.get("response_key_pairs") or []
        self._response_key_pairs = [
            (pair[0], pair[1]) if isinstance(pair, list) else pair for pair in raw_pairs
        ]
        if not self._response_key_pairs:
            self._response_key_pairs = [("T", "N")]
        self.accept_key = accept_key
        self.reject_key = reject_key
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_FleschTrial] = []
        self._trial_idx = 0
        self._visible_points = 0.0
        self._evaluator_points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if self._schedule_trials is not None:
            del seed
            self._trials = list(self._schedule_trials)
            self._trial_idx = 0
            self._visible_points = 0.0
            self._evaluator_points = 0.0
            self._done = False
            if self._trials:
                self.accept_key, self.reject_key = self._trials[0].valid_actions
            return (
                render_initial_observation(self.instruction, self._trials[0].observation),
                self._info(None, None),
            )
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        if self.accept_key is None or self.reject_key is None:
            self.accept_key, self.reject_key = self._rng.choice(self._response_key_pairs)
        self._trials = []
        for garden in ("North", "South"):
            best_leaf = self._rng.randint(0, 4)
            best_branch = self._rng.randint(0, 4)
            for _ in range(self.n_training_trials // 2):
                leaf = self._rng.randint(0, 4)
                branch = self._rng.randint(0, 4)
                optimal = leaf == best_leaf and branch == best_branch
                accept_reward = self._rng.choice([50.0, 25.0]) if optimal else self._rng.choice(
                    [-50.0, -25.0, 0.0]
                )
                self._trials.append(
                    self._make_trial(garden, leaf, branch, accept_reward, has_feedback=True)
                )
            for _ in range(self.n_test_trials // 2):
                leaf = self._rng.randint(0, 4)
                branch = self._rng.randint(0, 4)
                optimal = leaf == best_leaf and branch == best_branch
                accept_reward = 50.0 if optimal else -50.0
                self._trials.append(
                    self._make_trial(garden, leaf, branch, accept_reward, has_feedback=False)
                )
        self._trial_idx = 0
        self._visible_points = 0.0
        self._evaluator_points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._trials[0].observation),
            self._info(None, None),
        )

    def _make_trial(
        self, garden: str, leaf: int, branch: int, accept_reward: float, has_feedback: bool
    ) -> _FleschTrial:
        observation = (
            "You get a tree with level {} of leafiness and level {} of branchiness in the {} garden."
        ).format(leaf, branch, garden)
        outcomes = {self.accept_key: accept_reward, self.reject_key: 0.0}
        return _FleschTrial(
            observation=observation,
            valid_actions=(self.accept_key, self.reject_key),
            outcomes_by_action=outcomes,
            has_feedback=has_feedback,
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed FleschTreeGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(None, None)
            info["invalid_action"] = submitted
            return "Invalid tree action <<{}>>.".format(submitted), 0.0, False, True, info

        latent_reward = float(trial.outcomes_by_action[submitted])
        self._evaluator_points += latent_reward
        if not trial.has_feedback:
            feedback = "{} You press <<{}>>.".format(trial.observation, submitted)
            reward = 0.0
        else:
            reward = latent_reward
            self._visible_points += latent_reward
            alt_key = self.reject_key if submitted == self.accept_key else self.accept_key
            alt_reward = float(trial.outcomes_by_action[alt_key])
            alt_clause = (
                "rejected to plant the tree"
                if submitted == self.accept_key
                else "accepted to plant the tree"
            )
            feedback = (
                "{} You press <<{}>> and get {} points. You would have gotten {} points, had you {}."
            ).format(
                trial.observation,
                submitted,
                int(reward) if reward.is_integer() else reward,
                int(alt_reward) if alt_reward.is_integer() else alt_reward,
                alt_clause,
            )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[_FleschTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._visible_points,
            "evaluator_points": self._evaluator_points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "generative_transcript_calibrated",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None:
            info["has_feedback"] = trial.has_feedback
            if selected_action is not None and trial.has_feedback:
                info["latent_reward"] = float(trial.outcomes_by_action[selected_action])
        return info
