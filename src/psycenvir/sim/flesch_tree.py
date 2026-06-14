"""Recorded-path Flesch tree-planting task."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import FleschTreeTrial


class FleschTreeRecordedEnv:
    """Exact-transition tree planting with explicit or pooled counterfactual feedback."""

    def __init__(
        self,
        experiment_id: str,
        trials: Iterable[FleschTreeTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.trials: List[FleschTreeTrial] = list(trials)
        if not self.trials:
            raise ValueError("FleschTreeRecordedEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.trials[0].observation),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed FleschTreeRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid tree action <<{}>>.".format(submitted), 0.0, False, True, info

        if not trial.has_feedback or trial.outcomes_by_action is None:
            if submitted != trial.human_action:
                self._done = True
                info = self._info(trial, None)
                info["unsupported_counterfactual"] = "no_feedback_trial"
                return (
                    "No recorded counterfactual outcomes are available for this no-feedback trial.",
                    0.0,
                    False,
                    True,
                    info,
                )
            self._trial_idx += 1
            self._done = self._trial_idx >= len(self.trials)
            observation = "" if self._done else self.trials[self._trial_idx].observation
            return observation, 0.0, self._done, False, self._info(trial, submitted)

        accept_key, reject_key = trial.valid_actions
        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward
        alt_key = reject_key if submitted == accept_key else accept_key
        alt_reward = float(trial.outcomes_by_action[alt_key])
        if submitted == accept_key:
            alt_clause = "rejected to plant the tree"
        else:
            alt_clause = "accepted to plant the tree"
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
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(
        self, trial: Optional[FleschTreeTrial], selected_action: Optional[str]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
        return info
