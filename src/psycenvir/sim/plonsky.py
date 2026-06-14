"""Recorded-path Plonsky gambling problems with counterfactual feedback."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import PlonskyGambleTrial


class PlonskyGambleRecordedEnv:
    """Exact-transition gambling with explicit counterfactual feedback when recorded."""

    EXPERIMENT_ID = "plonsky2018when/exp1.csv"

    def __init__(
        self,
        trials: Iterable[PlonskyGambleTrial],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[PlonskyGambleTrial] = list(trials)
        if not self.trials:
            raise ValueError("PlonskyGambleRecordedEnv requires at least one trial.")
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
            raise RuntimeError("Cannot step a completed PlonskyGambleRecordedEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info

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
            if self.instruction and self._trial_idx == 1:
                observation = render_initial_observation(self.instruction, observation)
            return observation, 0.0, self._done, False, self._info(trial, submitted)

        chosen = trial.outcomes_by_action[submitted]
        alternative = trial.valid_actions[1] if submitted == trial.valid_actions[0] else trial.valid_actions[0]
        forgone = trial.outcomes_by_action[alternative]
        reward = float(chosen)
        self._points += reward
        if chosen >= 0:
            chosen_text = "gain {} points".format(int(chosen) if chosen.is_integer() else chosen)
        else:
            chosen_text = "lose {} points".format(int(abs(chosen)) if chosen.is_integer() else abs(chosen))
        if forgone >= 0:
            forgone_text = "gained {} points".format(int(forgone) if forgone.is_integer() else forgone)
        else:
            forgone_text = "lost {} points".format(int(abs(forgone)) if forgone.is_integer() else abs(forgone))
        feedback = (
            "{}\nYou press <<{}>> and {}. You would have {} had you chosen option {}."
        ).format(trial.observation, submitted, chosen_text, forgone_text, alternative)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self.trials[self._trial_idx].observation)
        return feedback, reward, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[PlonskyGambleTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None:
            info["has_feedback"] = trial.has_feedback
            if self.include_human_ref:
                info["human_ref"] = trial.human_action
                if trial.outcomes_by_action is not None:
                    info["outcomes_by_action"] = dict(trial.outcomes_by_action)
        return info
