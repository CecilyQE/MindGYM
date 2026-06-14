"""Category-learning simulation for Badham et al. transcripts."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import CategoryTrial


class BadhamCategoryEnv:
    """Exact-transition category feedback environment.

    A trial's correct category comes from the stimulus/label record and never
    from the human action in the source transcript. Human actions can be
    exposed in ``info`` only when evaluating alignment.
    """

    def __init__(
        self,
        trials: Iterable[CategoryTrial],
        include_human_ref: bool = False,
        valid_actions: Optional[Iterable[str]] = None,
        instruction: str = "",
    ) -> None:
        self.trials: List[CategoryTrial] = list(trials)
        if not self.trials:
            raise ValueError("BadhamCategoryEnv requires at least one trial.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self.valid_actions = set(valid_actions) if valid_actions is not None else None
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_stimulus(self.trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed BadhamCategoryEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if self.valid_actions is not None and submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid category action <<{}>>.".format(submitted), 0.0, False, True, info

        is_correct = submitted == trial.correct_action
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        feedback = "You press <<{}>>. The correct category is {}.".format(
            submitted, trial.correct_action
        )
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{} {}".format(feedback, self._render_stimulus(self.trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, is_correct)

    @staticmethod
    def _render_stimulus(trial: CategoryTrial) -> str:
        stimulus = "You see {}.".format(trial.stimulus)
        if trial.observation_prefix:
            return "{} {}".format(trial.observation_prefix, stimulus)
        return stimulus

    def _info(
        self, trial: Optional[CategoryTrial], is_correct: Optional[bool]
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": "badham2017deficits/exp1.csv",
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "action_space_validated": self.valid_actions is not None,
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["correct_action"] = trial.correct_action
            info["is_correct"] = is_correct
            if self.include_human_ref and trial.human_action is not None:
                info["human_ref"] = trial.human_action
        return info
