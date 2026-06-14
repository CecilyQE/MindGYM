"""Exact-transition judgment-learning simulation for Collsio et al. transcripts."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import JudgmentTrial


class CollsioJudgmentEnv:
    """Caldionine judgment task with label-driven feedback on each source trial."""

    EXPERIMENT_ID = "collsiöö2023MCPL/exp3.csv"

    def __init__(
        self,
        trials: Iterable[JudgmentTrial],
        valid_actions: Iterable[str],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.trials: List[JudgmentTrial] = list(trials)
        if not self.trials:
            raise ValueError("CollsioJudgmentEnv requires at least one trial.")
        self.valid_actions = set(valid_actions)
        if not self.valid_actions:
            raise ValueError("CollsioJudgmentEnv requires declared response values.")
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
            render_initial_observation(self.instruction, self._render_stimulus(self.trials[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed CollsioJudgmentEnv; call reset().")
        submitted = normalize_action(action)
        trial = self.trials[self._trial_idx]
        if submitted not in self.valid_actions:
            self._done = True
            info = self._info(trial, False)
            info["invalid_action"] = submitted
            return "Invalid Caldionine response <<{}>>.".format(submitted), 0.0, False, True, info
        is_correct = submitted == trial.correct_action
        reward = 1.0 if is_correct else 0.0
        self._points += reward
        evaluation = "correct" if is_correct else "incorrect"
        feedback = (
            "You say that the Caldionine concentration is <<{}>>. That is {}. "
            "The correct concentration of Caldionine is {}."
        ).format(submitted, evaluation, trial.correct_action)
        self._trial_idx += 1
        self._done = self._trial_idx >= len(self.trials)
        if not self._done:
            feedback = "{} {}".format(feedback, self._render_stimulus(self.trials[self._trial_idx]))
        return feedback, reward, self._done, False, self._info(trial, is_correct)

    @staticmethod
    def _render_stimulus(trial: JudgmentTrial) -> str:
        return "Progladine: {}. Amalydine: {}.".format(trial.progladine, trial.amalydine)

    def _info(self, trial: Optional[JudgmentTrial], is_correct: Optional[bool]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "action_space_validated": True,
            "instruction_shown": bool(self.instruction),
        }
        if trial is not None:
            info["correct_action"] = trial.correct_action
            info["is_correct"] = is_correct
            if self.include_human_ref and trial.human_action is not None:
                info["human_ref"] = trial.human_action
        return info
