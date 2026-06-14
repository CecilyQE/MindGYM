"""Recorded-feedback simulation for choices13k gamble blocks."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import PetersonGambleBlock, PetersonGambleTrial


class PetersonRecordedFeedbackEnv:
    """Recorded Peterson blocks: full feedback (counterfactual) and no-feedback (press only)."""

    EXPERIMENT_ID = "peterson2021using/exp1.csv"

    def __init__(
        self,
        blocks: Iterable[PetersonGambleBlock],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.blocks: List[PetersonGambleBlock] = list(blocks)
        if not self.blocks:
            raise ValueError("PetersonRecordedFeedbackEnv requires at least one block.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self.omitted_no_feedback_blocks = sum(
            1 for block in self.blocks if not block.has_feedback
        )
        self._block_idx = 0
        self._block_trial_idx = 0
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._block_idx = 0
        self._block_trial_idx = 0
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self.blocks[0].observation),
            self._info(None, None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed PetersonRecordedFeedbackEnv; call reset().")
        submitted = normalize_action(action)
        block = self.blocks[self._block_idx]
        trial = block.trials[self._block_trial_idx]
        if submitted not in block.valid_actions:
            self._done = True
            info = self._info(block, trial, None)
            info["invalid_action"] = submitted
            return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info

        if not block.has_feedback:
            if submitted != trial.human_action:
                self._done = True
                info = self._info(block, trial, None)
                info["unsupported_counterfactual"] = "no_feedback_block"
                return (
                    "No counterfactual outcomes are available for this no-feedback block.",
                    0.0,
                    False,
                    True,
                    info,
                )
            reward = 0.0
            observation = "You press <<{}>>.".format(submitted)
        else:
            assert trial.outcomes_by_action is not None
            alternative = (
                block.valid_actions[1]
                if submitted == block.valid_actions[0]
                else block.valid_actions[0]
            )
            selected_outcome = trial.outcomes_by_action[submitted]
            forgone_outcome = trial.outcomes_by_action[alternative]
            reward = selected_outcome
            self._points += reward
            observation = (
                "You press <<{}>>. You receive {} points by selecting this option. "
                "You would have received {} points had you chosen the other option."
            ).format(submitted, selected_outcome, forgone_outcome)

        self._trial_idx += 1
        self._block_trial_idx += 1
        if self._block_trial_idx >= len(block.trials):
            self._block_idx += 1
            self._block_trial_idx = 0
        self._done = self._block_idx >= len(self.blocks)
        if not self._done and self._block_trial_idx == 0:
            observation = "{}\n\n{}".format(observation, self.blocks[self._block_idx].observation)
        selected = selected_outcome if block.has_feedback else None
        return observation, reward, self._done, False, self._info(block, trial, selected)

    def _info(
        self,
        block: Optional[PetersonGambleBlock],
        trial: Optional[PetersonGambleTrial],
        selected_outcome: Optional[float],
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.EXPERIMENT_ID,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition_on_recorded_trials",
            "source_feedback_only": True,
            "omitted_no_feedback_blocks": self.omitted_no_feedback_blocks,
            "instruction_shown": bool(self.instruction),
        }
        if block is not None:
            info["source_block_idx"] = block.source_block_idx
        if selected_outcome is not None:
            info["selected_outcome"] = selected_outcome
        if trial is not None and self.include_human_ref:
            info["human_ref"] = trial.human_action
        return info
