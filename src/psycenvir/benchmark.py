"""Evaluation runners that isolate policy context from causal task transitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from psycenvir.core.base import normalize_action
from psycenvir.psych101.parse import BADHAM_TRIAL_RE, parse_badham_category_trials, parse_instruction_prefix
from psycenvir.sim.category import BadhamCategoryEnv


BenchmarkPolicy = Callable[[str, Dict[str, Any]], str]
BADHAM_ID = "badham2017deficits/exp1.csv"


class ContextCondition(str, Enum):
    FULL = "full"
    NO_INSTRUCTION = "no_instruction"
    NO_HISTORY = "no_history"
    PERMUTED_FEEDBACK = "permuted_feedback"


@dataclass(frozen=True)
class BadhamBenchmarkResult:
    condition: ContextCondition
    actions: List[str]
    rewards: List[float]
    human_matches: List[bool]
    prompts: List[str]
    terminated: bool
    truncated: bool
    total_trials: int
    feedback_perturbed: bool

    @property
    def task_accuracy(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def human_match_rate(self) -> float:
        return sum(self.human_matches) / len(self.human_matches) if self.human_matches else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "condition": self.condition.value,
            "actions": self.actions,
            "evaluated_trials": len(self.actions),
            "total_trials": self.total_trials,
            "task_accuracy": self.task_accuracy,
            "human_match_rate": self.human_match_rate,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "feedback_perturbed": self.feedback_perturbed,
        }


def _render_trial_state(trial: Any) -> str:
    state = "You see {}.".format(trial.stimulus)
    return "{} {}".format(trial.observation_prefix, state).strip()


def _feedback_mapping(valid_actions: Sequence[str]) -> Dict[str, str]:
    ordered = sorted(set(valid_actions))
    if len(ordered) <= 1:
        return {action: action for action in ordered}
    return {action: ordered[-index - 1] for index, action in enumerate(ordered)}


def _render_badham_prompt(
    instruction: str,
    trials: Sequence[Any],
    submitted_actions: Sequence[str],
    trial_idx: int,
    condition: ContextCondition,
    feedback_mapping: Dict[str, str],
) -> str:
    sections: List[str] = []
    if condition != ContextCondition.NO_INSTRUCTION and instruction:
        sections.append(instruction)
    if condition != ContextCondition.NO_HISTORY:
        for previous_idx, action in enumerate(submitted_actions):
            correct_action = trials[previous_idx].correct_action
            if condition == ContextCondition.PERMUTED_FEEDBACK:
                correct_action = feedback_mapping[correct_action]
            sections.append(
                "{} You press <<{}>>. The correct category is {}.".format(
                    _render_trial_state(trials[previous_idx]), action, correct_action
                )
            )
    sections.append(_render_trial_state(trials[trial_idx]))
    return "\n\n".join(section for section in sections if section)


def run_badham_benchmark(
    text: str,
    policy: BenchmarkPolicy,
    condition: ContextCondition = ContextCondition.FULL,
    max_trials: Optional[int] = None,
) -> BadhamBenchmarkResult:
    """Run a Badham policy under a controlled context condition.

    The policy receives only action-space and condition metadata. Correct
    labels and held-out human choices remain outside its input and are used
    only to compute evaluation metrics after each action.
    """
    all_trials = parse_badham_category_trials(text)
    valid_actions = tuple(sorted({trial.correct_action for trial in all_trials}))
    trials = all_trials
    if max_trials is not None:
        if max_trials <= 0:
            raise ValueError("max_trials must be positive.")
        trials = trials[:max_trials]
    instruction = parse_instruction_prefix(text, BADHAM_TRIAL_RE)
    env = BadhamCategoryEnv(trials, valid_actions=valid_actions, instruction=instruction)
    env.reset()
    feedback_mapping = _feedback_mapping(valid_actions)
    feedback_perturbed = any(action != mapped for action, mapped in feedback_mapping.items())
    submitted_actions: List[str] = []
    rewards: List[float] = []
    human_matches: List[bool] = []
    prompts: List[str] = []
    terminated = False
    truncated = False
    for trial_idx, trial in enumerate(trials):
        prompt = _render_badham_prompt(
            instruction,
            trials,
            submitted_actions,
            trial_idx,
            condition,
            feedback_mapping,
        )
        safe_info = {
            "experiment_id": BADHAM_ID,
            "trial_idx": trial_idx,
            "condition": condition.value,
            "valid_actions": valid_actions,
        }
        raw_action = policy(prompt, safe_info)
        submitted = normalize_action(raw_action)
        _, reward, terminated, truncated, _ = env.step(submitted)
        prompts.append(prompt)
        submitted_actions.append(submitted)
        rewards.append(reward)
        human_matches.append(submitted == trial.human_action)
        if terminated or truncated:
            break
    return BadhamBenchmarkResult(
        condition=condition,
        actions=submitted_actions,
        rewards=rewards,
        human_matches=human_matches,
        prompts=prompts,
        terminated=terminated,
        truncated=truncated,
        total_trials=len(trials),
        feedback_perturbed=feedback_perturbed and condition == ContextCondition.PERMUTED_FEEDBACK,
    )
