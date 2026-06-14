"""Generative Plonsky gambling problems with no-feedback then counterfactual blocks."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.generative.dynamics import sample_discrete_outcome
from psycenvir.generative.instructions import PLONSKY_GAMBLE_INSTRUCTION
from psycenvir.psych101.parse import parse_plonsky_option_distribution

PLONSKY_EXPERIMENT_ID = "plonsky2018when/exp1.csv"


@dataclass(frozen=True)
class _PlonskyTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    has_feedback: bool


class PlonskyGambleGenerativeEnv:
    """Fresh two-option gambles with 5 silent trials then 20 feedback trials."""

    def __init__(
        self,
        experiment_id: str = PLONSKY_EXPERIMENT_ID,
        n_problems: int = 2,
        trials_per_problem: int = 25,
        no_feedback_trials: int = 5,
        instruction: str = PLONSKY_GAMBLE_INSTRUCTION,
        include_human_ref: bool = False,
        seed: Optional[int] = None,
        schedule_trials: Optional[List[_PlonskyTrial]] = None,
        transcript_bound: bool = False,
    ) -> None:
        self._schedule_trials = list(schedule_trials) if schedule_trials is not None else None
        self.transcript_bound = transcript_bound
        if self._schedule_trials is not None and not self._schedule_trials:
            raise ValueError("schedule_trials must not be empty when provided.")
        if no_feedback_trials > trials_per_problem:
            raise ValueError("no_feedback_trials cannot exceed trials_per_problem.")
        self.experiment_id = experiment_id
        self.n_problems = n_problems
        self.trials_per_problem = trials_per_problem
        self.no_feedback_trials = no_feedback_trials
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._seed = seed
        self._rng = random.Random(seed)
        self._trials: List[_PlonskyTrial] = []
        self._trial_idx = 0
        self._points = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        if self._schedule_trials is not None:
            del seed
            self._trials = list(self._schedule_trials)
            self._trial_idx = 0
            self._points = 0.0
            self._done = False
            return (
                render_initial_observation(self.instruction, self._trials[0].observation),
                self._info(None, None),
            )
        if seed is not None:
            self._rng = random.Random(seed)
        elif self._seed is not None:
            self._rng = random.Random(self._seed)
        self._trials = []
        for _ in range(self.n_problems):
            keys = self._rng.sample(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 2)
            option_a = self._random_option_line(keys[0])
            option_b = self._random_option_line(keys[1])
            observation = option_a + "\n" + option_b
            dist_a = parse_plonsky_option_distribution(option_a)
            dist_b = parse_plonsky_option_distribution(option_b)
            for trial_number in range(self.trials_per_problem):
                outcome_a = self._draw(dist_a, self._rng)
                outcome_b = self._draw(dist_b, self._rng)
                has_feedback = trial_number >= self.no_feedback_trials
                self._trials.append(
                    _PlonskyTrial(
                        observation=observation,
                        valid_actions=(keys[0], keys[1]),
                        outcomes_by_action={keys[0]: outcome_a, keys[1]: outcome_b},
                        has_feedback=has_feedback,
                    )
                )
        self._trial_idx = 0
        self._points = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._trials[0].observation),
            self._info(None, None),
        )

    def _random_option_line(self, key: str) -> str:
        parts = []
        remaining = 1.0
        n_outcomes = self._rng.randint(1, 3)
        for index in range(n_outcomes):
            value = round(self._rng.uniform(-20.0, 100.0), 1)
            if index == n_outcomes - 1:
                probability = remaining
            else:
                probability = self._rng.uniform(0.05, remaining - 0.05 * (n_outcomes - index - 1))
                remaining -= probability
            parts.append(
                "{} points with {:.4g}% chance".format(value, probability * 100.0)
            )
        return "Option {} delivers {}.".format(key, ", ".join(parts))

    @staticmethod
    def _draw(distribution: List[Tuple[float, float]], rng: random.Random) -> float:
        values = [value for value, _ in distribution]
        probabilities = [probability for _, probability in distribution]
        return sample_discrete_outcome(rng, values, probabilities)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed PlonskyGambleGenerativeEnv; call reset().")
        submitted = normalize_action(action)
        trial = self._trials[self._trial_idx]
        if submitted not in trial.valid_actions:
            self._done = True
            info = self._info(trial, None)
            info["invalid_action"] = submitted
            return "Invalid gamble action <<{}>>.".format(submitted), 0.0, False, True, info

        reward = float(trial.outcomes_by_action[submitted])
        self._points += reward if trial.has_feedback else 0.0
        alternative = trial.valid_actions[1] if submitted == trial.valid_actions[0] else trial.valid_actions[0]
        forgone = trial.outcomes_by_action[alternative]

        if trial.has_feedback:
            if reward >= 0:
                chosen_text = "gain {} points".format(int(reward) if reward.is_integer() else reward)
            else:
                chosen_text = "lose {} points".format(int(abs(reward)) if reward.is_integer() else abs(reward))
            if forgone >= 0:
                forgone_text = "gained {} points".format(int(forgone) if forgone.is_integer() else forgone)
            else:
                forgone_text = "lost {} points".format(int(abs(forgone)) if forgone.is_integer() else abs(forgone))
            feedback = (
                "{}\nYou press <<{}>> and {}. You would have {} had you chosen option {}."
            ).format(trial.observation, submitted, chosen_text, forgone_text, alternative)
        else:
            feedback = "{}\nYou press <<{}>>.".format(trial.observation, submitted)

        self._trial_idx += 1
        self._done = self._trial_idx >= len(self._trials)
        if not self._done:
            feedback = "{}\n\n{}".format(feedback, self._trials[self._trial_idx].observation)
        return feedback, reward if trial.has_feedback else 0.0, self._done, False, self._info(trial, submitted)

    def _info(self, trial: Optional[_PlonskyTrial], selected_action: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "trial_idx": self._trial_idx,
            "points": self._points,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": (
                "generative_transcript_calibrated"
                if self.transcript_bound
                else "generative_exact"
            ),
            "episode_generative": True,
            "instruction_shown": bool(self.instruction),
            "selected_action": selected_action,
        }
        if trial is not None and self.include_human_ref:
            info["outcomes_by_action"] = dict(trial.outcomes_by_action)
            info["has_feedback"] = trial.has_feedback
        return info
