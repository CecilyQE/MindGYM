#!/usr/bin/env python3
"""Smoke-check generative simulators and write a short coverage report."""

import json
import re
from pathlib import Path

from psycenvir import make_generative_env

LEFEBVRE_MACHINES_RE = re.compile(r"machines (\S+) and (\S+)")

SUPPORTED = [
    "badham2017deficits/exp1.csv",
    "wu2018generalisation/exp1.csv",
    "frey2017risk/exp1.csv",
    "peterson2021using/exp1.csv",
    "wilson2014humans/exp1.csv",
    "lefebvre2017behavioural/exp1.csv",
    "lefebvre2017behavioural/exp2.csv",
    "gershman2020reward/exp1.csv",
    "speekenbrink2008learning/exp1.csv",
    "frey2017cct/exp1.csv",
    "kool2016when/exp1.csv",
    "kool2016when/exp2.csv",
    "kool2017cost/exp1.csv",
    "gershman2018deconstructing/exp1.csv",
    "bahrami2020four/exp.csv",
    "hilbig2014generalized/exp1.csv",
    "wulff2018description/exp1.csv",
    "gershman2018deconstructing/exp2.csv",
    "wulff2018sampling/exp1.csv",
    "plonsky2018when/exp1.csv",
    "schulz2020finding/exp1.csv",
    "schulz2020finding/exp2.csv",
    "schulz2020finding/exp3.csv",
    "schulz2020finding/exp4.csv",
    "schulz2020finding/exp5.csv",
    "tomov2021multitask/exp1.csv",
    "tomov2021multitask/exp3.csv",
    "steingroever2015data/exp1.csv",
    "cox2017information/exp1.csv",
    "steingroever2015data/exp3.csv",
    "flesch2018comparing/exp1.csv",
    "enkavi2019digitspan/exp1.csv",
    "enkavi2019gonogo/exp1.csv",
    "kool2017cost/exp2.csv",
]


def _build_env(experiment_id: str, seed: int):
    if experiment_id.startswith("badham"):
        return make_generative_env(experiment_id, seed=seed, n_problems=2, trials_per_problem=8)
    if experiment_id.startswith("wu2018"):
        return make_generative_env(experiment_id, seed=seed, n_environments=2, choices_short=2, choices_long=2)
    if experiment_id.startswith("frey2017risk"):
        return make_generative_env(experiment_id, seed=seed, n_balloons=3, min_threshold=50, max_threshold=50)
    if experiment_id.startswith("peterson"):
        return make_generative_env(
            experiment_id,
            seed=seed,
            n_blocks=2,
            trials_per_block=2,
            outcome_pairs=[(5.0, 1.0), (0.0, 3.0)],
        )
    if experiment_id.startswith("wilson"):
        return make_generative_env(experiment_id, seed=seed, n_games=2, instructed_trials=2, free_trials_choices=(1, 1))
    if experiment_id.startswith("lefebvre"):
        return make_generative_env(experiment_id, seed=seed, n_casinos=2, visits_per_casino=3)
    if experiment_id.startswith("gershman"):
        return make_generative_env(experiment_id, seed=seed, n_games=2, trials_per_game=4)
    if experiment_id.startswith("speekenbrink"):
        return make_generative_env(experiment_id, seed=seed, n_trials=8)
    if experiment_id.startswith("frey2017cct"):
        return make_generative_env(experiment_id, seed=seed, n_rounds=3)
    if experiment_id.startswith("kool2016when/exp1"):
        return make_generative_env(experiment_id, seed=seed, n_days=5, timeout_probability=0.0)
    if experiment_id.startswith("kool2016when/exp2"):
        return make_generative_env(experiment_id, seed=seed, n_days=3)
    if experiment_id == "kool2017cost/exp2.csv":
        return make_generative_env(experiment_id, seed=seed, n_days=3)
    if experiment_id.startswith("kool2017cost"):
        return make_generative_env(experiment_id, seed=seed, n_days=5)
    if experiment_id.startswith("gershman2018"):
        return make_generative_env(experiment_id, seed=seed, n_games=2, trials_per_game=3)
    if experiment_id.startswith("bahrami"):
        return make_generative_env(experiment_id, seed=seed, n_trials=10)
    if experiment_id.startswith("hilbig"):
        return make_generative_env(experiment_id, seed=seed, n_trials=8)
    if experiment_id.startswith("wulff2018description"):
        return make_generative_env(experiment_id, seed=seed, n_problems=2)
    if experiment_id.startswith("wulff2018sampling"):
        return make_generative_env(
            experiment_id, seed=seed, n_problems=2, max_samples_before_stop=8
        )
    if experiment_id.startswith("plonsky"):
        return make_generative_env(
            experiment_id, seed=seed, n_problems=1, trials_per_problem=8, no_feedback_trials=2
        )
    if experiment_id.startswith("schulz"):
        return make_generative_env(
            experiment_id, seed=seed, n_rounds=2, trials_per_round=5, n_arms=8
        )
    if experiment_id.startswith("tomov2021multitask"):
        return make_generative_env(experiment_id, seed=seed, n_rounds=2)
    if experiment_id.startswith("steingroever"):
        return make_generative_env(experiment_id, seed=seed, n_trials=10)
    if experiment_id.startswith("cox2017"):
        return make_generative_env(experiment_id, seed=seed, n_test_trials=10)
    if experiment_id.startswith("flesch2018"):
        return make_generative_env(
            experiment_id, seed=seed, n_training_trials=6, n_test_trials=4
        )
    if experiment_id.startswith("enkavi2019digitspan"):
        return make_generative_env(experiment_id, seed=seed, n_spans=3, min_length=2, max_length=4)
    if experiment_id.startswith("enkavi2019gonogo"):
        return make_generative_env(
            experiment_id, seed=seed, n_practice_trials=4, n_test_trials=6
        )
    raise KeyError(experiment_id)


def _pick_action(experiment_id: str, steps: int, observation: str = "") -> str:
    if experiment_id.startswith("badham"):
        return "<<A>>" if steps % 2 == 0 else "<<B>>"
    if experiment_id.startswith("wu2018"):
        return str(1 + (steps % 5))
    if experiment_id.startswith("frey2017risk"):
        return "W" if steps % 3 == 2 else "H"
    if experiment_id.startswith("peterson"):
        return "Z"
    if experiment_id.startswith("wilson"):
        if "instructed to press" in observation:
            match = re.search(r"instructed to press ([A-Z])", observation)
            if match:
                return match.group(1)
        return "C" if steps % 2 == 0 else "A"
    if experiment_id.startswith("lefebvre"):
        match = LEFEBVRE_MACHINES_RE.search(observation)
        if match:
            return match.group(1)
        return "A"
    if experiment_id.startswith("gershman"):
        return "S"
    if experiment_id.startswith("speekenbrink"):
        return "E" if steps % 2 == 0 else "J"
    if experiment_id.startswith("frey2017cct"):
        return "C" if steps % 4 == 3 else "E"
    if experiment_id == "kool2017cost/exp2.csv":
        if "alien" in observation.lower() or "see alien" in observation.lower():
            return "Q"
        return "V"
    if experiment_id.startswith("kool2016when/exp1") or experiment_id.startswith("kool2017cost"):
        return "P" if steps % 2 == 0 else "F"
    if experiment_id.startswith("kool2016when/exp2"):
        if "alien" in observation.lower() or "see alien" in observation.lower():
            return "W"
        return "R"
    if experiment_id.startswith("gershman2018"):
        return "U" if steps % 2 == 0 else "P"
    if experiment_id.startswith("bahrami"):
        return "L"
    if experiment_id.startswith("hilbig"):
        return "A"
    if experiment_id.startswith("wulff2018description"):
        return "W"
    if experiment_id.startswith("wulff2018sampling"):
        if "stop sampling" in observation.lower() or steps % 5 == 4:
            return "X" if steps % 10 != 9 else "K"
        return "K" if steps % 2 == 0 else "D"
    if experiment_id.startswith("plonsky"):
        return "F"
    if experiment_id.startswith("schulz"):
        return str(1 + (steps % 8))
    if experiment_id.startswith("tomov2021multitask"):
        return ("I", "P", "G", "V", "F", "Z")[steps % 6]
    if experiment_id == "steingroever2015data/exp3.csv":
        return ("U", "F", "I", "S")[steps % 4]
    if experiment_id.startswith("steingroever"):
        return ("H", "V", "J", "D")[steps % 4]
    if experiment_id.startswith("flesch2018"):
        return "T" if steps % 2 == 0 else "N"
    if experiment_id.startswith("enkavi2019digitspan"):
        return "S" if steps % 5 == 4 else str(steps % 10)
    if experiment_id.startswith("enkavi2019gonogo"):
        from psycenvir.models import GONOGO_NO_PRESS

        return GONOGO_NO_PRESS if "colour2" in observation else "X"
    if experiment_id.startswith("cox2017"):
        return "D" if steps % 2 == 0 else "N"
    return "A"


def _run_episode(experiment_id: str, seed: int) -> dict:
    env = _build_env(experiment_id, seed)
    observation, info = env.reset(seed=seed)
    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    while steps < 500:
        action = _pick_action(experiment_id, steps, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "fidelity_level": info.get("fidelity_level"),
    }


def main() -> None:
    results = [_run_episode(experiment_id, seed=index) for index, experiment_id in enumerate(SUPPORTED)]
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "generative_smoke.json"
    md_path = output_dir / "generative_smoke.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Generative Simulator Smoke Report", ""]
    for row in results:
        lines.append(
            "- `{}`: {} steps, total_reward={:.2f}, fidelity={}".format(
                row["experiment_id"], row["steps"], row["total_reward"], row["fidelity_level"]
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {} and {}".format(json_path, md_path))


if __name__ == "__main__":
    main()
