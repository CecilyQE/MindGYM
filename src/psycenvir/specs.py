"""Load declarative experiment specifications."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from psycenvir.models import ExperimentSpec, FidelityLevel, RewardMode
from psycenvir.session_texts.resolver import get_session_texts


DEFAULT_SPEC_DIR = Path(__file__).resolve().parent / "experiment_specs"


def load_spec(path: Path, *, attach_session_texts: bool = True) -> ExperimentSpec:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    experiment_id = raw["experiment_id"]
    session_texts = get_session_texts(experiment_id) if attach_session_texts else None
    return ExperimentSpec(
        experiment_id=experiment_id,
        family=raw["family"],
        implementation_status=raw["implementation_status"],
        fidelity_level=FidelityLevel(raw["fidelity_level"]),
        reward_mode=RewardMode(raw["reward_mode"]),
        phases=list(raw["phases"]),
        action_parser=raw["action_parser"],
        payoff_rule=dict(raw.get("payoff_rule", {})),
        sources=dict(raw.get("sources", {})),
        ethics_recorded=raw.get("ethics_recorded"),
        payoff_rule_recorded=raw.get("payoff_rule_recorded", "unknown"),
        verbatim_text_available=raw.get("verbatim_text_available"),
        notes=raw.get("notes", ""),
        session_texts=session_texts,
    )


def load_specs(spec_dir: Optional[Path] = None) -> Dict[str, ExperimentSpec]:
    directory = spec_dir or DEFAULT_SPEC_DIR
    specs = [load_spec(path) for path in sorted(directory.glob("*.yaml"))]
    return {spec.experiment_id: spec for spec in specs}
