"""Shared model objects for experiment specifications and parsed trials."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RewardMode(str, Enum):
    NONE = "none"
    OUTCOME = "outcome"
    CORRECTNESS = "correctness"
    NORMATIVE = "normative"


class FidelityLevel(str, Enum):
    EXACT_TRANSITION = "exact_transition"
    GENERATIVE_EXACT = "generative_exact"
    DISTRIBUTION_RECONSTRUCTED = "distribution_reconstructed"
    PARTIAL_SIM = "partial_sim"


class SessionTextSource(str, Enum):
    """Provenance of participant-facing session copy."""

    VERBATIM_FROM_PAPER = "verbatim_from_paper"
    RECONSTRUCTED_FROM_PAPER = "reconstructed_from_paper"
    FROM_TASK_INSTRUCTION_MODULE = "from_task_instruction_module"
    FROM_PSYCH101_TRANSCRIPT = "from_psych101_transcript"
    NOT_IN_PAPER = "not_in_paper"


@dataclass(frozen=True)
class SessionTextBlock:
    text: str
    source: SessionTextSource
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExperimentSessionTexts:
    experiment_id: str
    paper_citation: str
    info: SessionTextBlock
    consent: SessionTextBlock
    debrief: SessionTextBlock

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "paper_citation": self.paper_citation,
            "info": self.info.as_dict(),
            "consent": self.consent.as_dict(),
            "debrief": self.debrief.as_dict(),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    implementation_status: str
    fidelity_level: FidelityLevel
    reward_mode: RewardMode
    phases: List[str]
    action_parser: str
    payoff_rule: Dict[str, Any]
    sources: Dict[str, Any] = field(default_factory=dict)
    ethics_recorded: Optional[bool] = None
    payoff_rule_recorded: str = "unknown"
    verbatim_text_available: Optional[bool] = None
    notes: str = ""
    session_texts: Optional[ExperimentSessionTexts] = None


@dataclass(frozen=True)
class ActionEvent:
    verb: str
    human_action: str
    continuation: str
    action_segments: Tuple[str, ...] = ()
    human_actions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedTranscript:
    text: str
    initial_observation: str
    events: List[ActionEvent]
    experiment_id: Optional[str] = None


@dataclass(frozen=True)
class CategoryTrial:
    stimulus: str
    correct_action: str
    human_action: Optional[str] = None
    observation_prefix: str = ""


@dataclass(frozen=True)
class JudgmentTrial:
    progladine: str
    amalydine: str
    correct_action: str
    human_action: Optional[str] = None


@dataclass(frozen=True)
class RecentProbeTrial:
    letters: Tuple[str, ...]
    probe: str
    correct_action: str
    human_action: Optional[str] = None


@dataclass(frozen=True)
class PetersonGambleTrial:
    human_action: str
    outcomes_by_action: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class PetersonGambleBlock:
    observation: str
    valid_actions: Tuple[str, str]
    trials: List[PetersonGambleTrial]
    source_block_idx: int

    @property
    def has_feedback(self) -> bool:
        return bool(self.trials) and self.trials[0].outcomes_by_action is not None


@dataclass(frozen=True)
class FreyCCTEvent:
    human_action: str
    event_type: str
    resulting_score: Optional[int] = None


@dataclass(frozen=True)
class GershmanBanditTrial:
    game_number: int
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, int]
    human_action: Optional[str] = None
    show_game_header: bool = False


@dataclass(frozen=True)
class GershmanMappingTrial:
    stimulus_id: int
    correct_action: str
    human_action: Optional[str] = None
    game_number: int = 1
    show_game_header: bool = False


@dataclass(frozen=True)
class LefebvreCasinoTrial:
    casino_id: int
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    human_action: Optional[str] = None


@dataclass(frozen=True)
class HilbigProductTrial:
    observation: str
    valid_actions: Tuple[str, str]
    correct_action: str
    ratings_a: Tuple[int, ...]
    ratings_b: Tuple[int, ...]
    human_action: Optional[str] = None


@dataclass(frozen=True)
class BahramiFourArmTrial:
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, float]
    human_action: Optional[str] = None


@dataclass(frozen=True)
class WulffSamplingProblem:
    problem_number: int
    sample_pools: Dict[str, List[float]]
    final_outcomes_by_action: Dict[str, float]
    sampling_arms: Tuple[str, str] = ("K", "D")
    stop_action: Optional[str] = "X"
    fixed_sample_count: Optional[int] = None
    human_final_action: Optional[str] = None
    sample_sequence: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True)
class WulffLotteryTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Dict[str, float]
    human_action: Optional[str] = None


@dataclass(frozen=True)
class PlonskyGambleTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Optional[Dict[str, float]]
    human_action: Optional[str] = None
    has_feedback: bool = True


@dataclass(frozen=True)
class SteingroeverIGTTrial:
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, Tuple[float, float]]
    human_action: str
    observation: str = "Select a card from one of the four decks."


GONOGO_NO_PRESS = "NO_PRESS"


@dataclass(frozen=True)
class KoolCostExp1Day:
    """One spaceship day in Kool et al. (2017) cost-task exp1."""

    multiplier: int
    ships: Tuple[str, str]
    planet_by_ship: Dict[str, str]
    pooled_treasure_by_ship: Dict[str, float]
    human_ship: str
    human_planet: str
    human_base_treasure: int
    human_received: int


@dataclass(frozen=True)
class GonogoTrial:
    """Single go/no-go trial with optional keypress and reaction time."""

    stimulus: str
    go_key: str
    human_key: Optional[str]
    human_rt_ms: Optional[float]
    is_practice: bool = False


@dataclass(frozen=True)
class KoolCostExp2Day:
    """One two-step cost-task day: spaceship then alien."""

    multiplier: int
    ships: Tuple[str, str]
    planet_by_ship: Dict[str, str]
    aliens_by_planet: Dict[str, Tuple[str, str]]
    pooled_treasure_by_alien: Dict[str, float]
    human_ship: str
    human_alien: str
    human_planet: str
    human_base_treasure: int


@dataclass(frozen=True)
class FleschTreeTrial:
    observation: str
    valid_actions: Tuple[str, str]
    outcomes_by_action: Optional[Dict[str, float]]
    human_action: str
    has_feedback: bool = True
    garden: str = ""
    leafiness: int = 0
    branchiness: int = 0


@dataclass(frozen=True)
class DigitSpanRecallTrial:
    observation: str
    valid_actions: Tuple[str, ...]
    correct_action: str
    human_action: str
    span_index: int = 0
    span_length: int = 0


@dataclass(frozen=True)
class CoxPairRecognitionTrial:
    observation: str
    valid_actions: Tuple[str, str]
    correct_action: str
    human_action: Optional[str] = None


@dataclass(frozen=True)
class TomovCastleTrial:
    observation: str
    room_number: int
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, float]
    human_action: str
    round_number: int
    resource_amounts: Tuple[float, float, float]
    show_round_header: bool = False
    market_prices: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class TomovSubwayTrial:
    observation: str
    valid_actions: Tuple[str, ...]
    human_action: str
    round_number: int
    show_round_header: bool = False
    completes_round: bool = False


@dataclass(frozen=True)
class SchulzFindingTrial:
    round_number: int
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, float]
    human_action: Optional[str] = None
    show_round_header: bool = False


@dataclass(frozen=True)
class WilsonSlotTrial:
    observation: str
    valid_actions: Tuple[str, ...]
    outcomes_by_action: Dict[str, int]
    trial_type: str
    human_action: Optional[str] = None


@dataclass(frozen=True)
class SpeekenbrinkWeatherTrial:
    cards_display: str
    weather: str
    human_action: Optional[str] = None


@dataclass(frozen=True)
class FreyCCTRound:
    round_number: int
    gain_amount: int
    loss_amount: int
    n_loss_cards: int
    turn_action: str
    stop_action: str
    events: List[FreyCCTEvent]
    final_score: int
