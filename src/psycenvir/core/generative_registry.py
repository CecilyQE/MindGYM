"""Factory for fresh-episode generative environments."""

from typing import Any, Callable, Dict, Optional, Tuple

from psycenvir.errors import EnvironmentNotReadyError
from psycenvir.generative.badham import BADHAM_EXPERIMENT_ID, BadhamGenerativeEnv
from psycenvir.generative.adaptive_nback import (
    ENKAVI_ADAPTIVE_NBACK_EXP1_ID,
    EnkaviAdaptiveNBackGenerativeEnv,
)
from psycenvir.generative.chunking import (
    WU_CHUNKING_EXP1_ID,
    WU_CHUNKING_EXP2_ID,
    WuChunkingGenerativeEnv,
)
from psycenvir.generative.cox_pair import COX_PAIR_EXP1_ID, CoxPairRecognitionGenerativeEnv
from psycenvir.generative.four_arm_bandit import BAHRAMI_EXPERIMENT_ID, BahramiFourArmGenerativeEnv
from psycenvir.generative.competitive_bandit import (
    GERSHMAN_DECONSTRUCT_EXP2_ID,
    GershmanCompetitiveBanditGenerativeEnv,
)
from psycenvir.generative.hilbig_product import HILBIG_EXPERIMENT_ID, HilbigProductGenerativeEnv
from psycenvir.generative.hazard_bandit import XIONG_NEURAL_EXP1_ID, XiongHazardBanditGenerativeEnv
from psycenvir.generative.balloon import FREY_RISK_EXPERIMENT_ID, FreyRiskBalloonGenerativeEnv
from psycenvir.generative.casino_bandit import (
    CasinoBanditGenerativeEnv,
    LEFEBVRE_EXP1_ID,
    LEFEBVRE_EXP2_ID,
)
from psycenvir.generative.cct_generative import FREY_CCT_EXPERIMENT_ID, FreyCCTGenerativeEnv
from psycenvir.generative.kool_cost import KOOL_COST_EXP1_ID, KoolCostExp1GenerativeEnv
from psycenvir.generative.kool_cost_exp2 import KOOL_COST_EXP2_ID, KoolCostExp2GenerativeEnv
from psycenvir.generative.kool_spaceship import KOOL_WHEN_EXP1_ID, KoolWhenExp1GenerativeEnv
from psycenvir.generative.kool_two_step import KOOL_WHEN_EXP2_ID, KoolWhenExp2GenerativeEnv
from psycenvir.generative.peterson import PETERSON_EXPERIMENT_ID, PetersonGenerativeEnv
from psycenvir.generative.plonsky_gamble import PLONSKY_EXPERIMENT_ID, PlonskyGambleGenerativeEnv
from psycenvir.generative.ruggeri_choice import (
    RUGGERI_GLOBALIZABILITY_EXP1_ID,
    RuggeriGlobalizabilityGenerativeEnv,
)
from psycenvir.generative.schulz_finding import (
    SCHULZ_FINDING_EXPERIMENT_IDS,
    SchulzFindingGenerativeEnv,
)
from psycenvir.generative.digit_span import ENKAVI_DIGIT_SPAN_EXP1_ID, EnkaviDigitSpanGenerativeEnv
from psycenvir.generative.gonogo import ENKAVI_GONOGO_EXP1_ID, EnkaviGonogoGenerativeEnv
from psycenvir.generative.flesch_tree import FLESCH_TREE_EXP1_ID, FleschTreeGenerativeEnv
from psycenvir.generative.steingroever_igt import (
    STEINGROEVER_IGT_EXP1_ID,
    STEINGROEVER_IGT_EXP2_ID,
    STEINGROEVER_IGT_EXP3_ID,
    DEFAULT_IGT_EXP2_DECKS,
    DEFAULT_IGT_EXP3_DECKS,
    SteingroeverIGTGenerativeEnv,
)
from psycenvir.generative.stimulus_mapping import (
    GERSHMAN_EXPERIMENT_ID,
    GershmanMappingGenerativeEnv,
)
from psycenvir.generative.instructions import STEINGROEVER_IGT_EXP3_INSTRUCTION
from psycenvir.generative.two_arm_slot import (
    WILSON_EXPERIMENT_CONFIGS,
    WILSON_EXPERIMENT_ID,
    TwoArmSlotGenerativeEnv,
)
from psycenvir.generative.recent_probes import (
    ENKAVI_RECENT_PROBES_EXP1_ID,
    EnkaviRecentProbesGenerativeEnv,
)
from psycenvir.generative.collsi_judgment import (
    COLLSI_EXP1_ID,
    COLLSI_EXP3_ID,
    CollsiJudgmentGenerativeEnv,
)
from psycenvir.generative.garcia_experiential import (
    GARCIA_EXPERIENTIAL_IDS,
    GarciaExperientialGenerativeEnv,
)
from psycenvir.generative.krueger_identifying import (
    KRUEGER_IDENTIFYING_EXP1_ID,
    KruegerIdentifyingGenerativeEnv,
)
from psycenvir.generative.ludwig_fruit import (
    LUDWIG_HUMAN_EXPERIMENT_IDS,
    LudwigFruitMarketGenerativeEnv,
)
from psycenvir.generative.volatile_bandit import (
    GERSHMAN_DECONSTRUCT_EXP1_ID,
    GershmanVolatileBanditGenerativeEnv,
)
from psycenvir.generative.weather_cards import (
    SPEEKENBRINK_EXPERIMENT_ID,
    SpeekenbrinkWeatherGenerativeEnv,
)
from psycenvir.generative.wulff_lottery import WULFF_DESCRIPTION_EXPERIMENT_ID, WulffDescriptionGenerativeEnv
from psycenvir.generative.wulff_sampling import WULFF_SAMPLING_EXPERIMENT_ID, WulffSamplingGenerativeEnv
from psycenvir.generative.wu_bandit import WU_EXPERIMENT_ID, WuSpatialBanditGenerativeEnv
from psycenvir.generative.zorowitz_space import (
    ZOROWITZ_DATA_EXP1_ID,
    ZorowitzSpaceTreasureGenerativeEnv,
)
from psycenvir.generative.grounding import (
    GroundedGenerativeEnv,
    apply_generative_defaults,
    get_grounding_profile,
)
from psycenvir.specs import load_specs

_GENERATIVE_BUILDERS: Dict[str, Callable[..., Any]] = {
    BADHAM_EXPERIMENT_ID: BadhamGenerativeEnv,
    WU_EXPERIMENT_ID: WuSpatialBanditGenerativeEnv,
    FREY_RISK_EXPERIMENT_ID: FreyRiskBalloonGenerativeEnv,
    PETERSON_EXPERIMENT_ID: PetersonGenerativeEnv,
    WILSON_EXPERIMENT_ID: lambda **kwargs: TwoArmSlotGenerativeEnv(
        experiment_id=WILSON_EXPERIMENT_ID, **kwargs
    ),
    LEFEBVRE_EXP1_ID: lambda **kwargs: CasinoBanditGenerativeEnv(
        experiment_id=LEFEBVRE_EXP1_ID, **kwargs
    ),
    LEFEBVRE_EXP2_ID: lambda **kwargs: CasinoBanditGenerativeEnv(
        experiment_id=LEFEBVRE_EXP2_ID, **kwargs
    ),
    SPEEKENBRINK_EXPERIMENT_ID: SpeekenbrinkWeatherGenerativeEnv,
    FREY_CCT_EXPERIMENT_ID: FreyCCTGenerativeEnv,
    KOOL_WHEN_EXP1_ID: KoolWhenExp1GenerativeEnv,
    KOOL_WHEN_EXP2_ID: KoolWhenExp2GenerativeEnv,
    KOOL_COST_EXP1_ID: KoolCostExp1GenerativeEnv,
    KOOL_COST_EXP2_ID: KoolCostExp2GenerativeEnv,
    GERSHMAN_DECONSTRUCT_EXP1_ID: GershmanVolatileBanditGenerativeEnv,
    BAHRAMI_EXPERIMENT_ID: BahramiFourArmGenerativeEnv,
    HILBIG_EXPERIMENT_ID: HilbigProductGenerativeEnv,
    WULFF_DESCRIPTION_EXPERIMENT_ID: WulffDescriptionGenerativeEnv,
    GERSHMAN_DECONSTRUCT_EXP2_ID: GershmanCompetitiveBanditGenerativeEnv,
    GERSHMAN_EXPERIMENT_ID: GershmanMappingGenerativeEnv,
    WULFF_SAMPLING_EXPERIMENT_ID: WulffSamplingGenerativeEnv,
    PLONSKY_EXPERIMENT_ID: PlonskyGambleGenerativeEnv,
    STEINGROEVER_IGT_EXP1_ID: SteingroeverIGTGenerativeEnv,
    COX_PAIR_EXP1_ID: CoxPairRecognitionGenerativeEnv,
    FLESCH_TREE_EXP1_ID: FleschTreeGenerativeEnv,
    ENKAVI_ADAPTIVE_NBACK_EXP1_ID: EnkaviAdaptiveNBackGenerativeEnv,
    ENKAVI_DIGIT_SPAN_EXP1_ID: EnkaviDigitSpanGenerativeEnv,
    ENKAVI_GONOGO_EXP1_ID: EnkaviGonogoGenerativeEnv,
    ENKAVI_RECENT_PROBES_EXP1_ID: EnkaviRecentProbesGenerativeEnv,
    RUGGERI_GLOBALIZABILITY_EXP1_ID: RuggeriGlobalizabilityGenerativeEnv,
    XIONG_NEURAL_EXP1_ID: XiongHazardBanditGenerativeEnv,
    COLLSI_EXP3_ID: CollsiJudgmentGenerativeEnv,
    KRUEGER_IDENTIFYING_EXP1_ID: KruegerIdentifyingGenerativeEnv,
    ZOROWITZ_DATA_EXP1_ID: ZorowitzSpaceTreasureGenerativeEnv,
}

def _steingroever_exp2_builder(**kwargs: Any) -> SteingroeverIGTGenerativeEnv:
    decks = kwargs.pop("decks", DEFAULT_IGT_EXP2_DECKS)
    if isinstance(decks, list):
        decks = tuple(decks)
    return SteingroeverIGTGenerativeEnv(
        experiment_id=STEINGROEVER_IGT_EXP2_ID, decks=decks, **kwargs
    )


_GENERATIVE_BUILDERS[STEINGROEVER_IGT_EXP2_ID] = _steingroever_exp2_builder
_GENERATIVE_BUILDERS[STEINGROEVER_IGT_EXP3_ID] = lambda **kwargs: SteingroeverIGTGenerativeEnv(
    experiment_id=STEINGROEVER_IGT_EXP3_ID,
    decks=DEFAULT_IGT_EXP3_DECKS,
    instruction=STEINGROEVER_IGT_EXP3_INSTRUCTION,
    **kwargs,
)

_GENERATIVE_BUILDERS[COLLSI_EXP1_ID] = lambda **kwargs: CollsiJudgmentGenerativeEnv(
    experiment_id=COLLSI_EXP1_ID, **kwargs
)
_GENERATIVE_BUILDERS[WU_CHUNKING_EXP1_ID] = lambda **kwargs: WuChunkingGenerativeEnv(
    experiment_id=WU_CHUNKING_EXP1_ID, **kwargs
)
_GENERATIVE_BUILDERS[WU_CHUNKING_EXP2_ID] = lambda **kwargs: WuChunkingGenerativeEnv(
    experiment_id=WU_CHUNKING_EXP2_ID, **kwargs
)

for _wilson_id in WILSON_EXPERIMENT_CONFIGS:
    if _wilson_id == WILSON_EXPERIMENT_ID:
        continue
    _GENERATIVE_BUILDERS[_wilson_id] = (
        lambda experiment_id=_wilson_id, **kwargs: TwoArmSlotGenerativeEnv(
            experiment_id=experiment_id, **kwargs
        )
    )

for _garcia_id in GARCIA_EXPERIENTIAL_IDS:
    _GENERATIVE_BUILDERS[_garcia_id] = (
        lambda experiment_id=_garcia_id, **kwargs: GarciaExperientialGenerativeEnv(
            experiment_id=experiment_id, **kwargs
        )
    )

for _ludwig_id in LUDWIG_HUMAN_EXPERIMENT_IDS:
    _GENERATIVE_BUILDERS[_ludwig_id] = (
        lambda experiment_id=_ludwig_id, **kwargs: LudwigFruitMarketGenerativeEnv(
            experiment_id=experiment_id, **kwargs
        )
    )

def _schulz_generative_builder(experiment_id: str) -> Callable[..., Any]:
    def builder(**kwargs: Any) -> SchulzFindingGenerativeEnv:
        return SchulzFindingGenerativeEnv(experiment_id=experiment_id, **kwargs)

    return builder


for _schulz_experiment_id in SCHULZ_FINDING_EXPERIMENT_IDS:
    _GENERATIVE_BUILDERS[_schulz_experiment_id] = _schulz_generative_builder(_schulz_experiment_id)


def list_generative_experiments() -> Tuple[str, ...]:
    """Experiment ids with a registered generative simulator."""
    return tuple(sorted(_GENERATIVE_BUILDERS.keys()))


def make_generative_env(
    experiment_id: str,
    seed: Optional[int] = None,
    include_human_ref: bool = False,
    **config: Any,
) -> Any:
    """Create a fresh-episode simulator for a supported experiment.

    Unlike ``make_task_env``, no participant transcript is required. Optional
    ``config`` kwargs are forwarded to the underlying environment constructor.
    """
    builder = _GENERATIVE_BUILDERS.get(experiment_id)
    if builder is None:
        specs = load_specs()
        if experiment_id not in specs:
            raise KeyError("No experiment specification registered for {!r}.".format(experiment_id))
        raise EnvironmentNotReadyError(
            "No generative simulator is registered for {!r}.".format(experiment_id)
        )
    profile = get_grounding_profile(experiment_id)
    config = apply_generative_defaults(experiment_id, config, profile)
    if include_human_ref:
        config["include_human_ref"] = True
    env = builder(seed=seed, **config)
    return GroundedGenerativeEnv(env, profile)
