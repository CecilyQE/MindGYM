"""Factory functions for replay and selected causal task environments."""

from typing import Any, Iterable, Optional

from psycenvir.core.replay import ReplayEnv
from psycenvir.errors import EnvironmentNotReadyError
from psycenvir.psych101.parse import (
    BADHAM_TRIAL_RE,
    COX_PAIR_TEST_TRIAL_RE,
    DIGIT_SPAN_BLOCK_RE,
    FLESCH_TREE_TRIAL_RE,
    KOOL_COST_EXP1_DAY_RE,
    KOOL_COST_EXP2_DAY_RE,
    GONOGO_TRIAL_RE,
    COLLSI_TRIAL_RE,
    ENKAVI_RECENT_PROBE_TRIAL_RE,
    FREY_CCT_ROUND_RE,
    GERSHMAN_BANDIT_TRIAL_RE,
    GERSHMAN_TRIAL_RE,
    HILBIG_TRIAL_RE,
    LEFEBVRE_TRIAL_RE,
    PETERSON_BLOCK_RE,
    PLONSKY_OPTION_BLOCK_RE,
    SCHULZ_TRIAL_RE,
    TOMOV_CASTLE_STEP_RE,
    TOMOV_STATION_STEP_RE,
    STEINGROEVER_IGT_TRIAL_RE,
    SPEEKENBRINK_TRIAL_RE,
    WILSON_GAME_HEADER_RE,
    WULFF_LOTTERY_PRESS_RE,
    BAHRAMI_TRIAL_RE,
    parse_badham_category_trials,
    parse_bahrami_four_arm_trials,
    parse_collsi_judgment_trials,
    parse_cox_pair_recognition_trials,
    parse_digit_span_recall_trials,
    parse_flesch_tree_trials,
    parse_gonogo_trials,
    parse_kool_cost_exp1_days,
    parse_kool_cost_exp2_days,
    parse_collsi_response_actions,
    parse_frey_cct_rounds,
    parse_gershman_bandit_trials,
    parse_gershman_mapping_trials,
    parse_gershman_response_actions,
    parse_hilbig_product_trials,
    parse_instruction_prefix,
    parse_lefebvre_casino_trials,
    parse_plonsky_gamble_trials,
    parse_wilson_slot_trials,
    parse_peterson_gamble_blocks,
    parse_recent_probe_actions,
    parse_recent_probe_trials,
    parse_schulz_finding_trials,
    parse_steingroever_igt_trials,
    parse_speekenbrink_weather_actions,
    parse_speekenbrink_weather_trials,
    parse_tomov_castle_trials,
    parse_tomov_subway_trials,
    parse_transcript,
    parse_wulff_description_trials,
    parse_wulff_sampling_problems,
)
from psycenvir.sim.bahrami import BahramiFourArmRecordedEnv
from psycenvir.sim.casino import LefebvreCasinoRecordedEnv
from psycenvir.sim.cox_pair import CoxPairRecognitionRecordedEnv
from psycenvir.sim.digit_span import EnkaviDigitSpanRecordedEnv
from psycenvir.sim.flesch_tree import FleschTreeRecordedEnv
from psycenvir.sim.gonogo import EnkaviGonogoRecordedEnv
from psycenvir.sim.kool_cost_exp1 import KoolCostExp1RecordedEnv
from psycenvir.sim.kool_cost_exp2 import KoolCostExp2RecordedEnv
from psycenvir.sim.competitive_bandit import GershmanCompetitiveBanditRecordedEnv
from psycenvir.sim.category import BadhamCategoryEnv
from psycenvir.sim.cct import FreyCCTRecordedPathEnv
from psycenvir.sim.gamble import PetersonRecordedFeedbackEnv
from psycenvir.sim.hilbig import HilbigProductRecordedEnv
from psycenvir.sim.judgment import CollsioJudgmentEnv
from psycenvir.sim.mapping import GershmanMappingEnv
from psycenvir.sim.memory import EnkaviRecentProbeEnv
from psycenvir.sim.plonsky import PlonskyGambleRecordedEnv
from psycenvir.sim.schulz_finding import SchulzFindingRecordedEnv
from psycenvir.sim.steingroever_igt import SteingroeverIGTRecordedEnv
from psycenvir.sim.tomov_castle import TomovCastleRecordedEnv
from psycenvir.sim.tomov_subway import TomovSubwayRecordedEnv
from psycenvir.sim.two_arm_recorded import WilsonSlotRecordedEnv
from psycenvir.sim.weather import SpeekenbrinkWeatherEnv
from psycenvir.sim.wulff_lottery import WulffDescriptionRecordedEnv
from psycenvir.sim.wulff_sampling import WulffSamplingRecordedEnv
from psycenvir.specs import load_specs


BADHAM_ID = "badham2017deficits/exp1.csv"
PETERSON_ID = "peterson2021using/exp1.csv"
FREY_CCT_ID = "frey2017cct/exp1.csv"
COLLSI_EXP3_ID = "collsiöö2023MCPL/exp3.csv"
ENKAVI_RECENT_PROBES_ID = "enkavi2019recentprobes/exp1.csv"
GERSHMAN_MAPPING_ID = "gershman2020reward/exp1.csv"
SPEEKENBRINK_WEATHER_ID = "speekenbrink2008learning/exp1.csv"
LEFEBVRE_EXP1_ID = "lefebvre2017behavioural/exp1.csv"
LEFEBVRE_EXP2_ID = "lefebvre2017behavioural/exp2.csv"
WILSON_EXP1_ID = "wilson2014humans/exp1.csv"
BAHRAMI_ID = "bahrami2020four/exp.csv"
HILBIG_ID = "hilbig2014generalized/exp1.csv"
WULFF_DESCRIPTION_ID = "wulff2018description/exp1.csv"
WULFF_SAMPLING_ID = "wulff2018sampling/exp1.csv"
PLONSKY_ID = "plonsky2018when/exp1.csv"
GERSHMAN_DECONSTRUCT_EXP2_ID = "gershman2018deconstructing/exp2.csv"
SCHULZ_FINDING_EXP1_ID = "schulz2020finding/exp1.csv"
SCHULZ_FINDING_EXPERIMENT_IDS = tuple(
    "schulz2020finding/exp{}.csv".format(index) for index in range(1, 6)
)
TOMOV_SUBWAY_EXPERIMENT_IDS = (
    "tomov2020discovery/exp2.csv",
    "tomov2020discovery/exp4.csv",
    "tomov2020discovery/exp5.csv",
    "tomov2020discovery/exp7.csv",
)
TOMOV_CASTLE_EXPERIMENT_IDS = (
    "tomov2021multitask/exp1.csv",
    "tomov2021multitask/exp3.csv",
)
STEINGROEVER_IGT_EXP1_ID = "steingroever2015data/exp1.csv"
STEINGROEVER_IGT_EXP3_ID = "steingroever2015data/exp3.csv"
COX_PAIR_EXP1_ID = "cox2017information/exp1.csv"
FLESCH_TREE_EXP1_ID = "flesch2018comparing/exp1.csv"
ENKAVI_DIGIT_SPAN_EXP1_ID = "enkavi2019digitspan/exp1.csv"
KOOL_COST_EXP1_ID = "kool2017cost/exp1.csv"
KOOL_COST_EXP2_ID = "kool2017cost/exp2.csv"
ENKAVI_GONOGO_EXP1_ID = "enkavi2019gonogo/exp1.csv"


def make_replay_env(
    text: str, experiment_id: Optional[str] = None, include_human_ref: bool = False
) -> ReplayEnv:
    return ReplayEnv(
        parse_transcript(text, experiment_id=experiment_id),
        include_human_ref=include_human_ref,
    )


def make_task_env(
    experiment_id: str,
    text: str,
    include_human_ref: bool = False,
    valid_actions: Optional[Iterable[str]] = None,
) -> Any:
    specs = load_specs()
    if experiment_id not in specs:
        raise KeyError("No experiment specification registered for {!r}.".format(experiment_id))
    spec = specs[experiment_id]
    if experiment_id == BADHAM_ID:
        return BadhamCategoryEnv(
            parse_badham_category_trials(text),
            instruction=parse_instruction_prefix(text, BADHAM_TRIAL_RE),
            include_human_ref=include_human_ref,
            valid_actions=valid_actions,
        )
    if experiment_id == PETERSON_ID:
        if valid_actions is not None:
            raise ValueError(
                "Peterson valid actions are recovered per transcript block and cannot be overridden."
            )
        return PetersonRecordedFeedbackEnv(
            parse_peterson_gamble_blocks(text),
            instruction=parse_instruction_prefix(text, PETERSON_BLOCK_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == FREY_CCT_ID:
        if valid_actions is not None:
            raise ValueError(
                "Frey CCT valid actions are recovered from transcript instructions."
            )
        return FreyCCTRecordedPathEnv(
            parse_frey_cct_rounds(text),
            instruction=parse_instruction_prefix(text, FREY_CCT_ROUND_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == COLLSI_EXP3_ID:
        if valid_actions is not None:
            raise ValueError(
                "Collsio valid actions are recovered from transcript instructions."
            )
        return CollsioJudgmentEnv(
            parse_collsi_judgment_trials(text),
            valid_actions=parse_collsi_response_actions(text),
            instruction=parse_instruction_prefix(text, COLLSI_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == ENKAVI_RECENT_PROBES_ID:
        if valid_actions is not None:
            raise ValueError(
                "Recent-probes valid actions are recovered from transcript instructions."
            )
        return EnkaviRecentProbeEnv(
            parse_recent_probe_trials(text),
            valid_actions=parse_recent_probe_actions(text),
            instruction=parse_instruction_prefix(text, ENKAVI_RECENT_PROBE_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == GERSHMAN_MAPPING_ID:
        if valid_actions is not None:
            raise ValueError(
                "Gershman valid actions are recovered from transcript instructions."
            )
        return GershmanMappingEnv(
            parse_gershman_mapping_trials(text),
            valid_actions=parse_gershman_response_actions(text),
            instruction=parse_instruction_prefix(text, GERSHMAN_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == SPEEKENBRINK_WEATHER_ID:
        if valid_actions is not None:
            raise ValueError(
                "Speekenbrink valid actions are fixed to rainy/fine forecast keys."
            )
        return SpeekenbrinkWeatherEnv(
            parse_speekenbrink_weather_trials(text),
            valid_actions=parse_speekenbrink_weather_actions(text),
            instruction=parse_instruction_prefix(text, SPEEKENBRINK_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id in {LEFEBVRE_EXP1_ID, LEFEBVRE_EXP2_ID}:
        if valid_actions is not None:
            raise ValueError(
                "Lefebvre valid actions are recovered per casino visit from the transcript."
            )
        return LefebvreCasinoRecordedEnv(
            parse_lefebvre_casino_trials(text),
            instruction=parse_instruction_prefix(text, LEFEBVRE_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == WILSON_EXP1_ID:
        if valid_actions is not None:
            raise ValueError(
                "Wilson valid actions are recovered per trial from the transcript schedule."
            )
        return WilsonSlotRecordedEnv(
            parse_wilson_slot_trials(text),
            instruction=parse_instruction_prefix(text, WILSON_GAME_HEADER_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == BAHRAMI_ID:
        if valid_actions is not None:
            raise ValueError("Bahrami valid actions are recovered from the transcript.")
        return BahramiFourArmRecordedEnv(
            parse_bahrami_four_arm_trials(text),
            instruction=parse_instruction_prefix(text, BAHRAMI_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == HILBIG_ID:
        if valid_actions is not None:
            raise ValueError("Hilbig valid actions are fixed to A and R.")
        return HilbigProductRecordedEnv(
            parse_hilbig_product_trials(text),
            instruction=parse_instruction_prefix(text, HILBIG_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == WULFF_DESCRIPTION_ID:
        if valid_actions is not None:
            raise ValueError("Wulff description valid actions are recovered per problem.")
        return WulffDescriptionRecordedEnv(
            parse_wulff_description_trials(text),
            instruction=parse_instruction_prefix(text, WULFF_LOTTERY_PRESS_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == PLONSKY_ID:
        if valid_actions is not None:
            raise ValueError("Plonsky valid actions are recovered per problem block.")
        return PlonskyGambleRecordedEnv(
            parse_plonsky_gamble_trials(text),
            instruction=parse_instruction_prefix(text, PLONSKY_OPTION_BLOCK_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == WULFF_SAMPLING_ID:
        if valid_actions is not None:
            raise ValueError("Wulff sampling valid actions are fixed to K, D, and X.")
        return WulffSamplingRecordedEnv(
            parse_wulff_sampling_problems(text),
            include_human_ref=include_human_ref,
        )
    if experiment_id == GERSHMAN_DECONSTRUCT_EXP2_ID:
        if valid_actions is not None:
            raise ValueError(
                "Gershman deconstructing valid actions are recovered per game from the transcript."
            )
        return GershmanCompetitiveBanditRecordedEnv(
            parse_gershman_bandit_trials(text),
            include_human_ref=include_human_ref,
        )
    if experiment_id in SCHULZ_FINDING_EXPERIMENT_IDS:
        if valid_actions is not None:
            raise ValueError("Schulz finding valid actions are recovered per round from the transcript.")
        return SchulzFindingRecordedEnv(
            experiment_id,
            parse_schulz_finding_trials(text),
            instruction=parse_instruction_prefix(text, SCHULZ_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id in TOMOV_SUBWAY_EXPERIMENT_IDS:
        if valid_actions is not None:
            raise ValueError("Tomov subway valid actions are recovered per station from the transcript.")
        return TomovSubwayRecordedEnv(
            experiment_id,
            parse_tomov_subway_trials(text),
            instruction=parse_instruction_prefix(text, TOMOV_STATION_STEP_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id in {STEINGROEVER_IGT_EXP1_ID, STEINGROEVER_IGT_EXP3_ID}:
        if valid_actions is not None:
            raise ValueError("Steingroever IGT valid actions are recovered from the transcript.")
        return SteingroeverIGTRecordedEnv(
            experiment_id,
            parse_steingroever_igt_trials(text),
            instruction=parse_instruction_prefix(text, STEINGROEVER_IGT_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == FLESCH_TREE_EXP1_ID:
        if valid_actions is not None:
            raise ValueError("Flesch tree valid actions are recovered from transcript instructions.")
        return FleschTreeRecordedEnv(
            experiment_id,
            parse_flesch_tree_trials(text),
            instruction=parse_instruction_prefix(text, FLESCH_TREE_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == ENKAVI_DIGIT_SPAN_EXP1_ID:
        if valid_actions is not None:
            raise ValueError("Digit-span valid actions are recovered from transcript instructions.")
        return EnkaviDigitSpanRecordedEnv(
            parse_digit_span_recall_trials(text),
            instruction=parse_instruction_prefix(text, DIGIT_SPAN_BLOCK_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == KOOL_COST_EXP1_ID:
        if valid_actions is not None:
            raise ValueError("Kool cost exp1 valid actions are recovered from the transcript.")
        return KoolCostExp1RecordedEnv(
            experiment_id,
            parse_kool_cost_exp1_days(text),
            instruction=parse_instruction_prefix(text, KOOL_COST_EXP1_DAY_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == ENKAVI_GONOGO_EXP1_ID:
        if valid_actions is not None:
            raise ValueError("Go/no-go valid actions are recovered from transcript instructions.")
        return EnkaviGonogoRecordedEnv(
            parse_gonogo_trials(text),
            instruction=parse_instruction_prefix(text, GONOGO_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == KOOL_COST_EXP2_ID:
        if valid_actions is not None:
            raise ValueError("Kool cost exp2 valid actions are recovered from the transcript.")
        return KoolCostExp2RecordedEnv(
            experiment_id,
            parse_kool_cost_exp2_days(text),
            instruction=parse_instruction_prefix(text, KOOL_COST_EXP2_DAY_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id == COX_PAIR_EXP1_ID:
        if valid_actions is not None:
            raise ValueError("Cox pair recognition valid actions are fixed to D and N.")
        return CoxPairRecognitionRecordedEnv(
            experiment_id,
            parse_cox_pair_recognition_trials(text),
            instruction=parse_instruction_prefix(text, COX_PAIR_TEST_TRIAL_RE),
            include_human_ref=include_human_ref,
        )
    if experiment_id in TOMOV_CASTLE_EXPERIMENT_IDS:
        if valid_actions is not None:
            raise ValueError("Tomov castle valid actions are recovered from transcript door labels.")
        return TomovCastleRecordedEnv(
            experiment_id,
            parse_tomov_castle_trials(text),
            instruction=parse_instruction_prefix(text, TOMOV_CASTLE_STEP_RE),
            include_human_ref=include_human_ref,
        )
    raise EnvironmentNotReadyError(
        "{} is registered as {!r}; recovered causal transitions are required before simulation.".format(
            experiment_id, spec.implementation_status
        )
    )
