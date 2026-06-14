"""Map experiment_id to task instruction (info) text."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from psycenvir.generative import adaptive_nback, chunking, hazard_bandit, ludwig_fruit, ruggeri_choice, zorowitz_space
from psycenvir.generative.instructions import (
    BADHAM_INSTRUCTION,
    BAHRAMI_FOUR_ARM_INSTRUCTION,
    COLLSI_JUDGMENT_INSTRUCTION,
    COX_PAIR_INSTRUCTION,
    DIGIT_SPAN_INSTRUCTION,
    ENKAVI_RECENT_PROBE_INSTRUCTION,
    FLESCH_TREE_INSTRUCTION,
    FREY_CCT_INSTRUCTION,
    FREY_RISK_INSTRUCTION,
    GARCIA_EXPERIENTIAL_INSTRUCTION,
    GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION,
    GERSHMAN_DECONSTRUCT_INSTRUCTION,
    GERSHMAN_MAPPING_INSTRUCTION,
    GONOGO_INSTRUCTION,
    HILBIG_PRODUCT_INSTRUCTION,
    KOOL_COST_EXP1_INSTRUCTION,
    KOOL_COST_EXP2_INSTRUCTION,
    KOOL_WHEN_EXP1_INSTRUCTION,
    KOOL_WHEN_EXP2_INSTRUCTION,
    KRUEGER_IDENTIFYING_INSTRUCTION,
    LEFEBVRE_INSTRUCTION,
    PETERSON_INSTRUCTION,
    PLONSKY_GAMBLE_INSTRUCTION,
    SCHULZ_FINDING_INSTRUCTION,
    SPEEKENBRINK_WEATHER_INSTRUCTION,
    STEINGROEVER_IGT_EXP3_INSTRUCTION,
    STEINGROEVER_IGT_INSTRUCTION,
    TOMOV_CASTLE_INSTRUCTION,
    TOMOV_SUBWAY_INSTRUCTION,
    WILSON_BANDIT_INSTRUCTION,
    WU_BANDIT_INSTRUCTION,
    WULFF_DESCRIPTION_INSTRUCTION,
    WULFF_SAMPLING_INSTRUCTION,
)
from psycenvir.models import SessionTextSource

InfoEntry = Tuple[str, SessionTextSource, str]

_MODULE = SessionTextSource.FROM_TASK_INSTRUCTION_MODULE


def _entries() -> Dict[str, InfoEntry]:
    shared: Dict[str, InfoEntry] = {}
    for exp_id in (
        "badham2017deficits/exp1.csv",
    ):
        shared[exp_id] = (BADHAM_INSTRUCTION, _MODULE, "generative/instructions.py:BADHAM_INSTRUCTION")
    for exp_id in (
        "peterson2021using/exp1.csv",
    ):
        shared[exp_id] = (PETERSON_INSTRUCTION, _MODULE, "generative/instructions.py:PETERSON_INSTRUCTION")
    for exp_id in ("frey2017cct/exp1.csv",):
        shared[exp_id] = (FREY_CCT_INSTRUCTION, _MODULE, "generative/instructions.py:FREY_CCT_INSTRUCTION")
    for exp_id in ("frey2017risk/exp1.csv",):
        shared[exp_id] = (FREY_RISK_INSTRUCTION, _MODULE, "generative/instructions.py:FREY_RISK_INSTRUCTION")
    for exp_id in ("wu2018generalisation/exp1.csv",):
        shared[exp_id] = (WU_BANDIT_INSTRUCTION, _MODULE, "generative/instructions.py:WU_BANDIT_INSTRUCTION")
    for exp_id in (
        "collsiöö2023MCPL/exp1.csv",
        "collsiöö2023MCPL/exp2.csv",
        "collsiöö2023MCPL/exp3.csv",
    ):
        shared[exp_id] = (COLLSI_JUDGMENT_INSTRUCTION, _MODULE, "generative/instructions.py:COLLSI_JUDGMENT_INSTRUCTION")
    for exp_id in ("enkavi2019recentprobes/exp1.csv",):
        shared[exp_id] = (ENKAVI_RECENT_PROBE_INSTRUCTION, _MODULE, "generative/instructions.py:ENKAVI_RECENT_PROBE_INSTRUCTION")
    for exp_id in ("enkavi2019digitspan/exp1.csv",):
        shared[exp_id] = (DIGIT_SPAN_INSTRUCTION, _MODULE, "generative/instructions.py:DIGIT_SPAN_INSTRUCTION")
    for exp_id in ("enkavi2019gonogo/exp1.csv",):
        shared[exp_id] = (GONOGO_INSTRUCTION, _MODULE, "generative/instructions.py:GONOGO_INSTRUCTION")
    for exp_id in ("enkavi2019adaptivenback/exp1.csv",):
        shared[exp_id] = (
            adaptive_nback.ADAPTIVE_NBACK_INSTRUCTION,
            _MODULE,
            "generative/adaptive_nback.py:ADAPTIVE_NBACK_INSTRUCTION",
        )
    for exp_id in ("kool2016when/exp1.csv",):
        shared[exp_id] = (KOOL_WHEN_EXP1_INSTRUCTION, _MODULE, "generative/instructions.py:KOOL_WHEN_EXP1_INSTRUCTION")
    for exp_id in ("kool2016when/exp2.csv",):
        shared[exp_id] = (KOOL_WHEN_EXP2_INSTRUCTION, _MODULE, "generative/instructions.py:KOOL_WHEN_EXP2_INSTRUCTION")
    for exp_id in ("kool2017cost/exp1.csv",):
        shared[exp_id] = (KOOL_COST_EXP1_INSTRUCTION, _MODULE, "generative/instructions.py:KOOL_COST_EXP1_INSTRUCTION")
    for exp_id in ("kool2017cost/exp2.csv",):
        shared[exp_id] = (KOOL_COST_EXP2_INSTRUCTION, _MODULE, "generative/instructions.py:KOOL_COST_EXP2_INSTRUCTION")
    for exp_id in ("gershman2018deconstructing/exp1.csv",):
        shared[exp_id] = (GERSHMAN_DECONSTRUCT_INSTRUCTION, _MODULE, "generative/instructions.py:GERSHMAN_DECONSTRUCT_INSTRUCTION")
    for exp_id in ("gershman2018deconstructing/exp2.csv",):
        shared[exp_id] = (GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION, _MODULE, "generative/instructions.py:GERSHMAN_DECONSTRUCT_EXP2_INSTRUCTION")
    for exp_id in ("gershman2020reward/exp1.csv",):
        shared[exp_id] = (GERSHMAN_MAPPING_INSTRUCTION, _MODULE, "generative/instructions.py:GERSHMAN_MAPPING_INSTRUCTION")
    for exp_id in (
        "wilson2014humans/exp1.csv",
        "wilson2014humans/exp2.csv",
        "wilson2014humans/exp3.csv",
        "wilson2014humans/exp4.csv",
        "wilson2014humans/exp5.csv",
        "feng2021dynamics/exp1.csv",
        "sadeghiyeh2020temporal/exp1.csv",
        "somerville2017charting/exp1.csv",
        "waltz2020differential/exp1.csv",
    ):
        shared[exp_id] = (WILSON_BANDIT_INSTRUCTION, _MODULE, "generative/instructions.py:WILSON_BANDIT_INSTRUCTION (exploration family)")
    for exp_id in ("lefebvre2017behavioural/exp1.csv", "lefebvre2017behavioural/exp2.csv"):
        shared[exp_id] = (LEFEBVRE_INSTRUCTION, _MODULE, "generative/instructions.py:LEFEBVRE_INSTRUCTION")
    for exp_id in (
        "steingroever2015data/exp1.csv",
        "steingroever2015data/exp2.csv",
    ):
        shared[exp_id] = (STEINGROEVER_IGT_INSTRUCTION, _MODULE, "generative/instructions.py:STEINGROEVER_IGT_INSTRUCTION")
    for exp_id in ("steingroever2015data/exp3.csv",):
        shared[exp_id] = (STEINGROEVER_IGT_EXP3_INSTRUCTION, _MODULE, "generative/instructions.py:STEINGROEVER_IGT_EXP3_INSTRUCTION")
    for exp_id in ("wulff2018description/exp1.csv",):
        shared[exp_id] = (WULFF_DESCRIPTION_INSTRUCTION, _MODULE, "generative/instructions.py:WULFF_DESCRIPTION_INSTRUCTION")
    for exp_id in ("wulff2018sampling/exp1.csv",):
        shared[exp_id] = (WULFF_SAMPLING_INSTRUCTION, _MODULE, "generative/instructions.py:WULFF_SAMPLING_INSTRUCTION")
    for exp_id in ("hilbig2014generalized/exp1.csv",):
        shared[exp_id] = (HILBIG_PRODUCT_INSTRUCTION, _MODULE, "generative/instructions.py:HILBIG_PRODUCT_INSTRUCTION")
    for exp_id in ("bahrami2020four/exp.csv",):
        shared[exp_id] = (BAHRAMI_FOUR_ARM_INSTRUCTION, _MODULE, "generative/instructions.py:BAHRAMI_FOUR_ARM_INSTRUCTION")
    for exp_id in ("plonsky2018when/exp1.csv",):
        shared[exp_id] = (PLONSKY_GAMBLE_INSTRUCTION, _MODULE, "generative/instructions.py:PLONSKY_GAMBLE_INSTRUCTION")
    for exp_id in (
        "schulz2020finding/exp1.csv",
        "schulz2020finding/exp2.csv",
        "schulz2020finding/exp3.csv",
        "schulz2020finding/exp4.csv",
        "schulz2020finding/exp5.csv",
    ):
        shared[exp_id] = (SCHULZ_FINDING_INSTRUCTION, _MODULE, "generative/instructions.py:SCHULZ_FINDING_INSTRUCTION")
    for exp_id in (
        "tomov2020discovery/exp2.csv",
        "tomov2020discovery/exp4.csv",
        "tomov2020discovery/exp5.csv",
        "tomov2020discovery/exp7.csv",
    ):
        shared[exp_id] = (TOMOV_SUBWAY_INSTRUCTION, _MODULE, "generative/instructions.py:TOMOV_SUBWAY_INSTRUCTION")
    for exp_id in ("tomov2021multitask/exp1.csv", "tomov2021multitask/exp3.csv"):
        shared[exp_id] = (TOMOV_CASTLE_INSTRUCTION, _MODULE, "generative/instructions.py:TOMOV_CASTLE_INSTRUCTION")
    for exp_id in ("cox2017information/exp1.csv",):
        shared[exp_id] = (COX_PAIR_INSTRUCTION, _MODULE, "generative/instructions.py:COX_PAIR_INSTRUCTION")
    for exp_id in ("flesch2018comparing/exp1.csv",):
        shared[exp_id] = (FLESCH_TREE_INSTRUCTION, _MODULE, "generative/instructions.py:FLESCH_TREE_INSTRUCTION")
    for exp_id in (
        "garcia2023experiential/exp1.csv",
        "garcia2023experiential/exp2.csv",
        "garcia2023experiential/exp3.csv",
        "garcia2023experiential/exp4.csv",
    ):
        shared[exp_id] = (GARCIA_EXPERIENTIAL_INSTRUCTION, _MODULE, "generative/instructions.py:GARCIA_EXPERIENTIAL_INSTRUCTION")
    for exp_id in ("krueger2022identifying/exp1.csv",):
        shared[exp_id] = (KRUEGER_IDENTIFYING_INSTRUCTION, _MODULE, "generative/instructions.py:KRUEGER_IDENTIFYING_INSTRUCTION")
    for exp_id in ("speekenbrink2008learning/exp1.csv",):
        shared[exp_id] = (SPEEKENBRINK_WEATHER_INSTRUCTION, _MODULE, "generative/instructions.py:SPEEKENBRINK_WEATHER_INSTRUCTION")
    for exp_id in ("ludwig2023human/exp0.csv", "ludwig2023human/exp1.csv", "ludwig2023human/exp2.csv"):
        shared[exp_id] = (ludwig_fruit.LUDWIG_INSTRUCTION, _MODULE, "generative/ludwig_fruit.py:LUDWIG_INSTRUCTION")
    for exp_id in ("ruggeri2022globalizability/exp1.csv",):
        shared[exp_id] = (ruggeri_choice.RUGGERI_INSTRUCTION, _MODULE, "generative/ruggeri_choice.py:RUGGERI_INSTRUCTION")
    for exp_id in ("wu2023chunking/exp1.csv", "wu2023chunking/exp2.csv"):
        shared[exp_id] = (chunking.CHUNKING_INSTRUCTION, _MODULE, "generative/chunking.py:CHUNKING_INSTRUCTION")
    for exp_id in ("xiong2023neural/exp1.csv",):
        shared[exp_id] = (hazard_bandit.XIONG_HAZARD_INSTRUCTION, _MODULE, "generative/hazard_bandit.py:XIONG_HAZARD_INSTRUCTION")
    for exp_id in ("zorowitz2023data/exp1.csv",):
        shared[exp_id] = (zorowitz_space.ZOROWITZ_INSTRUCTION, _MODULE, "generative/zorowitz_space.py:ZOROWITZ_INSTRUCTION")
    return shared


EXPERIMENT_INFO: Dict[str, InfoEntry] = _entries()


def get_registered_info(experiment_id: str) -> Optional[InfoEntry]:
    return EXPERIMENT_INFO.get(experiment_id)
