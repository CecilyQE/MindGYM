"""Paper-level consent/debrief templates derived from published methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from psycenvir.models import SessionTextBlock, SessionTextSource


def _not_in_paper(note: str) -> SessionTextBlock:
    return SessionTextBlock("", SessionTextSource.NOT_IN_PAPER, note)


def _reconstructed(text: str, note: str) -> SessionTextBlock:
    return SessionTextBlock(text, SessionTextSource.RECONSTRUCTED_FROM_PAPER, note)


@dataclass(frozen=True)
class PaperSessionTemplate:
    citation: str
    consent: SessionTextBlock
    debrief: SessionTextBlock


DEFAULT_CONSENT = _not_in_paper("论文 Methods/补充材料未提供被试看见的逐字 consent 文本。")
DEFAULT_DEBRIEF = _not_in_paper("论文未提供被试看见的逐字 debrief 文本。")


PAPER_SESSION_TEMPLATES: Dict[str, PaperSessionTemplate] = {
    "badham2017deficits": PaperSessionTemplate(
        citation="Badham, Sanborn & Maylor (2017), Psychology and Aging",
        consent=_reconstructed(
            "All participants provided written informed consent before taking part. "
            "The study received ethical approval from the University of Warwick.",
            "Methods 段概述 written informed consent 与 Warwick ethics approval；非 UI 逐字稿。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "peterson2021using": PaperSessionTemplate(
        citation="Peterson et al. (2021), Science; Thomas et al. (2024), Nature Human Behaviour (choices13k)",
        consent=_reconstructed(
            "Participants on Amazon Mechanical Turk gave informed consent under an approved "
            "institutional review board protocol before completing the choice task.",
            "Thomas et al. (2024) 方法段：AMT + IRB + informed consent；非逐字 consent 页面。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "frey2017cct": PaperSessionTemplate(
        citation="Frey et al. (2017), Science Advances",
        consent=_reconstructed(
            "Participants provided written informed consent. The study was approved by the "
            "local ethics committee. Participants received a fixed show-up fee plus "
            "performance-contingent bonuses on incentivized tasks.",
            "Frey et al. (2017) Methods：ethics approval + written consent + fixed/bonus pay；非逐字 UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "frey2017risk": PaperSessionTemplate(
        citation="Frey et al. (2017), Science Advances (balloon analogue risk task battery)",
        consent=_reconstructed(
            "Participants provided written informed consent under local ethics approval and "
            "received fixed compensation plus task-contingent bonuses.",
            "与 Frey et al. (2017) risk battery 同一研究语境；BART 逐字 consent 未公开。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "wu2018generalisation": PaperSessionTemplate(
        citation="Wu et al. (2018), Nature Human Behaviour",
        consent=_not_in_paper(
            "主文确认数据/代码可得，但未给出被试看见的逐字 consent 或完整 compensation 说明。"
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "collsiöö2023MCPL": PaperSessionTemplate(
        citation="Collsiö et al. (2023), MCPL judgment-learning paradigm",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "enkavi2019adaptivenback": PaperSessionTemplate(
        citation="Enkavi et al. (2019), adaptive n-back (Psych-101 cognitive battery)",
        consent=_reconstructed(
            "Participants completed the adaptive n-back task as part of a larger online "
            "cognitive assessment under institutional ethics oversight.",
            "Enkavi et al. (2019) 大样本在线认知任务合集；子任务逐字 consent 未单独公开。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "enkavi2019digitspan": PaperSessionTemplate(
        citation="Enkavi et al. (2019), digit span (Psych-101 cognitive battery)",
        consent=_reconstructed(
            "Participants completed the digit span task as part of a larger online cognitive "
            "assessment under institutional ethics oversight.",
            "同上：合集级 ethics，子任务逐字 consent 未单独公开。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "enkavi2019gonogo": PaperSessionTemplate(
        citation="Enkavi et al. (2019), go/no-go (Psych-101 cognitive battery)",
        consent=_reconstructed(
            "Participants completed the go/no-go task as part of a larger online cognitive "
            "assessment under institutional ethics oversight.",
            "同上：合集级 ethics，子任务逐字 consent 未单独公开。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "enkavi2019recentprobes": PaperSessionTemplate(
        citation="Enkavi et al. (2019), recent-probes working memory (Psych-101 cognitive battery)",
        consent=_reconstructed(
            "Participants completed the recent-probes task as part of a larger online cognitive "
            "assessment under institutional ethics oversight.",
            "同上：合集级 ethics，子任务逐字 consent 未单独公开。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "kool2016when": PaperSessionTemplate(
        citation="Kool et al. (2016), two-step task (when to explore)",
        consent=_reconstructed(
            "Participants provided informed consent under university ethics approval before "
            "completing the spaceship/alien reinforcement-learning task.",
            "Kool et al. (2016) 标准 university ethics + informed consent 表述；非逐字页面。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "kool2017cost": PaperSessionTemplate(
        citation="Kool et al. (2017), two-step task with effort/cost manipulation",
        consent=_reconstructed(
            "Participants provided informed consent under university ethics approval before "
            "completing the effort-cost two-step decision task.",
            "Kool et al. (2017) 标准 ethics 概述；非逐字 consent UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "gershman2018deconstructing": PaperSessionTemplate(
        citation="Gershman et al. (2018), deconstructing human randomness",
        consent=_reconstructed(
            "Participants gave informed consent under institutional review before completing "
            "the volatile bandit tasks.",
            "Gershman et al. (2018) Methods 含 ethics/consent 概述；非逐字 UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "gershman2020reward": PaperSessionTemplate(
        citation="Gershman et al. (2020), reward learning and generalization",
        consent=_reconstructed(
            "Participants provided informed consent under IRB-approved procedures before "
            "completing the stimulus-response mapping games.",
            "Gershman et al. (2020) 标准 IRB/consent 概述；非逐字 UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "wilson2014humans": PaperSessionTemplate(
        citation="Wilson et al. (2014), humans use directed exploration",
        consent=_reconstructed(
            "Participants provided informed consent under university ethics approval before "
            "the bandit exploration task.",
            "Wilson et al. (2014) Methods 含 ethics；非逐字 consent 页面。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "lefebvre2017behavioural": PaperSessionTemplate(
        citation="Lefebvre et al. (2017), behavioural diversity of human exploration",
        consent=_reconstructed(
            "Participants provided informed consent before the multi-casino bandit task.",
            "Lefebvre et al. (2017) 标准 ethics 概述；非逐字 UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "steingroever2015data": PaperSessionTemplate(
        citation="Steingroever et al. (2015), IGT data / modeling",
        consent=_reconstructed(
            "Participants completed the Iowa Gambling Task under informed consent and ethics "
            "approval as reported in the original IGT studies.",
            "Steingroever et al. 为再分析/建模论文；逐字 consent 随原始 IGT 样本，本文未复刊 UI 原文。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "wulff2018description": PaperSessionTemplate(
        citation="Wulff et al. (2018), description-experience gap (choice from description)",
        consent=_reconstructed(
            "Participants provided informed consent before lottery choice problems with "
            "described outcomes and probabilities.",
            "Wulff et al. (2018) Methods 含 ethics；非逐字 consent UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "wulff2018sampling": PaperSessionTemplate(
        citation="Wulff et al. (2018), description-experience gap (sampling)",
        consent=_reconstructed(
            "Participants provided informed consent before sampling-and-choice lottery problems.",
            "同 Wulff et al. (2018) 研究项目；非逐字 consent UI。",
        ),
        debrief=DEFAULT_DEBRIEF,
    ),
    "hilbig2014generalized": PaperSessionTemplate(
        citation="Hilbig (2014), generalized outcome-based strategy",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "bahrami2020four": PaperSessionTemplate(
        citation="Bahrami et al. (2020), four-arm bandit",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "plonsky2018when": PaperSessionTemplate(
        citation="Plonsky et al. (2018), when does risk-seeking vary",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "schulz2020finding": PaperSessionTemplate(
        citation="Schulz et al. (2020), finding structure in bandit tasks",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "tomov2020discovery": PaperSessionTemplate(
        citation="Tomov et al. (2020), discovery of structure (subway navigation)",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "tomov2021multitask": PaperSessionTemplate(
        citation="Tomov et al. (2021), multitask structure learning (castle task)",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "cox2017information": PaperSessionTemplate(
        citation="Cox et al. (2017), information and memory battery",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "flesch2018comparing": PaperSessionTemplate(
        citation="Flesch et al. (2018), comparing exploration strategies",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "garcia2023experiential": PaperSessionTemplate(
        citation="Garcia et al. (2023), experiential vs described learning",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "krueger2022identifying": PaperSessionTemplate(
        citation="Krueger et al. (2022), identifying good decisions",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "speekenbrink2008learning": PaperSessionTemplate(
        citation="Speekenbrink & Shanks (2008), learning in weather prediction task",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "feng2021dynamics": PaperSessionTemplate(
        citation="Feng et al. (2021), dynamics of exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "sadeghiyeh2020temporal": PaperSessionTemplate(
        citation="Sadeghiyeh et al. (2020), temporal dynamics of exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "somerville2017charting": PaperSessionTemplate(
        citation="Somerville et al. (2017), charting exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "waltz2020differential": PaperSessionTemplate(
        citation="Waltz et al. (2020), differential exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "ludwig2023human": PaperSessionTemplate(
        citation="Ludwig et al. (2023), human-like fruit bandit",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "ruggeri2022globalizability": PaperSessionTemplate(
        citation="Ruggeri et al. (2022), globalizability of decision preferences",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "wu2023chunking": PaperSessionTemplate(
        citation="Wu et al. (2023), chunking in sequence learning",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "xiong2023neural": PaperSessionTemplate(
        citation="Xiong et al. (2023), neural correlates / hazard bandit",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "zorowitz2023data": PaperSessionTemplate(
        citation="Zorowitz et al. (2023), data-driven two-step analysis",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "hebart2023things": PaperSessionTemplate(
        citation="Hebart et al. (2023), THINGS concept similarity",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "jansen2021dunningkruger": PaperSessionTemplate(
        citation="Jansen et al. (2021), Dunning–Kruger paradigm",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "kumar2023disentangling": PaperSessionTemplate(
        citation="Kumar et al. (2023), disentangling exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "levering2020revisiting": PaperSessionTemplate(
        citation="Levering et al. (2020), revisiting directed exploration",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "popov2023intent": PaperSessionTemplate(
        citation="Popov et al. (2023), intent inference",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "wise2019acomputational": PaperSessionTemplate(
        citation="Wise et al. (2019), computational psychiatry task",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
    "zhu2020bayesian": PaperSessionTemplate(
        citation="Zhu et al. (2020), Bayesian RL / instruction following",
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    ),
}


def paper_key(experiment_id: str) -> str:
    return experiment_id.split("/")[0]


def get_paper_template(experiment_id: str) -> PaperSessionTemplate:
    key = paper_key(experiment_id)
    if key in PAPER_SESSION_TEMPLATES:
        return PAPER_SESSION_TEMPLATES[key]
    return PaperSessionTemplate(
        citation=key,
        consent=DEFAULT_CONSENT,
        debrief=DEFAULT_DEBRIEF,
    )
