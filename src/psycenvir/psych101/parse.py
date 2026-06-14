"""Parse Psych-101 natural-language transcripts into environment inputs."""

import ast
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from psycenvir.errors import TranscriptParseError
from psycenvir.models import (
    ActionEvent,
    BahramiFourArmTrial,
    CategoryTrial,
    FreyCCTEvent,
    FreyCCTRound,
    GershmanBanditTrial,
    GershmanMappingTrial,
    HilbigProductTrial,
    JudgmentTrial,
    LefebvreCasinoTrial,
    ParsedTranscript,
    PetersonGambleBlock,
    PetersonGambleTrial,
    PlonskyGambleTrial,
    RecentProbeTrial,
    CoxPairRecognitionTrial,
    DigitSpanRecallTrial,
    FleschTreeTrial,
    GonogoTrial,
    GONOGO_NO_PRESS,
    KoolCostExp1Day,
    KoolCostExp2Day,
    SchulzFindingTrial,
    SteingroeverIGTTrial,
    TomovCastleTrial,
    TomovSubwayTrial,
    SpeekenbrinkWeatherTrial,
    WilsonSlotTrial,
    WulffLotteryTrial,
    WulffSamplingProblem,
)


BRACKET_ACTION_RE = re.compile(r"<<(?P<action>[^<>]+)>>")
ACTION_PHRASE_PATTERNS = [
    (
        "predict",
        re.compile(r"You\s+predict\s+that\b.*?(?=\.\s+After\s+that,)", re.IGNORECASE),
    ),
    (
        "say",
        re.compile(r"You\s+say\s+that\b[^\n]*?<<[^<>]+>>", re.IGNORECASE),
    ),
    ("press", re.compile(r"You\s+press\s+<<[^<>]+>>", re.IGNORECASE)),
    ("say", re.compile(r"You\s+say\s+<<[^<>]+>>", re.IGNORECASE)),
    ("choose", re.compile(r"You\s+choose\s+<<[^<>]+>>", re.IGNORECASE)),
    ("estimate", re.compile(r"You\s+estimate\s+<<[^<>]+>>", re.IGNORECASE)),
    ("press", re.compile(r"\bpress\s+<<[^<>]+>>", re.IGNORECASE)),
]
BADHAM_TRIAL_RE = re.compile(
    r"You\s+see\s+(?P<stimulus>.*?)\.\s*"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.\s*"
    r"The\s+correct\s+category\s+is\s+(?P<correct_action>[^.\s]+)\.",
    re.IGNORECASE | re.DOTALL,
)
COLLSI_TRIAL_RE = re.compile(
    r"Progladine:\s*(?P<progladine>[^.]+)\.\s*"
    r"Amalydine:\s*(?P<amalydine>[^.]+)\.\s*"
    r"You\s+say\s+that\s+the\s+Caldionine\s+concentration\s+is\s+"
    r"<<(?P<human_action>[^<>]+)>>\.\s*"
    r"That\s+is\s+(?:correct|incorrect)\.\s*"
    r"The\s+correct\s+concentration\s+of\s+Caldionine\s+is"
    r"(?:\s+indeed)?\s+(?P<correct_action>[^.]+)\.",
    re.IGNORECASE,
)
COLLSI_ACTION_RE = re.compile(
    r"You\s+say\s+that\s+the\s+Caldionine\s+concentration\s+is\s+<<[^<>]+>>",
    re.IGNORECASE,
)
COLLSI_VALUES_RE = re.compile(
    r"Caldionine\s+can\s+take\s+nine\s+values\s+\((?P<values>[^)]+)\)\.",
    re.IGNORECASE,
)
ENKAVI_RECENT_PROBE_ACTIONS_RE = re.compile(
    r"If\s+you\s+think\s+it\s+was,\s+you\s+have\s+to\s+press\s+(?P<present>\S+)\.\s*"
    r"If\s+you\s+think\s+it\s+was\s+not,\s+press\s+(?P<absent>\S+)\.",
    re.IGNORECASE,
)
ENKAVI_RECENT_PROBE_TRIAL_RE = re.compile(
    r"You\s+are\s+shown\s+the\s+letters\s+(?P<letters>\[[^\]]+\])\.\s*"
    r"You\s+see\s+the\s+letter\s+(?P<probe>[^.\s]+)\.\s*"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.",
    re.IGNORECASE,
)
PETERSON_BLOCK_RE = re.compile(
    r"(?P<observation>"
    r"Option\s+(?P<action_a>[^.\s]+)\s+delivers\s+[^\n]+\n"
    r"Option\s+(?P<action_b>[^.\s]+)\s+delivers\s+[^\n]+)"
    r"\n(?P<body>.*?)(?=\n\nOption\s+[^.\s]+\s+delivers\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
PETERSON_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\."
    r"(?:\s+You\s+receive\s+(?P<chosen>-?\d+(?:\.\d+)?)\s+points\s+by\s+selecting\s+this\s+option\."
    r"\s+You\s+would\s+have\s+received\s+(?P<forgone>-?\d+(?:\.\d+)?)\s+points\s+"
    r"had\s+you\s+chosen\s+the\s+other\s+option\.)?",
    re.IGNORECASE,
)
FREY_CCT_ACTIONS_RE = re.compile(
    r"Press\s+(?P<turn_action>\S+)\s+to\s+turn\s+a\s+card\s+over,\s+or\s+"
    r"(?P<stop_action>\S+)\s+to\s+stop\s+the\s+round\s+and\s+claim\s+your\s+current\s+payout\.",
    re.IGNORECASE,
)
FREY_CCT_ROUND_RE = re.compile(
    r"Round\s+(?P<round_number>\d+):\n"
    r"You\s+will\s+be\s+awarded\s+(?P<gain_amount>\d+)\s+points\s+for\s+turning\s+over\s+a\s+gain\s+card\.\n"
    r"You\s+will\s+lose\s+(?P<loss_amount>\d+)\s+points\s+for\s+turning\s+over\s+a\s+loss\s+card\.\n"
    r"There\s+are\s+(?P<n_loss_cards>\d+)\s+loss\s+cards\s+in\s+this\s+round\.\n"
    r"(?P<body>.*?)(?=\n\nRound\s+\d+:|\Z)",
    re.DOTALL,
)
FREY_CCT_EVENT_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+"
    r"(?:turn\s+over\s+a\s+(?P<card_type>gain|loss)\s+card\.\s+"
    r"Your\s+current\s+score\s+is\s+(?P<score>-?\d+)\."
    r"(?:\s+The\s+round\s+has\s+now\s+ended\s+because\s+you\s+encountered\s+a\s+loss\s+card\.)?"
    r"|(?P<claim>claim\s+your\s+payout)\.)",
    re.IGNORECASE,
)
FREY_CCT_FINAL_SCORE_RE = re.compile(
    r"Your\s+final\s+score\s+for\s+this\s+round\s+is\s+(?P<score>-?\d+)\.",
    re.IGNORECASE,
)
GERSHMAN_TRIAL_RE = re.compile(
    r"You\s+see\s+stimulus\s+(?P<stimulus_id>\d+)\.\s*"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+get\s+(?P<points>\d+)\s+points\.",
    re.IGNORECASE,
)
GERSHMAN_GAME_RE = re.compile(r"Game\s+(?P<game_number>\d+):", re.IGNORECASE)
GERSHMAN_ACTIONS_RE = re.compile(
    r"The\s+three\s+responses\s+available\s+are\s+(?P<actions>[^.]+)\.",
    re.IGNORECASE,
)
SPEEKENBRINK_TRIAL_RE = re.compile(
    r"You\s+are\s+seeing\s+the\s+following:\s+(?P<cards>[^.]+)\.\s*"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.\s*"
    r"You\s+are\s+(?P<correctness>correct|wrong),\s+the\s+weather\s+is\s+(?:indeed\s+)?(?P<weather>rainy|fine)\.",
    re.IGNORECASE,
)
SPEEKENBRINK_ACTIONS_RE = re.compile(
    r"rainy\s+weather\s+\(by\s+pressing\s+(?P<rainy>\w+)\)\s+or\s+"
    r"fine\s+weather\s+\(by\s+pressing\s+(?P<fine>\w+)\)",
    re.IGNORECASE,
)
LEFEBVRE_TRIAL_RE = re.compile(
    r"You\s+go\s+to\s+casino\s+(?P<casino>\d+)\.\s+"
    r"You\s+can\s+choose\s+between\s+machines\s+(?P<machine_a>\w+)\s+and\s+(?P<machine_b>\w+)\.\s+"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+receive\s+(?P<outcome>-?\d+(?:\.\d+)?)\s+points\.",
    re.IGNORECASE,
)
WILSON_INSTRUCTED_RE = re.compile(
    r"You\s+are\s+instructed\s+to\s+press\s+(?P<arm>\w+)\s+and\s+get\s+(?P<points>-?\d+)\s+points\.",
    re.IGNORECASE,
)
WILSON_FREE_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+get\s+(?P<points>-?\d+)\s+points\.",
    re.IGNORECASE,
)
WILSON_GAME_HEADER_RE = re.compile(
    r"Game\s+(?P<game_number>\d+)\.\s+There\s+are\s+(?P<n_trials>\d+)\s+trials\s+in\s+this\s+game\.",
    re.IGNORECASE,
)
HILBIG_TRIAL_RE = re.compile(
    r"Product\s+(?P<label_a>\w+)\s+ratings:\s+\[(?P<ratings_a>[^\]]+)\]\.\s+"
    r"Product\s+(?P<label_b>\w+)\s+ratings:\s+\[(?P<ratings_b>[^\]]+)\]\.\s+"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.",
    re.IGNORECASE,
)
BAHRAMI_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+get\s+(?P<outcome>[\d.]+)\s+points\.",
    re.IGNORECASE,
)
SCHULZ_ROUND_RE = re.compile(r"You\s+are\s+playing\s+round\s+(?P<round>\d+):", re.IGNORECASE)
SCHULZ_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>\d+)>>\s+and\s+get\s+(?P<outcome>[\d.]+)\s+points\.",
    re.IGNORECASE,
)
SCHULZ_EIGHT_ARM_HINT = "options 1, 2, 3, 4, 5, 6, 7 and 8"
DEFAULT_SCHULZ_ARMS = tuple(str(index) for index in range(1, 9))
TOMOV_DIRECTION_KEYS_RE = re.compile(
    r"go\s+north\s+by\s+pressing\s+(?P<north>\w+),\s+"
    r"west\s+by\s+pressing\s+(?P<west>\w+),\s+"
    r"south\s+by\s+pressing\s+(?P<south>\w+),\s+and\s+east\s+by\s+pressing\s+(?P<east>\w+)\.",
    re.IGNORECASE,
)
TOMOV_GOAL_KEY_RE = re.compile(
    r"When\s+you\s+reach\s+the\s+goal\s+station,\s+press\s+(?P<goal>\w+)\s+to\s+end",
    re.IGNORECASE,
)
TOMOV_ROUND_RE = re.compile(
    r"The\s+new\s+starting\s+station\s+is\s+(?P<start>\d+)\s+and\s+the\s+goal\s+station\s+is\s+(?P<goal>\d+)\.",
    re.IGNORECASE,
)
TOMOV_STATION_STEP_RE = re.compile(
    r"Your\s+station:\s+(?P<station>\d+)\.\s+"
    r"Neighboring\s+stations:\s+(?P<neighbors>.*?)\.\s+"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.",
    re.IGNORECASE,
)
TOMOV_CASTLE_DOORS_RE = re.compile(r"The\s+doors\s+are\s+labeled\s+([^.]+)\.", re.IGNORECASE)
TOMOV_CASTLE_PRICES_RE = re.compile(
    r"The\s+current\s+market\s+prices\s+are\s+(?P<wood>-?[\d.]+)\s+for\s+wood,\s+"
    r"(?P<stone>-?[\d.]+)\s+for\s+stone,\s+and\s+(?P<iron>-?[\d.]+)\s+for\s+iron\.",
    re.IGNORECASE,
)
STEINGROEVER_IGT_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<deck>[^<>]+)>>\.\s+"
    r"You\s+win\s+(?P<win>[\d.]+)\$\s+and\s+lose\s+(?P<loss>[\d.]+)\$",
    re.IGNORECASE,
)
COX_PAIR_RECOGNITION_SECTION_RE = re.compile(
    r"You\s+study\s+the\s+following\s+20\s+word\s+pairs:\n(?P<pairs>(?:[^\n]+\n){20})\s*"
    r"You\s+will\s+now\s+view\s+a\s+single\s+pair\s+of\s+words\.\n"
    r"Your\s+task\s+is\s+to\s+indicate\s+if\s+the\s+pair\s+of\s+words\s+you\s+see\s+on\s+the\s+screen\s+was\s+studied\s+as\s+a\s+pair\s+on\s+the\s+list\s+you\s+just\s+studied\s+\(by\s+pressing\s+(?P<studied_key>[^)]+)\)\s+or\s+was\s+not\s+a\s+pair\s+\(by\s+pressing\s+(?P<novel_key>[^)]+)\)\.\s*\n\n"
    r"(?P<body>.*?)(?=\nYou\s+will\s+now\s+view|\Z)",
    re.IGNORECASE | re.DOTALL,
)
COX_PAIR_TEST_TRIAL_RE = re.compile(
    r"You\s+view\s+the\s+word\s+pair\s+(?P<pair>[^.]+)\.\s+You\s+press\s+<<(?P<human_action>[^<>]+)>>\.",
    re.IGNORECASE,
)
FLESCH_TREE_TRIAL_RE = re.compile(
    r"You\s+get\s+a\s+tree\s+with\s+level\s+(?P<leaf>\d+)\s+of\s+leafiness\s+and\s+level\s+(?P<branch>\d+)\s+of\s+branchiness\s+in\s+the\s+"
    r"(?P<garden>North|South)\s+garden\.\s+You\s+press\s+<<(?P<human>\w+)>>"
    r"(?:\s+and\s+get\s+(?P<human_reward>-?\d+)\s+points\.\s+You\s+would\s+have\s+gotten\s+(?P<alt_reward>-?\d+)\s+points,\s+had\s+you\s+"
    r"(?P<alt_clause>accepted|rejected)(?:\s+to\s+plant\s+the\s+tree)?\.)?"
    r"(?:\.\s*|\s+)",
    re.IGNORECASE,
)
KOOL_COST_EXP1_DAY_RE = re.compile(
    r"There\s+is\s+(?P<multiplier>no\s+treasure\s+multiplier|a\s+treasure\s+multiplier)\.\s+"
    r"You\s+are\s+presented\s+with\s+spaceships\s+(?P<ship0>\w+)\s+and\s+(?P<ship1>\w+)\.\s+"
    r"You\s+press\s+<<(?P<ship>\w+)>>\.\s+You\s+end\s+up\s+on\s+planet\s+(?P<planet>\w+)\.\s+"
    r"You\s+find\s+(?P<base>\d+)\s+pieces\s+of\s+space\s+treasure\.\s+"
    r"You\s+receive\s+(?P<received>\d+)\s+pieces\s+of\s+space\s+treasure\.",
    re.IGNORECASE,
)
GONOGO_GO_KEY_RE = re.compile(
    r"press\s+button\s+(\w+)\s+when\s+you\s+see\s+colour1",
    re.IGNORECASE,
)
GONOGO_PHASE_COUNTS_RE = re.compile(
    r"(\d+)\s+practice\s+trials\s+followed\s+by\s+(\d+)\s+test\s+trials",
    re.IGNORECASE,
)
GONOGO_TRIAL_RE = re.compile(
    r"You\s+see\s+(?P<stimulus>colour1|colour2)\s+and\s+"
    r"(?:press\s+nothing|press\s+<<(?P<key>\w+)>>\s+in\s+(?P<rt>[\d.]+)ms)\.",
    re.IGNORECASE,
)
KOOL_COST_EXP2_DAY_RE = re.compile(
    r"There\s+is\s+(?P<multiplier>no\s+treasure\s+multiplier|a\s+treasure\s+multiplier)\.\s+"
    r"You\s+are\s+presented\s+with\s+spaceships\s+(?P<ship0>\w+)\s+and\s+(?P<ship1>\w+)\.\s+"
    r"You\s+press\s+<<(?P<ship>\w+)>>\.\s+You\s+end\s+up\s+on\s+planet\s+(?P<planet>\w+)\.\s+"
    r"You\s+see\s+alien\s+(?P<alien0>\w+)\s+and\s+alien\s+(?P<alien1>\w+)\.\s+"
    r"You\s+press\s+<<(?P<alien>\w+)>>\.\s+You\s+find\s+(?P<base>\d+)\s+pieces\s+of\s+space\s+treasure\.",
    re.IGNORECASE,
)
KOOL_COST_EXP2_HEADER_RE = re.compile(
    r"You\s+will\s+be\s+taking\s+one\s+of\s+the\s+spaceships\s+(\w+)\s+or\s+(\w+)\s+to\s+one\s+of\s+the\s+planets\s+(\w+)\s+or\s+(\w+)\.",
    re.IGNORECASE,
)
KOOL_COST_EXP2_ALIENS_RE = re.compile(
    r"Planet\s+(\w+)\s+has\s+aliens\s+(\w+)\s+and\s+(\w+),\s+and\s+planet\s+(\w+)\s+has\s+aliens\s+(\w+)\s+and\s+(\w+)",
    re.IGNORECASE,
)
FLESCH_RESPONSE_KEYS_RE = re.compile(
    r"accept\s+to\s+plant\s+the\s+tree\s+by\s+pressing\s+(\w+)\s+and\s+reject\s+to\s+plant\s+it\s+by\s+pressing\s+(\w+)",
    re.IGNORECASE,
)
DIGIT_SPAN_BLOCK_RE = re.compile(
    r"The\s+digits\s+are\s+the\s+following:\s+\[(?P<digits>[^\]]+)\]\n(?P<presses>(?:You\s+press\s+<<[^>]+>>\.\s*\n?)+)",
    re.IGNORECASE,
)
DIGIT_SPAN_END_KEY_RE = re.compile(
    r"please\s+press\s+'([^']+)'\s+to\s+indicate\s+the\s+end",
    re.IGNORECASE,
)
TOMOV_CASTLE_STEP_RE = re.compile(
    r"You\s+are\s+in\s+room\s+(?P<room>\d+)\.\s+"
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+you\s+find\s+"
    r"(?P<wood>[\d.]+)\s+wood,\s+(?P<stone>[\d.]+)\s+stone,\s+and\s+(?P<iron>[\d.]+)\s+iron\.\s+"
    r"You\s+get\s+(?P<reward>-?[\d.]+)\s+points\.",
    re.IGNORECASE,
)
GERSHMAN_BANDIT_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+get\s+(?P<points>-?\d+)\s+points\.",
    re.IGNORECASE,
)
WULFF_LOTTERY_LINE_RE = re.compile(r"^Lottery\s+(\w+)\s+offers\s+(.+)$", re.MULTILINE | re.IGNORECASE)
WULFF_LOTTERY_PRESS_RE = re.compile(
    r"((?:Lottery\s+[^\n]+\n)+)You\s+press\s+<<(?P<human_action>[^<>]+)>>\.\s*",
    re.IGNORECASE,
)
WULFF_OUTCOME_PART_RE = re.compile(
    r"(-?[\d.]+)\s+points\s+with\s+([\d.]+)%\s+probability", re.IGNORECASE
)
WULFF_SAMPLING_ARMS_RE = re.compile(
    r"(?:by\s+pressing|sample\s+from)[^\n]*?\b(?P<arm0>[A-Z])\s+or\s+(?P<arm1>[A-Z])\b",
    re.IGNORECASE,
)
WULFF_SAMPLING_STOP_RE = re.compile(
    r"stop\s+sampling\s+by\s+pressing\s+(?P<stop>[A-Z])\b",
    re.IGNORECASE,
)
WULFF_FIXED_SAMPLE_COUNT_RE = re.compile(
    r"After\s+sampling\s+(?P<count>\d+)\s+times",
    re.IGNORECASE,
)
WULFF_FIXED_FINAL_RE = re.compile(
    r"You\s+are\s+asked\s+to\s+choose\s+one\s+lottery\s+for\s+real\s+and\s+you\s+press\s+<<(?P<arm>[^<>]+)>>\.",
    re.IGNORECASE,
)
WULFF_SAMPLING_SAMPLE_RE = re.compile(
    r"You\s+press\s+<<(?P<arm>[^<>]+)>>\s+and\s+observe\s+(?P<outcome>-?[\d.]+)\s+points\.",
    re.IGNORECASE,
)
WULFF_SAMPLING_FINAL_RE = re.compile(
    r"You\s+press\s+<<(?P<stop>[^<>]+)>>\s+to\s+stop\s+sampling\s+and\s+then\s+press\s+<<(?P<arm>[^<>]+)>>\.",
    re.IGNORECASE,
)
PLONSKY_OUTCOME_PART_RE = re.compile(
    r"(-?[\d.]+)\s+points\s+with\s+([\d.]+)%\s+chance", re.IGNORECASE
)
PLONSKY_OPTION_BLOCK_RE = re.compile(
    r"(?P<option_a>Option\s+[^\n]+)\n(?P<option_b>Option\s+[^\n]+)\n(?P<body>.+?)(?=\nOption\s+|\Z)",
    re.DOTALL,
)
PLONSKY_OPTION_KEY_RE = re.compile(r"^Option\s+(\w+)\s+delivers", re.IGNORECASE)
PLONSKY_NO_FEEDBACK_TRIAL_RE = re.compile(r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\.\s*", re.IGNORECASE)
PLONSKY_FEEDBACK_TRIAL_RE = re.compile(
    r"You\s+press\s+<<(?P<human_action>[^<>]+)>>\s+and\s+(?P<verb>gain|lose)\s+(?P<chosen>[\d.]+)\s+points\.\s+"
    r"You\s+would\s+have\s+(?P<alt_verb>gained|lost)\s+(?P<forgone>[\d.]+)\s+points\s+had\s+you\s+chosen\s+option\s+(?P<alt_action>\w+)\.",
    re.IGNORECASE,
)


def parse_transcript(text: str, experiment_id: Optional[str] = None) -> ParsedTranscript:
    """Parse action boundaries while retaining transcript-native observations.

    Replay only needs to know what text followed each recorded action. The
    returned continuation deliberately does not claim to be a causal response
    to an alternative action.
    """
    phrase_matches = []
    for verb, pattern in ACTION_PHRASE_PATTERNS:
        phrase_matches.extend((match.start(), match.end(), verb, match) for match in pattern.finditer(text))
    phrase_matches.extend(
        (match.start(), match.end(), "respond", match) for match in BRACKET_ACTION_RE.finditer(text)
    )
    phrase_matches.sort(key=lambda entry: (entry[0], -(entry[1] - entry[0])))

    matches = []
    occupied_until = -1
    for start, end, verb, match in phrase_matches:
        if start < occupied_until:
            continue
        matches.append((verb, match))
        occupied_until = end
    if not matches:
        raise TranscriptParseError("Transcript does not contain a supported encoded action phrase.")

    events: List[ActionEvent] = []
    for index, (verb, match) in enumerate(matches):
        next_start = matches[index + 1][1].start() if index + 1 < len(matches) else len(text)
        phrase = match.group(0)
        action_tokens = tuple(
            token.group("action").strip() for token in BRACKET_ACTION_RE.finditer(phrase)
        )
        if not action_tokens:
            raise TranscriptParseError("Matched action phrase contains no encoded action.")
        events.append(
            ActionEvent(
                verb=verb.lower(),
                human_action=" || ".join(action_tokens),
                continuation=text[match.end() : next_start],
                action_segments=tuple(BRACKET_ACTION_RE.split(phrase)[::2]),
                human_actions=action_tokens,
            )
        )
    return ParsedTranscript(
        text=text,
        initial_observation=text[: matches[0][1].start()],
        events=events,
        experiment_id=experiment_id,
    )


def parse_instruction_prefix(text: str, first_task_pattern: re.Pattern) -> str:
    """Return transcript instructions preceding the first task-specific state."""
    match = first_task_pattern.search(text)
    if not match:
        raise TranscriptParseError("No first task state found while extracting instructions.")
    return text[: match.start()].strip()


def parse_badham_category_trials(text: str) -> List[CategoryTrial]:
    """Recover exact category-learning trials exposed in Badham transcripts."""
    matches = list(BADHAM_TRIAL_RE.finditer(text))
    trials = []
    previous_end = None
    for match in matches:
        prefix = ""
        if previous_end is not None:
            prefix = text[previous_end : match.start()].strip()
        trials.append(
            CategoryTrial(
                stimulus=match.group("stimulus").strip(),
                correct_action=match.group("correct_action").strip(),
                human_action=match.group("human_action").strip(),
                observation_prefix=prefix,
            )
        )
        previous_end = match.end()
    if not trials:
        raise TranscriptParseError(
            "No Badham category trials found; expected 'The correct category is ...' feedback."
        )
    return trials


def parse_collsi_judgment_trials(text: str) -> List[JudgmentTrial]:
    """Recover exact Caldionine feedback trials where feedback is recorded."""
    trials = [
        JudgmentTrial(
            progladine=match.group("progladine").strip(),
            amalydine=match.group("amalydine").strip(),
            correct_action=match.group("correct_action").strip(),
            human_action=match.group("human_action").strip(),
        )
        for match in COLLSI_TRIAL_RE.finditer(text)
    ]
    action_count = len(COLLSI_ACTION_RE.findall(text))
    if not trials:
        raise TranscriptParseError("No Collsio judgment feedback trials found.")
    if len(trials) != action_count:
        raise TranscriptParseError(
            "Collsio transcript includes trials without recorded correct-concentration feedback."
        )
    return trials


def parse_collsi_response_actions(text: str) -> List[str]:
    """Return all valid Caldionine response values declared in the instruction."""
    match = COLLSI_VALUES_RE.search(text)
    if not match:
        raise TranscriptParseError("No Collsio Caldionine response values found in instructions.")
    actions = [value.strip() for value in match.group("values").split(",")]
    if len(actions) != 9 or len(set(actions)) != 9:
        raise TranscriptParseError("Collsio instructions do not declare nine unique response values.")
    return actions


def parse_recent_probe_trials(text: str) -> List[RecentProbeTrial]:
    """Recover objective old/new memory-probe trials from explicit task rules."""
    action_match = ENKAVI_RECENT_PROBE_ACTIONS_RE.search(text)
    if not action_match:
        raise TranscriptParseError("No recent-probes response mapping found in instructions.")
    present_action = action_match.group("present").strip()
    absent_action = action_match.group("absent").strip()
    trials: List[RecentProbeTrial] = []
    for match in ENKAVI_RECENT_PROBE_TRIAL_RE.finditer(text):
        raw_letters = ast.literal_eval(match.group("letters"))
        if not isinstance(raw_letters, list) or not raw_letters:
            raise TranscriptParseError("Recent-probes stimulus list is malformed.")
        letters = tuple(str(letter).strip() for letter in raw_letters)
        probe = match.group("probe").strip()
        correct_action = (
            present_action
            if probe.upper() in {letter.upper() for letter in letters}
            else absent_action
        )
        trials.append(
            RecentProbeTrial(
                letters=letters,
                probe=probe,
                correct_action=correct_action,
                human_action=match.group("human_action").strip(),
            )
        )
    if not trials:
        raise TranscriptParseError("No recent-probes trials found.")
    if len(trials) != len(re.findall(r"You\s+press\s+<<[^<>]+>>", text, re.IGNORECASE)):
        raise TranscriptParseError("Unrecognized recent-probes action format.")
    return trials


def parse_recent_probe_actions(text: str) -> List[str]:
    """Return response keys declared for present and absent probes."""
    match = ENKAVI_RECENT_PROBE_ACTIONS_RE.search(text)
    if not match:
        raise TranscriptParseError("No recent-probes response mapping found in instructions.")
    actions = [match.group("present").strip(), match.group("absent").strip()]
    if len(set(actions)) != 2:
        raise TranscriptParseError("Recent-probes instructions do not declare two distinct actions.")
    return actions


def parse_peterson_gamble_blocks(text: str) -> List[PetersonGambleBlock]:
    """Parse choices13k blocks, preserving the experiment's feedback condition."""
    blocks: List[PetersonGambleBlock] = []
    for source_block_idx, match in enumerate(PETERSON_BLOCK_RE.finditer(text)):
        valid_actions = (match.group("action_a").strip(), match.group("action_b").strip())
        body = match.group("body")
        trials: List[PetersonGambleTrial] = []
        for trial_match in PETERSON_TRIAL_RE.finditer(body):
            human_action = trial_match.group("human_action").strip()
            if human_action not in valid_actions:
                raise TranscriptParseError(
                    "Peterson action {!r} is not one of the displayed options {!r}.".format(
                        human_action, valid_actions
                    )
                )
            chosen = trial_match.group("chosen")
            forgone = trial_match.group("forgone")
            outcomes = None
            if chosen is not None and forgone is not None:
                alternative = (
                    valid_actions[1] if human_action == valid_actions[0] else valid_actions[0]
                )
                outcomes = {
                    human_action: float(chosen),
                    alternative: float(forgone),
                }
            trials.append(
                PetersonGambleTrial(
                    human_action=human_action,
                    outcomes_by_action=outcomes,
                )
            )

        if len(trials) != len(re.findall(r"You\s+press\s+<<[^<>]+>>", body, re.IGNORECASE)):
            raise TranscriptParseError("Unrecognized Peterson trial feedback format.")
        if len(trials) != 5:
            raise TranscriptParseError(
                "Peterson block has {} trials; expected five.".format(len(trials))
            )
        feedback_modes = {trial.outcomes_by_action is not None for trial in trials}
        if len(feedback_modes) != 1:
            raise TranscriptParseError("Peterson block mixes feedback and no-feedback trials.")
        blocks.append(
            PetersonGambleBlock(
                observation=match.group("observation"),
                valid_actions=valid_actions,
                trials=trials,
                source_block_idx=source_block_idx,
            )
        )

    if not blocks:
        raise TranscriptParseError("No Peterson gamble blocks found.")
    return blocks


def parse_frey_cct_rounds(text: str) -> List[FreyCCTRound]:
    """Parse recorded hot-CCT paths and validate their score arithmetic."""
    action_match = FREY_CCT_ACTIONS_RE.search(text)
    if not action_match:
        raise TranscriptParseError("No Frey CCT action mapping found in instructions.")
    turn_action = action_match.group("turn_action").strip()
    stop_action = action_match.group("stop_action").strip()
    rounds: List[FreyCCTRound] = []
    for match in FREY_CCT_ROUND_RE.finditer(text):
        gain_amount = int(match.group("gain_amount"))
        loss_amount = int(match.group("loss_amount"))
        body = match.group("body")
        events: List[FreyCCTEvent] = []
        current_score = 0
        for event_match in FREY_CCT_EVENT_RE.finditer(body):
            human_action = event_match.group("human_action").strip()
            card_type = event_match.group("card_type")
            if card_type:
                event_type = card_type.lower()
                expected_action = turn_action
                current_score += gain_amount if event_type == "gain" else -loss_amount
                resulting_score = int(event_match.group("score"))
                if resulting_score != current_score:
                    raise TranscriptParseError("Frey CCT event score does not match round arithmetic.")
            else:
                event_type = "stop"
                expected_action = stop_action
                resulting_score = None
            if human_action != expected_action:
                raise TranscriptParseError("Frey CCT event uses an unexpected response key.")
            events.append(
                FreyCCTEvent(
                    human_action=human_action,
                    event_type=event_type,
                    resulting_score=resulting_score,
                )
            )
        if len(events) != len(re.findall(r"You\s+press\s+<<[^<>]+>>", body, re.IGNORECASE)) or not events:
            raise TranscriptParseError("Unrecognized Frey CCT event format.")
        final_match = FREY_CCT_FINAL_SCORE_RE.search(body)
        if not final_match or int(final_match.group("score")) != current_score:
            raise TranscriptParseError("Frey CCT final score does not match round arithmetic.")
        if events[-1].event_type not in {"loss", "stop"}:
            raise TranscriptParseError("Frey CCT round does not contain a terminal event.")
        rounds.append(
            FreyCCTRound(
                round_number=int(match.group("round_number")),
                gain_amount=gain_amount,
                loss_amount=loss_amount,
                n_loss_cards=int(match.group("n_loss_cards")),
                turn_action=turn_action,
                stop_action=stop_action,
                events=events,
                final_score=current_score,
            )
        )
    if not rounds:
        raise TranscriptParseError("No Frey CCT rounds found.")
    return rounds


def parse_gershman_response_actions(text: str) -> List[str]:
    """Return the three response keys declared in Gershman instructions."""
    match = GERSHMAN_ACTIONS_RE.search(text)
    if not match:
        raise TranscriptParseError("No Gershman response mapping found in instructions.")
    actions = [
        re.sub(r"^(?:and|or)\s+", "", value.strip(), flags=re.IGNORECASE)
        for value in match.group("actions").split(",")
    ]
    if len(actions) != 3 or len(set(actions)) != 3:
        raise TranscriptParseError("Gershman instructions do not declare three unique response keys.")
    return actions


def parse_gershman_mapping_trials(text: str) -> List[GershmanMappingTrial]:
    """Recover stimulus-response trials with correct keys inferred from feedback."""
    valid_actions = parse_gershman_response_actions(text)
    game_starts = {
        match.start(): int(match.group("game_number")) for match in GERSHMAN_GAME_RE.finditer(text)
    }
    evidence: Dict[Tuple[int, int], Dict[str, set]] = {}
    raw_trials = []
    for match in GERSHMAN_TRIAL_RE.finditer(text):
        game_number = 1
        for start, number in sorted(game_starts.items()):
            if start <= match.start():
                game_number = number
        stimulus_id = int(match.group("stimulus_id"))
        human_action = match.group("human_action").strip()
        points = int(match.group("points"))
        key = (game_number, stimulus_id)
        evidence.setdefault(key, {}).setdefault(human_action, set()).add(points)
        raw_trials.append((match, game_number, stimulus_id, human_action, points))

    correct_by_key: Dict[Tuple[int, int], str] = {}
    for key, action_points in evidence.items():
        rewarded = [action for action, points in action_points.items() if 1 in points]
        if len(rewarded) == 1:
            correct_by_key[key] = rewarded[0]
            continue
        if len(rewarded) > 1:
            raise TranscriptParseError(
                "Conflicting Gershman rewards for game {} stimulus {}.".format(key[0], key[1])
            )
        ruled_out = {
            action
            for action, points in action_points.items()
            if points == {0}
        }
        remaining = [action for action in valid_actions if action not in ruled_out]
        if len(remaining) != 1:
            raise TranscriptParseError(
                "Could not infer a unique Gershman correct response for game {} stimulus {}.".format(
                    key[0], key[1]
                )
            )
        correct_by_key[key] = remaining[0]

    trials: List[GershmanMappingTrial] = []
    previous_game = None
    for match, game_number, stimulus_id, human_action, _ in raw_trials:
        show_game_header = game_number != previous_game
        previous_game = game_number
        trials.append(
            GershmanMappingTrial(
                stimulus_id=stimulus_id,
                correct_action=correct_by_key[(game_number, stimulus_id)],
                human_action=human_action,
                game_number=game_number,
                show_game_header=show_game_header,
            )
        )
    if not trials:
        raise TranscriptParseError("No Gershman mapping trials found.")
    return trials


def parse_speekenbrink_weather_trials(text: str) -> List[SpeekenbrinkWeatherTrial]:
    """Recover weather-forecast trials with recorded latent weather."""
    trials = [
        SpeekenbrinkWeatherTrial(
            cards_display=match.group("cards").strip(),
            weather=match.group("weather").strip().lower(),
            human_action=match.group("human_action").strip(),
        )
        for match in SPEEKENBRINK_TRIAL_RE.finditer(text)
    ]
    if not trials:
        raise TranscriptParseError("No Speekenbrink weather trials found.")
    if len(trials) != len(re.findall(r"You\s+press\s+<<[^<>]+>>", text, re.IGNORECASE)):
        raise TranscriptParseError("Unrecognized Speekenbrink action format.")
    return trials


def parse_speekenbrink_weather_actions(text: str) -> Tuple[str, str]:
    """Return the rainy and fine response keys declared in the instructions."""
    match = SPEEKENBRINK_ACTIONS_RE.search(text)
    if match is None:
        return ("E", "J")
    return (match.group("rainy").strip(), match.group("fine").strip())


def parse_lefebvre_casino_trials(text: str) -> List[LefebvreCasinoTrial]:
    """Recover casino visits with pooled arm outcomes for counterfactual feedback."""
    raw = []
    for match in LEFEBVRE_TRIAL_RE.finditer(text):
        casino_id = int(match.group("casino"))
        machines = tuple(sorted((match.group("machine_a").strip(), match.group("machine_b").strip())))
        human_action = match.group("human_action").strip()
        outcome = float(match.group("outcome"))
        if human_action not in machines:
            raise TranscriptParseError(
                "Lefebvre action {!r} is not one of machines {!r}.".format(human_action, machines)
            )
        raw.append((casino_id, machines, human_action, outcome))

    if not raw:
        raise TranscriptParseError("No Lefebvre casino trials found.")

    pools: Dict[Tuple[int, Tuple[str, str]], Dict[str, List[float]]] = {}
    for casino_id, machines, human_action, outcome in raw:
        key = (casino_id, machines)
        pools.setdefault(key, {}).setdefault(human_action, []).append(outcome)

    trials: List[LefebvreCasinoTrial] = []
    for casino_id, machines, human_action, outcome in raw:
        key = (casino_id, machines)
        outcomes_by_action: Dict[str, float] = {}
        for machine in machines:
            samples = pools[key].get(machine, [])
            outcomes_by_action[machine] = sum(samples) / len(samples) if samples else outcome
        trials.append(
            LefebvreCasinoTrial(
                casino_id=casino_id,
                valid_actions=machines,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
            )
        )
    return trials


def parse_wilson_slot_trials(text: str) -> List[WilsonSlotTrial]:
    """Recover Wilson slot-machine trials with per-game arm outcome pools."""
    game_starts = {
        match.start(): (int(match.group("game_number")), int(match.group("n_trials")))
        for match in WILSON_GAME_HEADER_RE.finditer(text)
    }
    raw_free = []
    for match in WILSON_FREE_RE.finditer(text):
        game_number = 1
        for start, (number, _) in sorted(game_starts.items()):
            if start <= match.start():
                game_number = number
        raw_free.append(
            (
                game_number,
                match.group("human_action").strip(),
                int(match.group("points")),
                match.start(),
            )
        )

    rewards_by_game: Dict[int, Dict[str, List[int]]] = {}
    for game_number, arm, points, _ in raw_free:
        rewards_by_game.setdefault(game_number, {}).setdefault(arm, []).append(points)

    def mean_reward(game_number: int, arm: str, fallback: int) -> int:
        samples = rewards_by_game.get(game_number, {}).get(arm, [])
        if not samples:
            return fallback
        return int(round(sum(samples) / len(samples)))

    trials: List[WilsonSlotTrial] = []
    game_number = 1
    for match in WILSON_GAME_HEADER_RE.finditer(text):
        game_number = int(match.group("game_number"))
        header_end = match.end()
        next_game = None
        for next_match in WILSON_GAME_HEADER_RE.finditer(text):
            if next_match.start() > match.start():
                next_game = next_match.start()
                break
        block = text[header_end : next_game if next_game is not None else len(text)]
        for instructed in WILSON_INSTRUCTED_RE.finditer(block):
            arm = instructed.group("arm").strip()
            points = int(instructed.group("points"))
            trials.append(
                WilsonSlotTrial(
                    observation=(
                        "Game {}. There are {} trials in this game.\n"
                        "You are instructed to press {} and get {} points."
                    ).format(
                        game_number,
                        int(match.group("n_trials")),
                        arm,
                        points,
                    ),
                    valid_actions=(arm,),
                    outcomes_by_action={arm: points},
                    trial_type="instructed",
                    human_action=arm,
                )
            )
        for free in WILSON_FREE_RE.finditer(block):
            human_action = free.group("human_action").strip()
            points = int(free.group("points"))
            arms = tuple(
                sorted(
                    {
                        instructed.group("arm").strip()
                        for instructed in WILSON_INSTRUCTED_RE.finditer(block)
                    }
                    | {
                        free_arm
                        for free_game, free_arm, _, _ in raw_free
                        if free_game == game_number
                    }
                )
            )
            if len(arms) != 2:
                arms = tuple(sorted(rewards_by_game.get(game_number, {}).keys()))
            if len(arms) != 2:
                arms = (human_action, human_action)
            outcomes = {
                human_action: points,
                arms[0]: mean_reward(game_number, arms[0], points),
                arms[1]: mean_reward(game_number, arms[1], points),
            }
            trials.append(
                WilsonSlotTrial(
                    observation="Game {}. Trial {}.".format(game_number, len(trials) + 1),
                    valid_actions=arms,
                    outcomes_by_action=outcomes,
                    trial_type="free",
                    human_action=human_action,
                )
            )

    if not trials:
        raise TranscriptParseError("No Wilson slot-machine trials found.")
    return trials


def _hilbig_correct_action(
    ratings_a: Tuple[int, ...],
    ratings_b: Tuple[int, ...],
    label_a: str = "A",
    label_b: str = "R",
) -> str:
    weights = (0.9, 0.8, 0.7, 0.6)
    score_a = sum(rating * weight for rating, weight in zip(ratings_a, weights))
    score_b = sum(rating * weight for rating, weight in zip(ratings_b, weights))
    if score_a > score_b:
        return label_a
    if score_b > score_a:
        return label_b
    return label_a


def _parse_hilbig_rating_vector(raw: str) -> Tuple[int, ...]:
    values = tuple(int(value) for value in raw.strip().split())
    if len(values) != 4:
        raise TranscriptParseError("Hilbig rating vector must contain four values.")
    return values


def parse_hilbig_product_trials(text: str) -> List[HilbigProductTrial]:
    """Recover Hilbig product trials with normative correct choices."""
    trials = []
    for match in HILBIG_TRIAL_RE.finditer(text):
        label_a = match.group("label_a").strip()
        label_b = match.group("label_b").strip()
        ratings_a = _parse_hilbig_rating_vector(match.group("ratings_a"))
        ratings_b = _parse_hilbig_rating_vector(match.group("ratings_b"))
        observation = (
            "Product {} ratings: {}. Product {} ratings: {}."
        ).format(label_a, list(ratings_a), label_b, list(ratings_b))
        trials.append(
            HilbigProductTrial(
                observation=observation,
                valid_actions=(label_a, label_b),
                correct_action=_hilbig_correct_action(ratings_a, ratings_b, label_a, label_b),
                ratings_a=ratings_a,
                ratings_b=ratings_b,
                human_action=match.group("human_action").strip(),
            )
        )
    if not trials:
        raise TranscriptParseError("No Hilbig product trials found.")
    return trials


def _parse_cox_studied_pairs_block(pairs_text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in pairs_text.strip().splitlines():
        if "," not in line:
            continue
        left, right = line.split(",", 1)
        pairs.append((left.strip(), right.strip()))
    if len(pairs) != 20:
        raise TranscriptParseError(
            "Expected 20 studied word pairs, found {}.".format(len(pairs))
        )
    return pairs


def _cox_pair_is_studied(pair_text: str, studied_pairs: List[Tuple[str, str]]) -> bool:
    normalized = tuple(word.strip() for word in pair_text.split(",", 1))
    if len(normalized) != 2:
        raise TranscriptParseError("Malformed Cox test pair: {!r}.".format(pair_text))
    for left, right in studied_pairs:
        if normalized == (left, right) or normalized == (right, left):
            return True
    return False


def parse_cox_pair_recognition_trials(text: str) -> List[CoxPairRecognitionTrial]:
    """Recover Cox pair-recognition trials with session-specific response keys."""
    trials: List[CoxPairRecognitionTrial] = []
    for section in COX_PAIR_RECOGNITION_SECTION_RE.finditer(text):
        studied_pairs = _parse_cox_studied_pairs_block(section.group("pairs"))
        studied_key = section.group("studied_key").strip()
        novel_key = section.group("novel_key").strip()
        valid_actions = (studied_key, novel_key)
        for match in COX_PAIR_TEST_TRIAL_RE.finditer(section.group("body")):
            pair_text = match.group("pair").strip()
            human_action = match.group("human_action").strip()
            if human_action not in valid_actions:
                continue
            is_studied = _cox_pair_is_studied(pair_text, studied_pairs)
            correct_action = studied_key if is_studied else novel_key
            trials.append(
                CoxPairRecognitionTrial(
                    observation="You view the word pair {}.".format(pair_text),
                    valid_actions=valid_actions,
                    correct_action=correct_action,
                    human_action=human_action,
                )
            )
    if not trials:
        raise TranscriptParseError("No Cox pair-recognition trials found.")
    return trials


def parse_kool_cost_exp1_days(text: str) -> List[KoolCostExp1Day]:
    """Recover single-step cost-task days with pooled ship treasure counterfactuals."""
    raw_days = []
    for match in KOOL_COST_EXP1_DAY_RE.finditer(text):
        multiplier = 5 if match.group("multiplier").lower().startswith("a") else 1
        ships = (match.group("ship0").strip(), match.group("ship1").strip())
        ship = match.group("ship").strip()
        planet = match.group("planet").strip()
        base = int(match.group("base"))
        received = int(match.group("received"))
        raw_days.append((multiplier, ships, ship, planet, base, received))

    if not raw_days:
        raise TranscriptParseError("No Kool cost exp1 days found.")

    ship_planet_votes: Dict[str, Counter] = {}
    treasure_pool: Dict[str, List[int]] = {}
    for _multiplier, ships, ship, planet, _base, _received in raw_days:
        ship_planet_votes.setdefault(ship, Counter())[planet] += 1
        treasure_pool.setdefault(ship, []).append(_base)

    planet_by_ship = {
        ship: votes.most_common(1)[0][0] for ship, votes in ship_planet_votes.items()
    }
    pooled_treasure_by_ship = {
        ship: sum(values) / len(values) for ship, values in treasure_pool.items()
    }

    days: List[KoolCostExp1Day] = []
    for multiplier, ships, ship, planet, base, received in raw_days:
        days.append(
            KoolCostExp1Day(
                multiplier=multiplier,
                ships=ships,
                planet_by_ship=dict(planet_by_ship),
                pooled_treasure_by_ship=pooled_treasure_by_ship,
                human_ship=ship,
                human_planet=planet,
                human_base_treasure=base,
                human_received=received,
            )
        )
    return days


def parse_gonogo_trials(text: str) -> List[GonogoTrial]:
    """Recover go/no-go trials with session-specific response keys."""
    key_match = GONOGO_GO_KEY_RE.search(text)
    if key_match is None:
        raise TranscriptParseError("No go/no-go response key found.")
    go_key = key_match.group(1).strip()

    phase_match = GONOGO_PHASE_COUNTS_RE.search(text)
    n_practice = int(phase_match.group(1)) if phase_match else 0

    trials: List[GonogoTrial] = []
    for index, match in enumerate(GONOGO_TRIAL_RE.finditer(text)):
        stimulus = match.group("stimulus").strip()
        key = match.group("key")
        rt_raw = match.group("rt")
        human_key = key.strip() if key else None
        human_rt = float(rt_raw) if rt_raw else None
        trials.append(
            GonogoTrial(
                stimulus=stimulus,
                go_key=go_key,
                human_key=human_key,
                human_rt_ms=human_rt,
                is_practice=index < n_practice,
            )
        )
    if not trials:
        raise TranscriptParseError("No go/no-go trials found.")
    return trials


def _parse_kool_cost_exp2_aliens_by_planet(text: str) -> Dict[str, Tuple[str, str]]:
    match = KOOL_COST_EXP2_ALIENS_RE.search(text)
    if match is None:
        raise TranscriptParseError("No Kool cost exp2 planet-alien mapping found.")
    planet_a, alien_a0, alien_a1, planet_b, alien_b0, alien_b1 = match.groups()
    return {
        planet_a.strip(): (alien_a0.strip(), alien_a1.strip()),
        planet_b.strip(): (alien_b0.strip(), alien_b1.strip()),
    }


def parse_kool_cost_exp2_days(text: str) -> List[KoolCostExp2Day]:
    """Recover two-step cost-task days with pooled alien treasure counterfactuals."""
    header = KOOL_COST_EXP2_HEADER_RE.search(text)
    if header is None:
        raise TranscriptParseError("No Kool cost exp2 instruction header found.")
    aliens_by_planet = _parse_kool_cost_exp2_aliens_by_planet(text)

    raw_days = []
    for match in KOOL_COST_EXP2_DAY_RE.finditer(text):
        multiplier = 5 if match.group("multiplier").lower().startswith("a") else 1
        ships = (match.group("ship0").strip(), match.group("ship1").strip())
        ship = match.group("ship").strip()
        planet = match.group("planet").strip()
        alien = match.group("alien").strip()
        base = int(match.group("base"))
        raw_days.append((multiplier, ships, ship, planet, alien, base))

    if not raw_days:
        raise TranscriptParseError("No Kool cost exp2 days found.")

    ship_planet_votes: Dict[str, Counter] = {}
    treasure_pool: Dict[str, List[int]] = {}
    for _multiplier, ships, ship, planet, alien, base in raw_days:
        ship_planet_votes.setdefault(ship, Counter())[planet] += 1
        treasure_pool.setdefault(alien, []).append(base)

    planet_by_ship = {
        ship: votes.most_common(1)[0][0] for ship, votes in ship_planet_votes.items()
    }
    pooled_treasure_by_alien = {
        alien: sum(values) / len(values) for alien, values in treasure_pool.items()
    }

    days: List[KoolCostExp2Day] = []
    for multiplier, ships, ship, planet, alien, base in raw_days:
        days.append(
            KoolCostExp2Day(
                multiplier=multiplier,
                ships=ships,
                planet_by_ship=dict(planet_by_ship),
                aliens_by_planet=aliens_by_planet,
                pooled_treasure_by_alien=pooled_treasure_by_alien,
                human_ship=ship,
                human_alien=alien,
                human_planet=planet,
                human_base_treasure=base,
            )
        )
    return days


def _flesch_tree_observation(garden: str, leafiness: int, branchiness: int) -> str:
    return (
        "You get a tree with level {} of leafiness and level {} of branchiness in the {} garden."
    ).format(leafiness, branchiness, garden)


def parse_flesch_tree_trials(text: str) -> List[FleschTreeTrial]:
    """Recover Flesch tree-planting trials with session-specific accept/reject keys."""
    key_match = FLESCH_RESPONSE_KEYS_RE.search(text)
    if key_match is None:
        raise TranscriptParseError("No Flesch accept/reject response keys found.")
    accept_key = key_match.group(1).strip()
    reject_key = key_match.group(2).strip()
    valid_actions = (accept_key, reject_key)

    trials: List[FleschTreeTrial] = []
    for match in FLESCH_TREE_TRIAL_RE.finditer(text):
        human_action = match.group("human").strip()
        if human_action not in valid_actions:
            continue
        garden = match.group("garden").strip()
        leafiness = int(match.group("leaf"))
        branchiness = int(match.group("branch"))
        observation = _flesch_tree_observation(garden, leafiness, branchiness)
        if match.group("human_reward") is not None:
            human_reward = float(match.group("human_reward"))
            alt_reward = float(match.group("alt_reward"))
            if human_action == accept_key:
                outcomes_by_action = {accept_key: human_reward, reject_key: alt_reward}
            else:
                outcomes_by_action = {reject_key: human_reward, accept_key: alt_reward}
            trials.append(
                FleschTreeTrial(
                    observation=observation,
                    valid_actions=valid_actions,
                    outcomes_by_action=outcomes_by_action,
                    human_action=human_action,
                    has_feedback=True,
                    garden=garden,
                    leafiness=leafiness,
                    branchiness=branchiness,
                )
            )
            continue
        trials.append(
            FleschTreeTrial(
                observation=observation,
                valid_actions=valid_actions,
                outcomes_by_action=None,
                human_action=human_action,
                has_feedback=False,
                garden=garden,
                leafiness=leafiness,
                branchiness=branchiness,
            )
        )
    if not trials:
        raise TranscriptParseError("No Flesch tree-planting trials found.")

    pooled: Dict[Tuple[str, int, int], Dict[str, List[float]]] = {}
    for trial in trials:
        if not trial.has_feedback or trial.outcomes_by_action is None:
            continue
        key = (trial.garden, trial.leafiness, trial.branchiness)
        for action, reward in trial.outcomes_by_action.items():
            pooled.setdefault(key, {}).setdefault(action, []).append(reward)

    filled: List[FleschTreeTrial] = []
    for trial in trials:
        if trial.has_feedback or trial.outcomes_by_action is not None:
            filled.append(trial)
            continue
        key = (trial.garden, trial.leafiness, trial.branchiness)
        bucket = pooled.get(key)
        if bucket and all(action in bucket for action in valid_actions):
            outcomes_by_action = {
                action: sum(values) / len(values) for action, values in bucket.items()
            }
            filled.append(
                FleschTreeTrial(
                    observation=trial.observation,
                    valid_actions=trial.valid_actions,
                    outcomes_by_action=outcomes_by_action,
                    human_action=trial.human_action,
                    has_feedback=True,
                    garden=trial.garden,
                    leafiness=trial.leafiness,
                    branchiness=trial.branchiness,
                )
            )
        else:
            filled.append(trial)
    return filled


def parse_digit_span_recall_trials(text: str) -> List[DigitSpanRecallTrial]:
    """Recover digit-span recall keypresses with session-specific end keys."""
    if not DIGIT_SPAN_END_KEY_RE.search(text):
        raise TranscriptParseError("No digit-span end key found.")
    digit_actions = tuple(str(digit) for digit in range(10))
    trials: List[DigitSpanRecallTrial] = []
    span_index = 0
    for block in DIGIT_SPAN_BLOCK_RE.finditer(text):
        target_digits = [part.strip() for part in block.group("digits").split(",")]
        presses = re.findall(r"You\s+press\s+<<([^>]+)>>", block.group("presses"))
        if not presses:
            continue
        end_key = presses[-1]
        valid_actions = digit_actions + (end_key,)
        expected = target_digits + [end_key]
        if len(presses) < len(expected):
            continue
        header = "The digits are the following: [{}]".format(block.group("digits").strip())
        span_length = len(expected)
        for position, human_action in enumerate(presses):
            correct_action = expected[position] if position < len(expected) else end_key
            observation = header if position == 0 else ""
            trials.append(
                DigitSpanRecallTrial(
                    observation=observation,
                    valid_actions=valid_actions,
                    correct_action=correct_action,
                    human_action=human_action,
                    span_index=span_index,
                    span_length=span_length,
                )
            )
        span_index += 1
    if not trials:
        raise TranscriptParseError("No digit-span recall trials found.")
    return trials


def parse_steingroever_igt_trials(text: str) -> List[SteingroeverIGTTrial]:
    """Recover Iowa Gambling Task trials with pooled deck win/loss outcomes."""
    raw = []
    for match in STEINGROEVER_IGT_TRIAL_RE.finditer(text):
        raw.append(
            (
                match.group("deck").strip(),
                float(match.group("win")),
                float(match.group("loss")),
            )
        )
    if not raw:
        raise TranscriptParseError("No Steingroever IGT trials found.")

    pools: Dict[str, List[Tuple[float, float]]] = {}
    for deck, win, loss in raw:
        pools.setdefault(deck, []).append((win, loss))

    decks = tuple(sorted(pools))
    trials: List[SteingroeverIGTTrial] = []
    for deck, win, loss in raw:
        outcomes_by_action = {
            action: (
                sum(values[0] for values in pools[action]) / len(pools[action]),
                sum(values[1] for values in pools[action]) / len(pools[action]),
            )
            for action in decks
        }
        trials.append(
            SteingroeverIGTTrial(
                valid_actions=decks,
                outcomes_by_action=outcomes_by_action,
                human_action=deck,
            )
        )
    return trials


def _parse_tomov_castle_door_keys(text: str) -> Tuple[str, ...]:
    match = TOMOV_CASTLE_DOORS_RE.search(text)
    if match is None:
        raise TranscriptParseError("Tomov castle transcript is missing door labels.")
    raw = match.group(1).replace(" and ", ", ")
    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if len(keys) < 2:
        raise TranscriptParseError("Tomov castle door labels could not be parsed.")
    return keys


def parse_tomov_castle_trials(text: str) -> List[TomovCastleTrial]:
    """Recover Tomov castle rounds with pooled per-room door rewards."""
    door_keys = _parse_tomov_castle_door_keys(text)
    price_starts = {
        match.start(): (
            float(match.group("wood")),
            float(match.group("stone")),
            float(match.group("iron")),
            match.group(0),
        )
        for match in TOMOV_CASTLE_PRICES_RE.finditer(text)
    }
    price_positions = sorted(price_starts)
    raw = []
    for match in TOMOV_CASTLE_STEP_RE.finditer(text):
        round_number = 1
        market_prices = (0.0, 0.0, 0.0)
        price_line = ""
        for price_start in price_positions:
            if price_start <= match.start():
                wood_price, stone_price, iron_price, line = price_starts[price_start]
                market_prices = (wood_price, stone_price, iron_price)
                price_line = line
        round_number = max(
            1,
            sum(1 for price_start in price_positions if price_start <= match.start()),
        )
        room_number = int(match.group("room"))
        human_action = match.group("human_action").strip()
        reward = float(match.group("reward"))
        resources = (
            float(match.group("wood")),
            float(match.group("stone")),
            float(match.group("iron")),
        )
        step_line = match.group(0)
        raw.append(
            (
                round_number,
                room_number,
                human_action,
                reward,
                market_prices,
                price_line,
                resources,
                step_line,
            )
        )

    if not raw:
        raise TranscriptParseError("No Tomov castle trials found.")

    pools: Dict[Tuple[int, str], List[float]] = {}
    for _, room_number, human_action, reward, _, _, _, _ in raw:
        pools.setdefault((room_number, human_action), []).append(reward)

    trials: List[TomovCastleTrial] = []
    previous_round = None
    for (
        round_number,
        room_number,
        human_action,
        reward,
        market_prices,
        price_line,
        resources,
        step_line,
    ) in raw:
        outcomes_by_action = {action: 0.0 for action in door_keys}
        for action in door_keys:
            samples = pools.get((room_number, action))
            if samples:
                outcomes_by_action[action] = sum(samples) / len(samples)
        valid_actions = door_keys
        if human_action not in door_keys:
            valid_actions = tuple(sorted(set(door_keys + (human_action,))))
            outcomes_by_action[human_action] = reward
        pre_step = "You are in room {}.".format(room_number)
        if round_number != previous_round:
            observation = "{}\n{}".format(price_line, pre_step)
            show_round_header = True
        else:
            observation = pre_step
            show_round_header = False
        trials.append(
            TomovCastleTrial(
                observation=observation,
                room_number=room_number,
                valid_actions=valid_actions,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
                round_number=round_number,
                show_round_header=show_round_header,
                market_prices=market_prices,
                resource_amounts=resources,
            )
        )
        previous_round = round_number
    return trials


def _parse_tomov_direction_keys(text: str) -> Dict[str, str]:
    direction_match = TOMOV_DIRECTION_KEYS_RE.search(text)
    goal_match = TOMOV_GOAL_KEY_RE.search(text)
    if direction_match is None or goal_match is None:
        raise TranscriptParseError("Tomov subway transcript is missing direction key instructions.")
    return {
        "north": direction_match.group("north").strip(),
        "west": direction_match.group("west").strip(),
        "south": direction_match.group("south").strip(),
        "east": direction_match.group("east").strip(),
        "goal": goal_match.group("goal").strip(),
    }


def _tomov_valid_actions(neighbors_text: str, station: str, goal_station: str, keys: Dict[str, str]) -> Tuple[str, ...]:
    actions: List[str] = []
    neighbor_text = neighbors_text.lower()
    if "circle on the north" not in neighbor_text:
        actions.append(keys["north"])
    if "circle on the west" not in neighbor_text:
        actions.append(keys["west"])
    if "circle on the south" not in neighbor_text:
        actions.append(keys["south"])
    if "circle on the east" not in neighbor_text:
        actions.append(keys["east"])
    if station == goal_station:
        actions.append(keys["goal"])
    return tuple(actions)


def parse_tomov_subway_trials(text: str) -> List[TomovSubwayTrial]:
    """Recover Tomov subway navigation steps with direction keys from instructions."""
    keys = _parse_tomov_direction_keys(text)
    rounds = [
        (
            match.start(),
            match.group("start"),
            match.group("goal"),
            "The new starting station is {} and the goal station is {}.".format(
                match.group("start"), match.group("goal")
            ),
        )
        for match in TOMOV_ROUND_RE.finditer(text)
    ]
    trials: List[TomovSubwayTrial] = []
    previous_round_key = None
    round_number = 0
    for match in TOMOV_STATION_STEP_RE.finditer(text):
        start_station = rounds[0][1] if rounds else "1"
        goal_station = rounds[0][2] if rounds else "1"
        round_header = rounds[0][3] if rounds else ""
        for round_start, start_value, goal_value, header in rounds:
            if round_start <= match.start():
                start_station = start_value
                goal_station = goal_value
                round_header = header
        round_key = (start_station, goal_station)
        if round_key != previous_round_key:
            round_number += 1
            previous_round_key = round_key
            show_round_header = True
        else:
            show_round_header = False

        station = match.group("station")
        neighbors = match.group("neighbors")
        human_action = match.group("human_action").strip()
        station_observation = "Your station: {}. Neighboring stations: {}.".format(
            station, neighbors
        )
        observation = (
            "{}\n{}".format(round_header, station_observation)
            if show_round_header
            else station_observation
        )
        valid_actions = _tomov_valid_actions(neighbors, station, goal_station, keys)
        if human_action not in valid_actions:
            valid_actions = tuple(sorted(set(valid_actions + (human_action,))))
        completes_round = station == goal_station and human_action == keys["goal"]
        trials.append(
            TomovSubwayTrial(
                observation=observation,
                valid_actions=valid_actions,
                human_action=human_action,
                round_number=round_number,
                show_round_header=show_round_header,
                completes_round=completes_round,
            )
        )

    if not trials:
        raise TranscriptParseError("No Tomov subway trials found.")
    return trials


def parse_schulz_finding_trials(text: str) -> List[SchulzFindingTrial]:
    """Recover Schulz finding rounds with pooled per-round arm outcomes."""
    round_starts = {
        match.start(): int(match.group("round"))
        for match in SCHULZ_ROUND_RE.finditer(text)
    }
    raw = []
    for match in SCHULZ_TRIAL_RE.finditer(text):
        round_number = 1
        for start, number in sorted(round_starts.items()):
            if start <= match.start():
                round_number = number
        human_action = match.group("human_action").strip()
        outcome = float(match.group("outcome"))
        raw.append((round_number, human_action, outcome))

    if not raw:
        raise TranscriptParseError("No Schulz finding trials found.")

    pools: Dict[Tuple[int, str], List[float]] = {}
    for round_number, human_action, outcome in raw:
        pools.setdefault((round_number, human_action), []).append(outcome)

    round_arms_map = {
        round_number: tuple(
            sorted({arm for rn, arm, _ in raw if rn == round_number}, key=int)
        )
        for round_number in {rn for rn, _, _ in raw}
    }
    use_eight_arms = SCHULZ_EIGHT_ARM_HINT.lower() in text.lower()

    trials: List[SchulzFindingTrial] = []
    previous_round = None
    for round_number, human_action, _ in raw:
        round_arms = DEFAULT_SCHULZ_ARMS if use_eight_arms else round_arms_map[round_number]
        outcomes_by_action = {}
        for arm in round_arms:
            samples = pools.get((round_number, arm), [])
            if samples:
                outcomes_by_action[arm] = sum(samples) / len(samples)
            else:
                outcomes_by_action[arm] = 0.0
        trials.append(
            SchulzFindingTrial(
                round_number=round_number,
                valid_actions=round_arms,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
                show_round_header=round_number != previous_round,
            )
        )
        previous_round = round_number
    return trials


def parse_bahrami_four_arm_trials(text: str) -> List[BahramiFourArmTrial]:
    """Recover Bahrami trials with pooled per-arm outcomes."""
    raw = []
    for match in BAHRAMI_TRIAL_RE.finditer(text):
        raw.append((match.group("human_action").strip(), float(match.group("outcome"))))
    if not raw:
        raise TranscriptParseError("No Bahrami bandit trials found.")
    pools: Dict[str, List[float]] = {}
    for action, outcome in raw:
        pools.setdefault(action, []).append(outcome)
    valid_actions = tuple(sorted(pools))
    trials = []
    for human_action, outcome in raw:
        outcomes_by_action = {
            action: sum(values) / len(values) for action, values in pools.items()
        }
        trials.append(
            BahramiFourArmTrial(
                valid_actions=valid_actions,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
            )
        )
    return trials


def parse_gershman_bandit_trials(text: str) -> List[GershmanBanditTrial]:
    """Recover Gershman deconstructing bandit trials with pooled arm outcomes."""
    game_starts = {
        match.start(): int(match.group("game_number")) for match in GERSHMAN_GAME_RE.finditer(text)
    }
    raw = []
    for match in GERSHMAN_BANDIT_TRIAL_RE.finditer(text):
        game_number = 1
        for start, number in sorted(game_starts.items()):
            if start <= match.start():
                game_number = number
        raw.append(
            (
                game_number,
                match.group("human_action").strip(),
                int(match.group("points")),
                match.start(),
            )
        )
    if not raw:
        raise TranscriptParseError("No Gershman bandit trials found.")

    pools: Dict[Tuple[int, str], List[int]] = {}
    for game_number, action, points, _ in raw:
        pools.setdefault((game_number, action), []).append(points)

    trials: List[GershmanBanditTrial] = []
    previous_game = None
    for game_number, human_action, _, _ in raw:
        arms = []
        for gid, action, _, _ in raw:
            if gid == game_number and action not in arms:
                arms.append(action)
        outcomes_by_action = {
            arm: int(round(sum(pools[(game_number, arm)]) / len(pools[(game_number, arm)])))
            for arm in arms
        }
        if len(arms) == 1:
            valid_actions = (arms[0],)
        else:
            valid_actions = (arms[0], arms[1])
        show_game_header = game_number != previous_game
        previous_game = game_number
        trials.append(
            GershmanBanditTrial(
                game_number=game_number,
                valid_actions=valid_actions,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
                show_game_header=show_game_header,
            )
        )
    return trials


def _parse_wulff_lottery_distribution(line: str) -> Tuple[Tuple[float, float], ...]:
    parts = WULFF_OUTCOME_PART_RE.findall(line)
    if not parts:
        raise TranscriptParseError("No Wulff lottery outcomes found in line: {!r}.".format(line))
    return tuple((float(value), float(probability) / 100.0) for value, probability in parts)


def parse_plonsky_option_distribution(option_line: str) -> List[Tuple[float, float]]:
    """Parse a Plonsky option line into value/probability pairs."""
    parts = PLONSKY_OUTCOME_PART_RE.findall(option_line)
    if not parts:
        raise TranscriptParseError(
            "No Plonsky outcome probabilities found in line: {!r}.".format(option_line)
        )
    return [(float(value), float(probability) / 100.0) for value, probability in parts]


def _parse_wulff_sampling_session_keys(
    text: str,
) -> Tuple[Tuple[str, str], Optional[str], Optional[int]]:
    """Read lottery arm labels and either a stop key or a fixed sample budget."""
    arms_match = WULFF_SAMPLING_ARMS_RE.search(text)
    if arms_match is None:
        raise TranscriptParseError("No Wulff sampling lottery arm labels found in instructions.")
    sampling_arms = (arms_match.group("arm0").upper(), arms_match.group("arm1").upper())
    stop_match = WULFF_SAMPLING_STOP_RE.search(text)
    if stop_match is not None:
        return sampling_arms, stop_match.group("stop").upper(), None
    fixed_match = WULFF_FIXED_SAMPLE_COUNT_RE.search(text)
    if fixed_match is not None:
        return sampling_arms, None, int(fixed_match.group("count"))
    raise TranscriptParseError(
        "No Wulff sampling stop key or fixed sample budget found in instructions."
    )


def parse_wulff_sampling_problems(text: str) -> List[WulffSamplingProblem]:
    """Recover Wulff sampling problems with per-arm outcome pools."""
    sampling_arms, stop_action, fixed_sample_count = _parse_wulff_sampling_session_keys(text)
    header, *blocks = re.split(
        r"\nYou encounter a new choice problem:\n", text, flags=re.IGNORECASE
    )
    del header
    problems: List[WulffSamplingProblem] = []
    for problem_number, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        pools: Dict[str, List[float]] = {arm: [] for arm in sampling_arms}
        sample_sequence: List[Tuple[str, float]] = []
        for match in WULFF_SAMPLING_SAMPLE_RE.finditer(block):
            arm = match.group("arm").strip().upper()
            if arm not in pools:
                continue
            outcome = float(match.group("outcome"))
            pools[arm].append(outcome)
            sample_sequence.append((arm, outcome))
        if stop_action is not None:
            final_match = WULFF_SAMPLING_FINAL_RE.search(block)
            if final_match is None:
                raise TranscriptParseError(
                    "Wulff sampling problem {} is missing a final lottery choice.".format(
                        problem_number
                    )
                )
            if final_match.group("stop").strip().upper() != stop_action:
                raise TranscriptParseError(
                    "Wulff sampling problem {} uses stop key <<{}>>; expected <<{}>>.".format(
                        problem_number, final_match.group("stop"), stop_action
                    )
                )
            final_arm = final_match.group("arm").strip().upper()
        else:
            final_match = WULFF_FIXED_FINAL_RE.search(block)
            if final_match is None:
                raise TranscriptParseError(
                    "Wulff sampling problem {} is missing a fixed-budget final choice.".format(
                        problem_number
                    )
                )
            final_arm = final_match.group("arm").strip().upper()
        if final_arm not in pools:
            raise TranscriptParseError(
                "Wulff sampling problem {} final arm <<{}>> is not one of {}.".format(
                    problem_number, final_arm, sampling_arms
                )
            )
        final_outcomes = {
            arm: (outcomes[-1] if outcomes else 0.0) for arm, outcomes in pools.items()
        }
        problems.append(
            WulffSamplingProblem(
                problem_number=problem_number,
                sample_pools=pools,
                final_outcomes_by_action=final_outcomes,
                sampling_arms=sampling_arms,
                stop_action=stop_action,
                fixed_sample_count=fixed_sample_count,
                human_final_action=final_arm,
                sample_sequence=tuple(sample_sequence),
            )
        )
    if not problems:
        raise TranscriptParseError("No Wulff sampling problems found.")
    return problems


def parse_wulff_description_trials(text: str) -> List[WulffLotteryTrial]:
    """Recover Wulff description lottery problems with deterministic paired draws."""
    trials: List[WulffLotteryTrial] = []
    for block_idx, match in enumerate(WULFF_LOTTERY_PRESS_RE.finditer(text)):
        observation = match.group(1).strip()
        human_action = match.group("human_action").strip()
        distributions: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        for line in observation.splitlines():
            line_match = WULFF_LOTTERY_LINE_RE.match(line.strip())
            if not line_match:
                continue
            label = line_match.group(1).strip()
            distributions[label] = _parse_wulff_lottery_distribution(line_match.group(2))
        if len(distributions) != 2:
            raise TranscriptParseError("Wulff block does not describe two lotteries.")
        labels = tuple(sorted(distributions))
        rng = random.Random(block_idx)
        outcomes_by_action = {}
        for label in labels:
            dist = distributions[label]
            values = [entry[0] for entry in dist]
            probabilities = [entry[1] for entry in dist]
            draw = rng.random()
            cumulative = 0.0
            chosen = values[-1]
            for value, probability in zip(values, probabilities):
                cumulative += probability
                if draw <= cumulative:
                    chosen = value
                    break
            outcomes_by_action[label] = chosen
        trials.append(
            WulffLotteryTrial(
                observation=observation,
                valid_actions=labels,
                outcomes_by_action=outcomes_by_action,
                human_action=human_action,
            )
        )
    if not trials:
        raise TranscriptParseError("No Wulff description lottery trials found.")
    return trials


def parse_plonsky_gamble_trials(text: str) -> List[PlonskyGambleTrial]:
    """Recover Plonsky gambling problems, including counterfactual feedback trials."""
    trials: List[PlonskyGambleTrial] = []
    for block in PLONSKY_OPTION_BLOCK_RE.finditer(text):
        option_a = block.group("option_a").strip()
        option_b = block.group("option_b").strip()
        key_a = PLONSKY_OPTION_KEY_RE.match(option_a).group(1).strip()
        key_b = PLONSKY_OPTION_KEY_RE.match(option_b).group(1).strip()
        observation = option_a + "\n" + option_b
        body = block.group("body")
        body_actions = set(
            action.strip()
            for action in re.findall(r"You\s+press\s+<<([^<>]+)>>", body, flags=re.IGNORECASE)
        )
        body_actions.update(
            action.strip()
            for action in re.findall(
                r"had\s+you\s+chosen\s+option\s+(\w+)\.", body, flags=re.IGNORECASE
            )
        )
        if key_a != key_b:
            valid_actions = (key_a, key_b)
        else:
            inferred = tuple(sorted(body_actions))
            valid_actions = inferred if len(inferred) == 2 else (key_a, key_b)
        events = []
        for match in PLONSKY_FEEDBACK_TRIAL_RE.finditer(body):
            events.append((match.start(), "feedback", match))
        for match in PLONSKY_NO_FEEDBACK_TRIAL_RE.finditer(body):
            if any(
                feedback.start() <= match.start() < feedback.end()
                for feedback in PLONSKY_FEEDBACK_TRIAL_RE.finditer(body)
            ):
                continue
            events.append((match.start(), "no_feedback", match))
        for _, event_type, match in sorted(events, key=lambda entry: entry[0]):
            if event_type == "feedback":
                human_action = match.group("human_action").strip()
                chosen = float(match.group("chosen"))
                if match.group("verb") == "lose":
                    chosen = -chosen
                forgone = float(match.group("forgone"))
                if match.group("alt_verb") == "lost":
                    forgone = -forgone
                alt_action = match.group("alt_action").strip()
                outcomes = {human_action: chosen, alt_action: forgone}
                trials.append(
                    PlonskyGambleTrial(
                        observation=observation,
                        valid_actions=valid_actions,
                        outcomes_by_action=outcomes,
                        human_action=human_action,
                        has_feedback=True,
                    )
                )
            else:
                trials.append(
                    PlonskyGambleTrial(
                        observation=observation,
                        valid_actions=valid_actions,
                        outcomes_by_action=None,
                        human_action=match.group("human_action").strip(),
                        has_feedback=False,
                    )
                )
    if not trials:
        raise TranscriptParseError("No Plonsky gamble trials found.")
    return trials
