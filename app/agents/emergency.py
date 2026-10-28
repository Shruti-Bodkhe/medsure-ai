import re
from dataclasses import dataclass
from typing import Optional

from app.core.helplines import EMERGENCY, MENTAL_HEALTH

FLAGS = re.IGNORECASE

# category -> patterns (English, Hindi, Marathi, Romanised Hindi)
RULES = {
    "cardiac": [
        r"chest\s+(pain|tightness|pressure|discomfort)",
        r"pain\s+(in|on)\s+(my\s+)?left\s+arm",
        r"left\s+arm\s+(pain|hurts?)",
        r"heart\s+attack",
        r"सीने\s*(में)?\s*(दर्द|जकड़न|भारीपन)",
        r"छाती\s*(में)?\s*दर्द",
        r"दिल\s*का\s*दौरा",
        r"छातीत\s*(दुखत|दुखणे|दुखतंय|दुखतेय)",
        r"(seene|sine|chhati|chati)\s+(me|mein)\s+dard",
        r"chest\s+(me|mein)\s+dard",
    ],
    "breathing": [
        r"(can'?t|cannot|unable\s+to)\s+breathe",
        r"(difficulty|trouble)\s+breathing",
        r"short(ness)?\s+of\s+breath",
        r"(baby|child|infant).{0,20}(not\s+breathing|turning\s+blue)",
        r"सांस\s*(लेने)?\s*में\s*(तकलीफ|दिक्कत|परेशानी)",
        r"सांस\s*नहीं\s*आ",
        r"श्वास\s*घ्यायला\s*(त्रास|अडचण)",
        r"saans\s+(nahi|lene\s+(me|mein)\s+(takleef|dikkat))",
    ],
    "stroke": [
        r"face\s+(is\s+)?drooping",
        r"slurred\s+speech",
        r"sudden\s+(weakness|numbness).{0,30}(one\s+side|arm|face|leg)",
        r"चेहरा\s*टेढ़ा",
        r"लकवा",
    ],
    "severe_bleeding": [
        r"(heavy|severe|uncontrolled)\s+bleeding",
        r"bleeding\s+(won'?t|will\s+not|does\s+not|doesn'?t)\s+stop",
        r"(vomit(ing)?|coughing(\s+up)?)\s+blood",
        r"खून\s*(बंद\s*नहीं|बहना\s*बंद\s*नहीं)",
        r"खून\s*की\s*उल्टी",
    ],
    "poisoning": [
        r"poison(ed|ing)?",
        r"(swallowed|drank|ate)\s+.{0,25}(pesticide|kerosene|bleach|phenyl|rat\s+poison)",
        r"overdose",
        r"ज़?हर",
    ],
    "snakebite": [
        r"snake\s*bite",
        r"bitten\s+by\s+(a\s+)?snake",
        r"snake\s+bit",
        r"सांप\s*(ने)?\s*काट",
        r"साँप\s*(ने)?\s*काट",
        r"साप\s*चावला",
        r"saanp\s+ne\s+kat",
    ],
    "unconscious_seizure": [
        r"unconscious",
        r"not\s+responding",
        r"passed\s+out",
        r"seizure|convulsion",
        r"बेहोश",
        r"दौरा\s*पड़",
        r"बेशुद्ध",
    ],
    "allergic_reaction": [
        r"(throat|tongue|face|lips)\s+(is\s+|are\s+)?swelling",
        r"allergic\s+reaction.{0,40}(breath|swell)",
    ],
    "pregnancy_emergency": [
        r"pregnan\w*.{0,40}(heavy\s+bleeding|bleeding|severe\s+(abdominal\s+)?pain)",
    ],
    "self_harm": [
        r"(want|going|plan(ning)?)\s+to\s+(die|kill\s+myself|end\s+(it|my\s+life))",
        r"kill\s+myself",
        r"end\s+my\s+life",
        r"suicid",
        r"आत्महत्या",
        r"मरना\s*चाह",
        r"मरावंसं\s*वाटत",
        r"mar\s*jana\s*chah",
    ],
}
COMPILED = {cat: [re.compile(p, FLAGS) for p in pats] for cat, pats in RULES.items()}

# General questions ("what is a heart attack?") should not trigger the emergency lane,
# unless the user talks about themselves or someone near them.
INFO_START = re.compile(
    r"^\s*(what\s+(is|are)|define|causes?\s+of|how\s+(to|can\s+i|do\s+i)\s+prevent|"
    r"prevention\s+of|symptoms?\s+of|tell\s+me\s+about)\b", FLAGS)
FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|my|me|mine|he|she|his|her|mother|father|mom|dad|wife|husband|"
    r"baby|child|son|daughter|friend)\b|मुझे|मेरे|मेरी|मेरा|मला|माझ|उसे|उन्हें", FLAGS)


@dataclass
class EmergencyResult:
    is_emergency: bool
    category: Optional[str] = None
    matched: Optional[str] = None


def detect(text: str) -> EmergencyResult:
    info = INFO_START.search(text)
    if info and not FIRST_PERSON.search(text[info.end():]):
        return EmergencyResult(False)
    for category, patterns in COMPILED.items():
        for p in patterns:
            m = p.search(text)
            if m:
                return EmergencyResult(True, category, m.group(0))
    return EmergencyResult(False)


MARATHI_WORDS = ("आहे", "आहेत", "काय", "मला", "नाही", "आणि", "करा", "झाले")


def guess_lang(text: str) -> str:
    """Rough guess only. The real Language Agent will replace this."""
    if re.search(r"[\u0900-\u097F]", text):
        return "mr" if any(w in text for w in MARATHI_WORDS) else "hi"
    return "en"


# Marathi and Hindi wording must be reviewed by a native speaker before any demo.
TEMPLATES = {
    "emergency": {
        "en": ("This may be a medical emergency. Call {e} now or go to the nearest "
               "hospital immediately. MedSure AI gives general information only and "
               "cannot diagnose."),
        "hi": ("यह एक मेडिकल इमरजेंसी हो सकती है। अभी {e} पर कॉल करें या तुरंत नज़दीकी "
               "अस्पताल जाएं। MedSure AI केवल सामान्य जानकारी देता है और निदान नहीं कर सकता।"),
        "mr": ("ही वैद्यकीय आणीबाणी असू शकते. आत्ताच {e} वर कॉल करा किंवा त्वरित जवळच्या "
               "रुग्णालयात जा. MedSure AI फक्त सामान्य माहिती देतो आणि निदान करू शकत नाही."),
    },
    "self_harm": {
        "en": ("I'm really sorry you're going through this. You don't have to face it "
               "alone. Please call Tele-MANAS at {m} (free, 24/7), or call {e} if you "
               "are in immediate danger. If you can, ask someone you trust to be with "
               "you right now."),
        "hi": ("आप जिस दौर से गुज़र रहे हैं, उसके लिए मुझे बहुत दुख है। आप अकेले नहीं हैं। "
               "कृपया Tele-MANAS को {m} पर कॉल करें (मुफ़्त, 24 घंटे), और अगर आप तुरंत "
               "खतरे में हैं तो {e} पर कॉल करें। किसी भरोसेमंद व्यक्ति को अभी अपने पास बुला लें।"),
        "mr": ("तुम्ही ज्या परिस्थितीतून जात आहात त्याबद्दल मला खूप वाईट वाटते. तुम्ही एकटे "
               "नाही आहात. कृपया Tele-MANAS ला {m} वर कॉल करा (मोफत, 24 तास), आणि तुम्ही "
               "त्वरित धोक्यात असाल तर {e} वर कॉल करा. एखाद्या विश्वासू व्यक्तीला आत्ताच "
               "तुमच्याजवळ बोलवा."),
    },
}


def build_response(result: EmergencyResult, lang: str = "en") -> str:
    kind = "self_harm" if result.category == "self_harm" else "emergency"
    return TEMPLATES[kind][lang].format(e=EMERGENCY, m=MENTAL_HEALTH)