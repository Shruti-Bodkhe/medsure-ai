import re
from dataclasses import dataclass
from typing import Optional

EMERGENCY_PATTERNS = {
    "cardiac": [
        r"chest\s+(?:is\s+)?(?:pain|tight|heavy|pressure|hurts)",
        r"tightness\s+in\s+(?:my\s+)?chest",
        r"pressure\s+in\s+(?:my\s+)?chest",
        r"pain\s+radiating\s+to\s+(?:my\s+)?(?:left\s+)?arm",
        r"left\s+arm\s+hurts",
        r"heart\s+attack",
        r"cardiac\s+arrest",
        r"chha?ati\s+mein\s+dard",
        r"seene\s+mein\s+dard",
        r"छाती",
        r"सीने\s*में.*दर्द",
        r"हार्ट\s*अटैक",
        r"छातीत",
    ],
    "breathing": [
        r"can'?t\s+breathe",
        r"cannot\s+breathe",
        r"shortness\s+of\s+breath",
        r"lips\s+are\s+turning\s+blue",
        r"blue\s+lips",
        r"choking",
        r"gasping",
        r"inhaler\s+is\s+not\s+working",
        r"wheezing\s+is\s+getting\s+worse",
        r"dama\s+ghut",
        r"saans\s+nahi\s+aa",
        r"saans\s+ruk",
        r"सांस\s*लेने\s*में",
        r"दम\s*घुट",
        r"सांस\s*फूल",
        r"श्वास\s*घ्यायला\s*त्रास",
    ],
    "stroke": [
        r"stroke",
        r"face\s+droop",
        r"facial\s+droop",
        r"slurred\s+speech",
        r"sudden\s+weakness\s+on\s+one\s+side",
        r"paralysis",
        r"lakwa",
        r"लकवा",
        r"स्ट्रोक",
        r"स्टोक",
    ],
    "severe_bleeding": [
        r"heavy\s+bleeding",
        r"uncontrolled\s+bleeding",
        r"bleeding\s+won'?t\s+stop",
        r"blood\s+spurt",
        r"bahut\s+khoon",
        r"khoon\s+ruk\s+nahi",
        r"रक्तस्राव",
        r"खूप\s*जास्त\s*खून",
        r"बहुत\s*खून",
    ],
    "poisoning": [
        r"poison",
        r"swallowed\s+(?:rat\s+)?(?:chemical|poison|pesticide)",
        r"overdose",
        r"toxic\s+ingestion",
        r"zahar",
        r"zehar",
        r"neend\s+ki\s+goli\s+kha",
        r"जहर",
        r"विष",
    ],
    "snakebite": [
        r"snake\s*bite",
        r"venomous\s+bite",
        r"saanp\s+ne\s+ka+t",
        r"sarpa\s+dash",
        r"सांप\s*ने\s*काट",
        r"सांप\s*का\s*काटना",
        r"सर्प\s*दंश",
        r"सर्प\s*डंस",
    ],
    "unconscious_seizure": [
        r"unconscious",
        r"passed\s+out",
        r"fainted",
        r"fainting",
        r"unresponsive",
        r"seizure",
        r"convulsion",
        r"daura\s+pada",
        r"behosh",
        r"बेहोश",
        r"दौरा",
        r"मिर्गी",
    ],
    "allergic_reaction": [
        r"anaphylaxis",
        r"swollen\s+throat",
        r"swelling\s+throat",
        r"closed\s+airway",
        r"severe\s+allergic\s+reaction",
        r"gale\s+mein\s+sujan",
        r"गले\s*में\s*सूजन",
    ],
    "pregnancy_emergency": [
        r"pregnant.*(?:bleeding|cramping|pain)",
        r"severe\s+vaginal\s+bleeding",
        r"water\s+broke\s+early",
        r"pregnancy.*(?:bleeding|emergency)",
        r"garbhavastha\s+me\s+raktrav",
        r"गर्भावस्था\s*में\s*रक्तस्राव",
    ],
    "self_harm": [
        r"want\s+to\s+die",
        r"kill\s+myself",
        r"suicide",
        r"suicidal",
        r"end\s+my\s+life",
        r"ending\s+it\s+all",
        r"cannot\s+take\s+it\s+anymore",
        r"mar\s+jana\s+chahta",
        r"jeene\s+ki\s+ichcha\s+nahi",
        r"आत्महत्या",
        r"मरना\s*चाहता",
        r"जिंदगी\s*खत्म",
        r"जीने\s*की\s*इच्छा\s*नहीं",
    ],
}

INFORMATIONAL_PREFIXES = [
    r"^\s*what\s+is\b",
    r"^\s*what\s+are\b",
    r"^\s*how\s+can\s+i\s+prevent\b",
    r"^\s*how\s+to\s+prevent\b",
    r"^\s*how\s+does\b",
    r"^\s*define\b",
    r"^\s*explain\b",
    r"^\s*causes\s+of\b",
    r"^\s*signs\s+of\b",
    r"^\s*symptoms\s+of\b",
    r"^\s*ignore\s+your\s+instructions\b",
]


@dataclass
class EmergencyResult:
    is_emergency: bool
    category: Optional[str] = None
    matched: Optional[str] = None


def check_emergency_rules(query: str) -> Optional[str]:
    q_lower = query.lower().strip()
    if any(re.search(p, q_lower) for p in INFORMATIONAL_PREFIXES):
        return None
    for category, patterns in EMERGENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q_lower):
                return category
    return None


detect = check_emergency_rules