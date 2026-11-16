import re
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EmergencyMatch:
    is_emergency: bool
    category: Optional[str] = None
    matched_pattern: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_emergency

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.category == other
        return super().__eq__(other)

EMERGENCY_PATTERNS: Dict[str, List[str]] = {
    "cardiac": [
        r"\b(chest\s*(pain|tightness|pressure|heaviness|discomfort|crush|burn|ache|hurts))\b",
        r"\b(chest\s+is\s+(tight|heavy|crushing|paining|hurting))\b",
        r"\b(chest\s+feels\s+(tight|heavy|crushing))\b",
        r"\b(tightness\s+in\s+(my\s+|the\s+)?chest)\b",
        r"\b(heart\s*attack|myocardial\s*infarction|cardiac\s*arrest)\b",
        r"\b(pain\s*radiating\s*to\s*(left\s*)?(arm|jaw|neck|back))\b",
        r"\b(left\s*arm\s*(pain|numbness|tingling|hurts))\b",
        r"(सीने\s*में\s*(अचानक\s*)?(बहुत\s*)?(तेज\s*|भारी\s*)?(दर्द|जकड़न|भारीपन|दबाव))",
        r"(छाती\s*में\s*(अचानक\s*)?(बहुत\s*)?(तेज\s*|भारी\s*)?(दर्द|जकड़न|भारीपन|दबाव))",
        r"(छातीत\s*(अचानक\s*)?(खूप\s*|तीव्र\s*)?(दुखतंय|दुखत\s*आहे|वेदना|जळजळ|दडपण|कळ))",
        r"(हृदयविकाराचा\s*झटका|हार्ट\s*अटॅक|हृदयात\s*वेदना)",
        r"\b(chha?ti\s*me(in)?\s*(bahut\s*)?(tej\s*)?dard|seene\s*me(in)?\s*(bahut\s*)?(tej\s*)?dard)\b",
    ],
    "breathing": [
        r"\b(can'?t\s*breathe|cannot\s*breathe|unable\s*to\s*breathe|gasping\s*for\s*air|choking)\b",
        r"\b(severe\s*(shortness\s*of\s*breath|breathing\s*difficulty|dyspnea|breathlessness))\b",
        r"\b(lips\s*(turning\s*|are\s*)?blue|cyanosis|struggling\s*to\s*breathe)\b",
        r"(सांस\s*(लेने\s*में\s*(बहुत\s*)?(तकलीफ|दिक्कत|परेशानी)|नहीं\s*आ\s*रही|फूल\s*रही|रुक\s*रही))",
        r"(दम\s*घुट\s*रहा)",
        r"(श्वास\s*(घेण्यास\s*(खूप\s*)?त्रास|कोंडतोय|लागत\s*नाही|घ्यायला\s*त्रास))",
        r"\b(saans\s*nahi\s*aa\s*rahi|saans\s*lene\s*me(in)?\s*takleef|shwas\s*ghyayla\s*tras)\b",
    ],
    "stroke": [
        r"\b(face\s*droop(ing)?|facial\s*droop|one\s*side\s*(of\s*face|weakness|numbness|paralysis))\b",
        r"\b(slurred\s*speech|sudden\s*confusion|trouble\s*speaking|inability\s*to\s*speak)\b",
        r"\b(paralysis|stroke\s*symptoms|sudden\s*loss\s*of\s*balance|sudden\s*blindness)\b",
        r"(लकवा|पक्षाघात|चेहरा\s*टेढ़ा|आवाज\s*लड़खड़ा|बोलने\s*में\s*असमर्थ)",
        r"(अर्धांगवायू|चेहरा\s*वाकडा|चेहरेचा\s*एक\s*भाग\s*वाकडा|बोलता\s*येत\s*नाही)",
        r"\b(lakwa\s*maar\s*gaya|chehra\s*tedha)\b",
    ],
    "severe_bleeding": [
        r"\b(severe\s*bleeding|uncontrolled\s*bleeding|heavy\s*bleeding|bleeding\s*won'?t\s*stop|hemorrhage)\b",
        r"\b(vomiting\s*(bright\s*red\s*)?blood|vomited\s*(a\s*large\s*amount\s*of\s*)?(bright\s*red\s*)?blood|coughing\s*up\s*blood|blood\s*spurting)\b",
        r"(खून\s*(बंद\s*नहीं\s*हो\s*रहा|की\s*उल्टी|बहुत\s*ज्यादा\s*बह\s*रहा)|तेज\s*रक्तस्राव)",
        r"(रक्तस्राव\s*(थांबत\s*नाही|बंद\s*नाही)|खूप\s*(जास्त\s*)?(रक्त|खून)\s*वाहत|रक्ताची\s*उलटी)",
        r"\b(khoon\s*band\s*nahi\s*ho\s*raha|khoon\s*ki\s*ulti)\b",
    ],
    "poisoning": [
        r"\b(swallowed\s*(poison|bleach|acid|pesticide|rat\s*poison|detergent|kerosene|phenyl))\b",
        r"\b(drank\s*(poison|bleach|acid|pesticide|rat\s*poison|detergent|kerosene|phenyl))\b",
        r"\b(overdose|took\s*too\s*many\s*(pills|tablets)|acute\s*poisoning|consumed\s*poison)\b",
        r"(जहर\s*(खा\s*लिया|पी\s*लिया)|कीटनाशक\s*पी\s*लिया|नींद\s*की\s*गोलियां\s*खा\s*ली)",
        r"(विष\s*प्राशन|अतिप्रमाणात\s*गोळ्या\s*घेतल्या|उंदीर\s*मारण्याचे\s*औषध)",
        r"\b(zeher\s*kha\s*liya|pesticide\s*pee\s*liya|dawa\s*pee\s*li\s*hai)\b",
    ],
    "snakebite": [
        r"\b(snake\s*bite|bitten\s*by\s*a?\s*snake|snake\s*envenomation|scorpion\s*sting)\b",
        r"(सांप\s*ने\s*(काटा|काट\s*लिया)|सर्पदंश|बिच्छू\s*ने\s*काटा)",
        r"(सापाने\s*चावा\s*घेतला|सर्पदंश|साप\s*चावला)",
        r"\b(saanp\s*ne\s*ka+ta\s*hai|saanp\s*ne\s*kaat\s*liya)\b",
    ],
    "unconscious_seizure": [
        r"\b(unconscious|fainted\s*and\s*not\s*waking|is\s*not\s*waking\s*up|not\s*responding|unresponsive|collapsed)\b",
        r"\b(having\s*a?\s*seizure|convulsions|fitting|status\s*epilepticus)\b",
        r"\b(lethargic\s*and\s*not\s*feeding)\b",
        r"(बेहोश|बेहोशी|होश\s*नहीं\s*आ\s*रहा|दौरा\s*पड़\s*रहा|मिर्गी\s*का\s*दौरा|चक्कर\s*आकर\s*गिर)",
        r"(बेशुद्ध|शुद्धीवर\s*नाही|आकडी\s*येणे|फिट्स\s*येत\s*आहेत)",
        r"\b(behosh\s*ho\s*gaya|daura\s*pad\s*raha\s*hai|hosh\s*nahi\s*aa\s*raha)\b",
    ],
    "anaphylaxis": [
        r"\b(anaphylaxis|anaphylactic\s*shock|throat\s*(closing|swelling)|swollen\s*tongue\s*and\s*lips)\b",
        r"\b(severe\s*allergic\s*reaction\s*with\s*breathing)\b",
        r"(गले\s*में\s*(अचानक\s*)?(भयानक\s*)?सूजन|जीभ\s*और\s*गले\s*में\s*सूजन)",
        r"(घशात\s*सूज\s*येऊन\s*श्वास\s*कोंडणे|तीव्र\s*ऍलर्जी)",
    ],
    "pregnancy_emergency": [
        r"\b(pregnant\s*(and|with)\s*(bleeding|heavy\s*vaginal\s*bleeding|severe\s*abdominal\s*(pain|bleeding)|water\s*broke\s*early|no\s*fetal\s*movement))\b",
        r"\b(eclampsia|postpartum\s*hemorrhage|baby\s*not\s*moving\s*in\s*womb)\b",
        r"(गर्भावस्था\s*में\s*(अचानक\s*)?(तेज\s*)?रक्तस्राव|गर्भवती\s*महिला\s*को\s*खून\s*आना)",
        r"(गरोदरपणात\s*रक्तस्राव|गर्भाची\s*हालचाल\s*थांबली)",
    ],
    "self_harm": [
        r"\b(want\s*to\s*die|kill\s*myself|suicidal|suicide\s*thoughts|end\s*my\s*life|ending\s*it\s*all|self\s*harm|overdose\s*to\s*die)\b",
        r"(आत्महत्या|खुदकुशी|मरना\s*चाहता\s*हूं|मरना\s*चाहती\s*हूं|जीने\s*की\s*इच्छा\s*नहीं|जीवन\s*समाप्त)",
        r"(आत्महत्या\s*करावीशी\s*वाटते|जगण्याची\s*इच्छा\s*नाही|जीवाचे\s*बरेवाईट)",
        r"\b(khudkushi|marna\s*chahta\s*hu|jeene\s*ki\s*iccha\s*nahi)\b",
    ],
}

EDUCATIONAL_SAFE_PATTERNS = [
    r"^\s*how\s+does\s+(dengue|malaria|diabetes|flu|virus|infection|disease)\b",
    r"^\s*what\s+is\s+(a\s+)?(heart\s+attack|stroke|diabetes|suicide\s+prevention|dengue|malaria|anaemia|asthma|hypertension)\b",
    r"^\s*what\s+are\s+the\s+(symptoms|causes|signs|risks|emergency\s+signs)\s+of\b",
    r"^\s*how\s+can\s+i\s+prevent\b",
    r"^\s*how\s+to\s+prevent\b",
    r"\bdog\s+bite\b",
    r"(के\s*लक्षण\s*क्या\s*होते\s*हैं)",
    r"(कसा\s*पसरतो)",
]

COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    cat: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for cat, pats in EMERGENCY_PATTERNS.items()
}
COMPILED_EDU = [re.compile(pat, re.IGNORECASE) for pat in EDUCATIONAL_SAFE_PATTERNS]

def detect(text: str) -> EmergencyMatch:
    if not text:
        return EmergencyMatch(is_emergency=False)
    for edu_pat in COMPILED_EDU:
        if edu_pat.search(text):
            return EmergencyMatch(is_emergency=False)
    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return EmergencyMatch(is_emergency=True, category=category, matched_pattern=pattern.pattern)
    return EmergencyMatch(is_emergency=False)
