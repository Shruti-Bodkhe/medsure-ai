import json
import re
from typing import Tuple
from app.core import llm

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

BILINGUAL_LEXICON = {
    "डेंगू के लक्षण": "dengue symptoms fever headache rash",
    "डेंग्यू कसा पसरतो": "how does dengue spread aedes mosquito transmission",
    "मलेरिया कसा पसरतो आणि त्याची मुख्य लक्षणे कोणती": "how does malaria spread transmission and main symptoms fever chills",
    "मधुमेह आणि उच्च रक्तदाब असलेल्या रुग्नाने कोणती काळजी घ्यावी": "diabetes and hypertension high blood pressure patient care management diet medication",
    "पेट में पिछले तीन दिन से हल्का दर्द और मरोड़ हो रही है": "mild stomach pain and abdominal cramps for three days causes management",
    "क्या डायबिटीस में आम खाना सुरक्षित है": "is eating mango safe in diabetes glycemic index blood sugar",
    "thyroid mein weight loss ke liye kya karein": "thyroid weight loss management diet exercise hypothyroidism",
    "dengue ke symptoms kya hain": "what are the symptoms of dengue fever",
    "mala shwas ghyayla tras hotoy": "shortness of breath difficulty breathing",
    "sar dard": "headache",
    "sir dard": "headache",
    "डोकेदुखी": "headache migraine",
    "डोके दुखत": "headache",
    "pet dard": "stomach pain",
    "pet mein dard": "stomach pain",
    "chhaati mein dard": "chest pain",
    "saans nahi aa rahi": "shortness of breath",
    "chakkar aa rahe hain": "dizziness",
    "bukhar": "fever",
    "khansi": "cough",
    "zukam": "cold",
    "ultai": "vomiting",
    "ultiyan": "vomiting",
    "kamzori": "weakness",
    "raktrav": "bleeding",
    "madhumeh": "diabetes",
    "uchch rakt-chap": "hypertension",
    "raktchap": "blood pressure",
}

def detect_language(text: str) -> str:
    text_lower = text.lower()
    if any("\u0900" <= c <= "\u097F" for c in text):
        if any(w in text for w in ["आहे", "आहोत", "माझा", "मला", "पण", "कसा", "कोणती", "रुग्नाने", "घ्यावी", "दुखतंय", "होतोय"]):
            return "mr"
        return "hi"

    marathi_roman = ["mala", "majha", "shwas", "ghyayla", "tras", "hotoy", "dukhate", "aahe"]
    words = text_lower.replace(",", " ").replace("?", " ").split()
    if any(m in words for m in marathi_roman):
        return "mr"

    hinglish_markers = ["hain", "hai", "kya", "kaise", "karein", "dard", "bukhar", "mein", "batao", "ke", "liye"]
    if any(marker in words for marker in hinglish_markers):
        return "hinglish"

    return "en"

def numbers_match(source: str, translated: str) -> bool:
    s_norm = source.translate(DEVANAGARI_DIGITS)
    t_norm = translated.translate(DEVANAGARI_DIGITS)
    s_nums = sorted(re.findall(r"\d+(?:\.\d+)?", s_norm))
    t_nums = sorted(re.findall(r"\d+(?:\.\d+)?", t_norm))
    return s_nums == t_nums

def to_english(text: str, lang: str) -> Tuple[str, bool]:
    q = text.strip()
    if lang == "en":
        return q, True
    q_lower = q.lower()
    for phrase, eng_equiv in BILINGUAL_LEXICON.items():
        if phrase in q or phrase in q_lower:
            return f"{eng_equiv} ({q})", True
    try:
        reply = llm.generate(f"Translate this medical question to English: {q}", temperature=0.0, max_tokens=200)
        return (reply.strip() or q), True
    except Exception:
        return q, False

def translate_message(message: str, target_lang: str) -> Tuple[str, str]:
    if target_lang in ("en", "hinglish") or any("\u0900" <= c <= "\u097F" for c in message):
        return message, "passthrough"
    lang_name = "Marathi" if target_lang == "mr" else "Hindi"
    sys_prompt = (
        f"Translate the clinical message into {lang_name}. Keep all numbers, dosages, "
        'and citation brackets like [1] unchanged. Reply ONLY with JSON: {"translation": "..."}'
    )
    try:
        raw = llm.generate(message, system=sys_prompt, temperature=0.0, max_tokens=1200)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            trans = str(data.get("translation", "")).strip()
            if trans:
                if not numbers_match(message, trans):
                    return message, "number_mismatch"
                return trans, "translated"
        if not numbers_match(message, raw):
            return message, "number_mismatch"
        return raw.strip(), "translated"
    except Exception:
        return message, "failed"
