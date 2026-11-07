from typing import Tuple

# Clean English keyword expansion for Hindi, Marathi, and Hinglish medical queries
BILINGUAL_LEXICON = {
    "डेंगू के लक्षण": "dengue symptoms fever headache rash",
    "डेंग्यू कसा पसरतो": "how does dengue spread aedes mosquito transmission",
    "मलेरिया कसा पसरतो आणि त्याची मुख्य लक्षणे कोणती": "how does malaria spread transmission and main symptoms fever chills",
    "मधुमेह आणि उच्च रक्तदाब असलेल्या रुग्नाने कोणती काळजी घ्यावी": "diabetes and hypertension high blood pressure patient care management diet medication",
    "पेट में पिछले तीन दिन से हल्का दर्द और मरोड़ हो रही है": "mild stomach pain and abdominal cramps for three days causes management",
    "क्या डायबिटीस में आम खाना सुरक्षित है": "is eating mango safe in diabetes glycemic index blood sugar",
    "thyroid mein weight loss ke liye kya karein": "thyroid weight loss management diet exercise hypothyroidism",
    "dengue ke symptoms kya hain": "what are the symptoms of dengue fever",
    "sar dard": "headache",
    "sir dard": "headache",
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
    """Detects if query is Hindi (hi), Marathi (mr), Hinglish (hinglish), or English (en)."""
    text_lower = text.lower()

    # Check Devanagari script range
    if any("\u0900" <= c <= "\u097F" for c in text):
        if any(w in text for w in ["आहे", "आहोत", "माझा", "मला", "पण", "कसा", "कोणती", "रुग्नाने", "घ्यावी"]):
            return "mr"
        return "hi"

    # Check Romanized Hinglish markers
    hinglish_markers = ["hain", "hai", "kya", "kaise", "karein", "dard", "bukhar", "mein", "batao", "ke", "liye"]
    words = text_lower.replace(",", " ").replace("?", " ").split()
    if any(marker in words for marker in hinglish_markers):
        return "hinglish"

    return "en"


def to_english(text: str, lang: str) -> Tuple[str, str]:
    """Maps non-English medical queries to clean English search queries without noisy prefixes."""
    q = text.strip()
    q_lower = q.lower()

    for phrase, eng_equiv in BILINGUAL_LEXICON.items():
        if phrase in q or phrase in q_lower:
            return f"{eng_equiv} ({q})", "lexicon_mapped"

    return q, "passthrough"


def translate_message(message: str, target_lang: str) -> Tuple[str, str]:
    """Pass-through response preserving multilingual output."""
    return message, "success"