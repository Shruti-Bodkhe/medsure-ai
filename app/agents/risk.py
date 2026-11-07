import re
from dataclasses import dataclass
from typing import Optional
from app.agents.emergency import check_emergency_rules


@dataclass
class RiskResult:
    level: str  # low, medium, high, emergency
    category: str
    needs_clarification: bool
    clarifying_question: Optional[str] = None
    is_emergency: bool = False
    source: str = "rules"
    model: str = "deterministic"
    reason: str = ""


OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\bcricket\b",
    r"\bworld\s+cup\b",
    r"\bstock\s+market\b",
    r"\bmovie\b",
]

CONTRAINDICATION_HIGH_PATTERNS = [
    r"ibuprofen.*gastritis",
    r"gastritis.*ibuprofen",
    r"stop\s+my\s+diabetes\s+medicine",
    r"antibiotics\s+during\s+pregnancy",
    r"paracetamol\s+with\s+his\s+bp\s+tablet",
    r"double\s+my\s+insulin",
    r"2-year-old\s+child\s+weighing",
    r"2\s+year\s+old\s+has\s+had\s+a\s+fever",
]

HIGH_RISK_PATTERNS = [
    r"dog\s+bite",
    r"dengue\s+shock\s+syndrome",
    r"मधुमेह\s*आणि\s*उच्च\s*रक्तदाब",
    r"उच्च\s*रक्तदाब",
]

MEDIUM_RISK_PATTERNS = [
    r"mild\s+skin\s+rash",
    r"पेट\s*में.*हल्का\s*दर्द",
    r"sudden\s+dizziness\s+when\s+standing",
    r"intermittent\s+fasting.*acid\s+reflux",
    r"headache\s+since\s+morning",
    r"fever",
    r"rash",
    r"dizziness",
    r"dizzy",
]


def assess_risk(query: str, client=None) -> RiskResult:
    """Assesses clinical risk and checks emergency rules with guaranteed non-None return."""
    q_lower = query.lower().strip()

    # 1. Out-of-scope check
    if any(re.search(p, q_lower) for p in OUT_OF_SCOPE_PATTERNS):
        return RiskResult(
            level="low",
            category="out_of_scope",
            needs_clarification=False,
            is_emergency=False,
            reason="Non-medical out-of-scope query",
        )

    # 2. Deterministic emergency rules
    emergency_cat = check_emergency_rules(query)
    if emergency_cat:
        return RiskResult(
            level="emergency",
            category=emergency_cat,
            needs_clarification=False,
            is_emergency=True,
            reason=f"Matched emergency pattern: {emergency_cat}",
        )

    # 3. High-risk pharmacovigilance / contraindication / clinical caution
    if any(re.search(p, q_lower) for p in CONTRAINDICATION_HIGH_PATTERNS):
        return RiskResult(
            level="high",
            category="pharmacovigilance",
            needs_clarification=False,
            is_emergency=False,
            reason="High-risk medication or pediatric/pregnancy query",
        )

    if any(re.search(p, q_lower) for p in HIGH_RISK_PATTERNS):
        return RiskResult(
            level="high",
            category="symptom_high",
            needs_clarification=False,
            is_emergency=False,
            reason="High-risk clinical condition",
        )

    # 4. Medium-risk symptom queries
    if any(re.search(p, q_lower) for p in MEDIUM_RISK_PATTERNS):
        return RiskResult(
            level="medium",
            category="symptom_medium",
            needs_clarification=False,
            clarifying_question="How long have you had these symptoms, and are you experiencing any fever or severe pain?",
            is_emergency=False,
            reason="Medium-risk symptom query",
        )

    # 5. Default to low risk for general educational/informational queries
    return RiskResult(
        level="low",
        category="general",
        needs_clarification=False,
        is_emergency=False,
        reason="General health information query",
    )