from dataclasses import dataclass, field
from typing import List, Optional
from app.agents.claims import VerificationReport
from app.agents.risk import RiskResult

ANSWER = "ANSWER"
REVISE = "REVISE"
CLARIFY = "CLARIFY"
ABSTAIN = "ABSTAIN"
ESCALATE = "ESCALATE"

VALID_RISK_LEVELS = {"low", "medium", "high", "emergency"}

@dataclass
class PolicyConfig:
    tau_low: float = 0.35
    tau_medium: float = 0.20
    tau_high: float = 0.05
    min_support_low: float = 0.50
    min_support_medium: float = 0.65
    min_support_high: float = 0.75

@dataclass
class Decision:
    action: str
    reason_code: str
    message: str
    kept_claims: Optional[List[str]] = field(default_factory=list)
    dropped_claims: Optional[List[str]] = field(default_factory=list)

EMERGENCY_MESSAGES = {
    "self_harm": (
        "?? **IMMEDIATE CRISIS SUPPORT REQUIRED**\n\n"
        "You are not alone, and immediate confidential help is available right now.\n"
        "- **National Tele-Mental Health Helpline (India):** Call **14416** or **1800-891-4416** (24/7, Toll-Free)\n"
        "- **Emergency Medical Services:** Call **112** or **108**\n"
        "Please reach out to a trusted family member, friend, or healthcare professional immediately."
    ),
    "default": (
        "?? **MEDICAL EMERGENCY DETECTED ? SEEK IMMEDIATE CARE**\n\n"
        "The symptoms described indicate a potentially life-threatening medical emergency.\n"
        "- **Call Emergency Services Immediately:** Dial **112** or **108** (National Ambulance Service)\n"
        "- **Do not wait** for symptoms to improve on their own and do not drive yourself to the hospital.\n"
        "- Keep the patient calm, seated or lying in a safe position, and monitor breathing until paramedics arrive."
    ),
}

OUT_OF_SCOPE_MESSAGE = (
    "I am MedSure AI, a specialized clinical governance and healthcare triage assistant. "
    "I can only assist with medical, symptom, medication safety, and wellness inquiries."
)

ABSTAIN_MESSAGE = (
    "I cannot provide a definitive recommendation based on the current clinical literature. "
    "Please consult a qualified physician for personalized guidance."
)

HIGH_RISK_CAUTION_PREFIX = (
    "**Important Clinical Safety Caution:** This information is for educational purposes and is not medical advice. "
    "Because this involves a high-risk clinical or medication scenario, please verify any treatment changes directly with a qualified doctor.\n\n"
)

def early_decision(question: str, risk: RiskResult) -> Optional[Decision]:
    if risk.category == "out_of_scope":
        return Decision(ABSTAIN, "OUT_OF_SCOPE", OUT_OF_SCOPE_MESSAGE)

    if risk.level not in VALID_RISK_LEVELS:
        return Decision(ABSTAIN, "UNKNOWN_RISK", ABSTAIN_MESSAGE)

    if risk.is_emergency or risk.level == "emergency":
        msg = EMERGENCY_MESSAGES["self_harm"] if risk.category == "self_harm" else EMERGENCY_MESSAGES["default"]
        return Decision(ESCALATE, f"EMERGENCY_{str(risk.category or 'GENERAL').upper()}", msg)

    if risk.needs_clarification and risk.level in ("medium", "high"):
        cq = risk.clarifying_question or "Could you share your age, symptom duration, and any current medications?"
        msg = f"{cq} (If you are experiencing severe or sudden symptoms, call 112 immediately.)"
        return Decision(CLARIFY, "MISSING_INFO", msg)

    return None

def final_decision(
    risk: RiskResult,
    report: VerificationReport,
    passage_count: int,
    draft_answer: str,
    cfg: PolicyConfig = PolicyConfig(),
) -> Decision:
    if passage_count <= 0:
        return Decision(ABSTAIN, "NO_EVIDENCE_RETRIEVED", ABSTAIN_MESSAGE)

    if not draft_answer or draft_answer.strip().upper().startswith("NO_ANSWER"):
        return Decision(ABSTAIN, "NO_RELEVANT_EVIDENCE", ABSTAIN_MESSAGE)

    if report is None or report.error or not report.claims:
        return Decision(ABSTAIN, "VERIFICATION_FAILED", ABSTAIN_MESSAGE)

    if any(c.verdict == "unverified" for c in report.claims):
        return Decision(ABSTAIN, "VERIFICATION_FAILED", ABSTAIN_MESSAGE)

    supported = [c.text for c in report.claims if c.verdict == "supported"]
    contradicted = [c.text for c in report.claims if c.verdict == "contradicted"]
    unsupported = [c.text for c in report.claims if c.verdict != "supported"]

    if not supported:
        return Decision(ABSTAIN, "ZERO_SUPPORTED_CLAIMS", ABSTAIN_MESSAGE, [], unsupported)

    if risk.level == "high" and len(supported) < 2:
        return Decision(ABSTAIN, "INSUFFICIENT_EVIDENCE", ABSTAIN_MESSAGE, supported, unsupported)

    total = len(report.claims)
    sup_ratio = len(supported) / total

    min_req = {
        "low": cfg.min_support_low,
        "medium": cfg.min_support_medium,
        "high": cfg.min_support_high,
    }.get(risk.level, cfg.min_support_low)

    if sup_ratio < min_req:
        return Decision(ABSTAIN, "LOW_CLAIM_SUPPORT", ABSTAIN_MESSAGE, supported, unsupported)

    if contradicted or unsupported:
        revised_body = ". ".join(supported).strip()
        if not revised_body.endswith("."):
            revised_body += "."
        if risk.level == "high":
            revised_body = HIGH_RISK_CAUTION_PREFIX + revised_body
        reason_code = "CONTRADICTION_REMOVED" if contradicted else "UNSUPPORTED_CLAIMS_REMOVED"
        return Decision(REVISE, reason_code, revised_body, supported, unsupported)

    final_msg = draft_answer
    if risk.level == "high" and "not medical advice" not in final_msg:
        final_msg = HIGH_RISK_CAUTION_PREFIX + final_msg
    return Decision(ANSWER, "ALL_CLAIMS_SUPPORTED", final_msg, supported, [])

def check_responsive(
    decision: Decision,
    risk_level: str,
    verdict: str,
    specific_question: Optional[str] = None,
) -> Decision:
    if decision.action not in (ANSWER, REVISE):
        return decision

    if verdict == "yes":
        return decision

    if verdict in ("partly", "error") and risk_level == "low":
        return decision

    if risk_level == "high":
        return Decision(ABSTAIN, "NOT_RESPONSIVE", ABSTAIN_MESSAGE)

    if risk_level == "medium":
        if verdict == "error":
            return Decision(ABSTAIN, "RESPONSIVENESS_ERROR", ABSTAIN_MESSAGE)
        q = specific_question or "Could you provide more details about your symptoms and how long you have had them?"
        return Decision(CLARIFY, "NOT_RESPONSIVE", q)

    return Decision(
        CLARIFY,
        "NOT_RESPONSIVE",
        specific_question or "Could you clarify your question so I can provide accurate clinical information?",
    )

ABSTAIN_MSG = ABSTAIN_MESSAGE
OUT_OF_SCOPE_MSG = OUT_OF_SCOPE_MESSAGE
HIGH_RISK_PREFIX = HIGH_RISK_CAUTION_PREFIX
UNAVAILABLE_MSG = ABSTAIN_MESSAGE
CAUTION = HIGH_RISK_CAUTION_PREFIX
CAUTION_PREFIX = HIGH_RISK_CAUTION_PREFIX
CLARIFY_MSG = "Could you share your age, symptom duration, and any current medications?"
EMERGENCY_MSG = EMERGENCY_MESSAGES["default"]
