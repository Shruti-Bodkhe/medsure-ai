from dataclasses import dataclass
from typing import List

# Action Constants
ANSWER = "ANSWER"
REVISE = "REVISE"
CLARIFY = "CLARIFY"
ABSTAIN = "ABSTAIN"
ESCALATE = "ESCALATE"

UNAVAILABLE_MSG = "Medbase clinical search is currently unavailable. Please consult a qualified healthcare provider."
ABSTAIN_MSG = "I cannot provide a definitive recommendation based on the current clinical literature. Please consult a qualified physician for personalized guidance."
CAUTION = "\n\n*Note: This information is for educational purposes and does not replace professional medical diagnosis.*"

@dataclass
class Decision:
    action: str
    reason_code: str
    message: str
    kept_claims: List[str] = None

@dataclass
class PolicyConfig:
    max_unsupported_low: float = 0.35
    max_unsupported_med: float = 0.20
    max_unsupported_high: float = 0.05
    min_supported_low: int = 1
    min_supported_med: int = 1
    min_supported_high: int = 1

def early_decision(question: str, risk_res) -> Decision:
    """Evaluates early pre-retrieval triage decisions."""
    if risk_res.level == "emergency" or risk_res.category in ("emergency", "self_harm"):
        return Decision(
            action=ESCALATE,
            reason_code="EMERGENCY_DETECTED",
            message=(
                "🚨 **IMMEDIATE EMERGENCY ACTION REQUIRED**\n\n"
                "The symptoms or situation you described indicate a potential medical emergency. "
                "Please call **112** immediately or proceed to the nearest hospital emergency department.\n\n"
                "If you are experiencing emotional distress or thoughts of self-harm, please call the "
                "National Tele-Mental Health Helpline (**14416**) available 24/7."
            )
        )
    
    if risk_res.needs_clarification:
        return Decision(
            action=CLARIFY,
            reason_code="MISSING_INFO",
            message=risk_res.clarifying_question or "Could you please provide more details about your symptoms and duration?"
        )
        
    if risk_res.category == "out_of_scope":
        return Decision(
            action=ABSTAIN,
            reason_code="OUT_OF_SCOPE",
            message="This query is outside the scope of clinical guidance. Please consult an appropriate professional."
        )
        
    return None

def check_responsive(decision: Decision, risk_level: str, verdict: str, clarifying_q: str = "") -> Decision:
    """Ensures response responsiveness based on judge verdict."""
    if verdict == "no":
        if risk_level in ("high", "emergency"):
            return Decision(
                action=ABSTAIN,
                reason_code="NOT_RESPONSIVE",
                message=ABSTAIN_MSG
            )
        return Decision(
            action=CLARIFY,
            reason_code="NEEDS_DETAIL",
            message=clarifying_q or "Could you please clarify your specific symptoms?"
        )
    return decision

def final_decision(risk, report, num_passages: int, draft: str, cfg: PolicyConfig) -> Decision:
    """Computes final governance action based on claim verification and risk strata."""
    if num_passages == 0:
        return Decision(action=ABSTAIN, reason_code="NO_RELEVANT_EVIDENCE", message=ABSTAIN_MSG)

    claims = report.claims
    if not claims:
        return Decision(action=ANSWER, reason_code="NO_CLAIMS_EXTRACTED", message=draft + CAUTION)

    contra = [c for c in claims if c.verdict == "contradicted"]
    unsup = [c for c in claims if c.verdict in ("not_found", "unverified")]
    sup = [c for c in claims if c.verdict == "supported"]

    gamma = len(unsup) / len(claims) if claims else 0.0

    # Map risk to policy thresholds
    if risk.level == "high":
        max_unsup = cfg.max_unsupported_high
        min_sup = cfg.min_supported_high
    elif risk.level == "medium":
        max_unsup = cfg.max_unsupported_med
        min_sup = cfg.min_supported_med
    else:
        max_unsup = cfg.max_unsupported_low
        min_sup = cfg.min_supported_low

    # Decision Logic
    if len(contra) == 0 and gamma <= max_unsup and len(sup) >= min_sup:
        return Decision(action=ANSWER, reason_code="ALL_CLAIMS_SUPPORTED", message=draft + CAUTION)

    if len(contra) == 0 and len(sup) >= min_sup:
        # Revise and filter out unsupported claims
        kept = [c.claim for c in sup]
        revised_msg = "\n".join([f"• {c}" for c in kept]) + CAUTION
        return Decision(action=REVISE, reason_code="REVISED_UNSUPPORTED_CLAIMS", message=revised_msg, kept_claims=kept)

    if risk.level in ("high", "emergency") or len(contra) > 0:
        return Decision(action=ABSTAIN, reason_code="UNSUPPORTED_OR_CONTRADICTED", message=ABSTAIN_MSG)

    return Decision(action=CLARIFY, reason_code="INSUFFICIENT_EVIDENCE_CLARIFY", message="Could you provide additional details regarding your symptoms?")