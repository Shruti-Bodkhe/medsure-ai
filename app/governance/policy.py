from dataclasses import dataclass, field
from typing import List, Optional

from app.agents.claims import ClaimResult, VerificationReport
from app.agents.emergency import EmergencyResult, build_response, guess_lang
from app.agents.risk import RiskResult

ANSWER, REVISE, CLARIFY, ABSTAIN, ESCALATE = (
    "ANSWER", "REVISE", "CLARIFY", "ABSTAIN", "ESCALATE")

CAUTION = ("\n\nThis is general information only, not medical advice. "
           "Please talk to a doctor or pharmacist about your own situation.")
CLARIFY_MSG = ("To give you safe information I need a little more detail, for example "
               "the person's age, how long this has been going on, and any medicines "
               "being taken. Could you tell me more?")
ABSTAIN_MSG = ("I don't have reliable information in my trusted sources to answer this "
               "safely. Please ask a doctor or pharmacist.")
UNAVAILABLE_MSG = ("I can't check this safely right now. Please try again in a few "
                   "minutes, or ask a doctor or pharmacist.")


@dataclass
class PolicyConfig:
    # share of claims allowed to be unsupported before the answer is revised
    max_unsupported_frac: dict = field(
        default_factory=lambda: {"low": 0.25, "medium": 0.10, "high": 0.0})
    # supported claims needed to release a revised answer instead of abstaining
    min_supported_after_revise: dict = field(
        default_factory=lambda: {"low": 1, "medium": 1, "high": 2})


@dataclass
class Decision:
    action: str
    reason_code: str
    message: str
    kept_claims: List[str] = field(default_factory=list)
    dropped_claims: List[str] = field(default_factory=list)


def _caution(text: str, level: str) -> str:
    return text + CAUTION if level in ("medium", "high") else text


def early_decision(question: str, risk: RiskResult) -> Optional[Decision]:
    """Decisions that can be made before any evidence is retrieved."""
    if risk.level == "emergency":
        category = "self_harm" if risk.category == "mental_health" else risk.category
        text = build_response(EmergencyResult(True, category), guess_lang(question))
        return Decision(ESCALATE, "EMERGENCY_DETECTED", text)
    if risk.level == "unknown":
        return Decision(ABSTAIN, "RISK_UNKNOWN", UNAVAILABLE_MSG)
    if risk.needs_clarification and risk.level in ("medium", "high"):
        return Decision(CLARIFY, "MISSING_INFO", CLARIFY_MSG)
    return None


def _revised_text(kept: List[ClaimResult]) -> str:
    lines = []
    for c in kept:
        cite = f" [{', '.join(str(p) for p in c.passages)}]" if c.passages else ""
        lines.append(f"- {c.text}{cite}")
    return ("Here is what my trusted sources support:\n" + "\n".join(lines) +
            "\n\nSome details were left out because I could not confirm them "
            "in the sources.")


def final_decision(risk: RiskResult, report: VerificationReport, n_evidence: int,
                   draft: str, cfg: Optional[PolicyConfig] = None) -> Decision:
    cfg = cfg or PolicyConfig()
    level = risk.level

    if n_evidence == 0:
        return Decision(ABSTAIN, "NO_EVIDENCE", ABSTAIN_MSG)
    if report.error or not report.claims or any(
            c.label == "unverified" for c in report.claims):
        return Decision(ABSTAIN, "VERIFICATION_FAILED", UNAVAILABLE_MSG)

    supported = [c for c in report.claims if c.label == "supported"]
    contradicted = [c for c in report.claims if c.label == "contradicted"]
    unsupported = [c for c in report.claims if c.label != "supported"]
    frac = len(unsupported) / len(report.claims)
    kept = [c.text for c in supported]
    dropped = [c.text for c in unsupported]

    if not contradicted and frac <= cfg.max_unsupported_frac[level]:
        code = "ALL_CLAIMS_SUPPORTED" if not unsupported else "MOSTLY_SUPPORTED"
        return Decision(ANSWER, code, _caution(draft, level), kept, dropped)

    if len(supported) < cfg.min_supported_after_revise[level]:
        return Decision(ABSTAIN, "INSUFFICIENT_EVIDENCE", ABSTAIN_MSG, kept, dropped)

    code = "CONTRADICTION_REMOVED" if contradicted else "UNSUPPORTED_REMOVED"
    return Decision(REVISE, code, _caution(_revised_text(supported), level),
                    kept, dropped)