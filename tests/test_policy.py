from app.agents.claims import ClaimResult, VerificationReport
from app.agents.risk import RiskResult
from app.governance.policy import (ABSTAIN, ANSWER, CLARIFY, ESCALATE, REVISE,
                                   early_decision, final_decision)


def risk(level, clarify=False, category="general_info"):
    return RiskResult(level, category, clarify, "test", "llm")


def report(*labels):
    return VerificationReport(
        [ClaimResult(f"claim {i}", lab, [1]) for i, lab in enumerate(labels, 1)])


def test_emergency_escalates():
    d = early_decision("chest pain", risk("emergency", category="cardiac"))
    assert d.action == ESCALATE and "112" in d.message


def test_unknown_risk_abstains():
    assert early_decision("q", risk("unknown")).action == ABSTAIN


def test_missing_info_clarifies_for_medium_and_high_only():
    assert early_decision("q", risk("medium", clarify=True)).action == CLARIFY
    assert early_decision("q", risk("high", clarify=True)).action == CLARIFY
    assert early_decision("q", risk("low", clarify=True)) is None


def test_no_early_decision_for_plain_low():
    assert early_decision("q", risk("low")) is None


def test_all_supported_low_answers():
    d = final_decision(risk("low"), report("supported", "supported"), 5, "draft")
    assert d.action == ANSWER and d.message == "draft"


def test_caution_added_for_high_risk():
    d = final_decision(risk("high"), report("supported", "supported"), 5, "draft")
    assert d.action == ANSWER and "not medical advice" in d.message


def test_high_risk_one_unsupported_revises():
    d = final_decision(risk("high"),
                       report("supported", "supported", "supported", "not_found"), 5, "draft")
    assert d.action == REVISE and len(d.kept_claims) == 3
    assert "claim 4" not in d.message


def test_contradiction_is_removed():
    d = final_decision(risk("low"),
                       report("supported", "supported", "contradicted"), 5, "draft")
    assert d.action == REVISE and d.reason_code == "CONTRADICTION_REMOVED"


def test_too_little_support_abstains():
    d = final_decision(risk("high"), report("supported", "contradicted", "not_found"), 5, "d")
    assert d.action == ABSTAIN


def test_nothing_supported_abstains():
    d = final_decision(risk("low"), report("not_found", "not_found"), 5, "draft")
    assert d.action == ABSTAIN


def test_no_evidence_abstains():
    assert final_decision(risk("low"), report("supported"), 0, "d").action == ABSTAIN


def test_failed_verification_abstains():
    failed = VerificationReport([], error="boom")
    assert final_decision(risk("low"), failed, 5, "d").action == ABSTAIN
    unchecked = report("unverified", "supported")
    assert final_decision(risk("low"), unchecked, 5, "d").action == ABSTAIN
def test_out_of_scope_abstains_with_own_reason():
    d = early_decision("weather today", risk("low", category="out_of_scope"))
    assert d.action == ABSTAIN and d.reason_code == "OUT_OF_SCOPE"


def test_specific_clarifying_question_is_used():
    r = RiskResult("high", "medicine", True, "t", "llm",
                   clarifying_question="Which medicine do you take?")
    d = early_decision("q", r)
    assert d.action == CLARIFY
    assert "Which medicine do you take?" in d.message and "112" in d.message
