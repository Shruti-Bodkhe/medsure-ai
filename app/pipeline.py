import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.agents.claims import VerificationReport, check_answer
from app.agents.language import detect_language, to_english, translate_message
from app.agents.responsive import check_responsiveness
from app.agents.risk import RiskResult, assess_risk
from app.baselines import clinical_fallback_synthesis, rag_only
from app.engine.pharmacovigilance import check_pharmacovigilance_safety
from app.governance.policy import (
    ABSTAIN,
    ABSTAIN_MSG,
    ANSWER,
    CAUTION_PREFIX,
    CLARIFY,
    Decision,
    ESCALATE,
    PolicyConfig,
    REVISE,
    check_responsive,
    early_decision,
    final_decision,
)

AUDIT_PATH = Path("data/audit/audit.jsonl")
_PIPELINE_CACHE: Dict[str, "PipelineResult"] = {}


def redact_pii(text: str) -> str:
    """Redacts emails, phone numbers, and 12-digit national ID patterns before audit logging."""
    if not text:
        return ""
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b", "[PHONE_REDACTED]", text)
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[ID_REDACTED]", text)
    return text


@dataclass
class PipelineResult:
    question: str
    language: str
    english_query: str
    translated: bool
    risk: RiskResult
    rag_answer: str
    sources: List[dict]
    report: Optional[VerificationReport]
    decision: Decision
    translation_status: str = "passthrough"
    latency_ms: float = 0.0
    pharmacovigilance_warnings: List[str] = field(default_factory=list)


def _audit(result: PipelineResult) -> None:
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(result)
        row["question"] = redact_pii(row.get("question", ""))
        row["english_query"] = redact_pii(row.get("english_query", ""))
        row["timestamp"] = time.time()
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run(
    question: str,
    retriever=None,
    cfg: PolicyConfig = PolicyConfig(),
    k: int = 5,
    patient_context: str = "",
    allow_clinical_fallback: bool = True,
    user_allergies: str = "",
    user_conditions: str = "",
) -> PipelineResult:
    t0 = time.time()
    q_clean = (question or "").strip()

    # Use cache only when real retriever is provided and no custom monkeypatching is active
    cache_key = f"{q_clean}||{patient_context}||{user_allergies}||{user_conditions}"
    if retriever is not None and cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    # 1. Language detection & normalization
    lang = detect_language(q_clean)
    eng_q, was_trans = to_english(q_clean, lang)

    # 2. Pre-retrieval Risk & Emergency Assessment
    risk = assess_risk(q_clean)
    if risk.level == "low" and lang != "en":
        eng_risk = assess_risk(eng_q)
        if eng_risk.level in ("emergency", "high", "medium") or eng_risk.category == "out_of_scope":
            risk = eng_risk

    # 3. Deterministic Pharmacovigilance Engine check against Patient Vault
    pv_warnings = check_pharmacovigilance_safety(q_clean, user_allergies, user_conditions)
    if pv_warnings and risk.level in ("low", "medium"):
        risk = RiskResult("high", "pharmacovigilance", False, "Patient Vault allergy/condition contraindication", "rules")

    # 4. Early Governance Short-Circuit (ESCALATE / OUT_OF_SCOPE / UNKNOWN_RISK / CLARIFY)
    early = early_decision(q_clean, risk)
    if early is not None:
        msg, t_status = translate_message(early.message, lang)
        early.message = msg
        res = PipelineResult(
            question=q_clean,
            language=lang,
            english_query=eng_q,
            translated=was_trans,
            risk=risk,
            rag_answer="",
            sources=[],
            report=None,
            decision=early,
            translation_status=t_status,
            latency_ms=round((time.time() - t0) * 1000, 2),
            pharmacovigilance_warnings=pv_warnings,
        )
        _audit(res)
        if retriever is not None:
            _PIPELINE_CACHE[cache_key] = res
        return res

    # 5. High-risk Pharmacovigilance Caution Synthesis (only when running live with a retriever)
    if retriever is not None and risk.category == "pharmacovigilance":
        synth_ans = clinical_fallback_synthesis(q_clean, lang, patient_context, is_caution=True)
        if pv_warnings:
            synth_ans = "\n".join(pv_warnings) + "\n\n" + synth_ans
        hits = retriever.hybrid_search(eng_q, k=2)
        sources = [
            {"n": i, "title": h["title"], "publisher": h["publisher"], "url": h["url"], "page": h["page"], "chunk_id": h["chunk_id"]}
            for i, h in enumerate(hits, start=1)
        ]
        dec = Decision(REVISE, "PHARMACOVIGILANCE_CAUTION_SYNTHESIS", synth_ans, [synth_ans], [])
        res = PipelineResult(
            question=q_clean,
            language=lang,
            english_query=eng_q,
            translated=was_trans,
            risk=risk,
            rag_answer=synth_ans,
            sources=sources,
            report=None,
            decision=dec,
            translation_status="native",
            latency_ms=round((time.time() - t0) * 1000, 2),
            pharmacovigilance_warnings=pv_warnings,
        )
        _audit(res)
        _PIPELINE_CACHE[cache_key] = res
        return res

    # 6. Tier-1 Governed Hybrid RAG
    rag = rag_only(eng_q, retriever, k=k, patient_context=patient_context)
    draft = str(rag.get("answer", "")).strip()
    passages = rag.get("passages", [])
    sources = rag.get("sources", [])

    # 7. Handle NO_ANSWER from Tier-1 RAG
    if not draft or draft.upper().startswith("NO_ANSWER"):
        if not allow_clinical_fallback or retriever is None:
            dec = Decision(ABSTAIN, "NO_RELEVANT_EVIDENCE", ABSTAIN_MSG)
            res = PipelineResult(
                question=q_clean,
                language=lang,
                english_query=eng_q,
                translated=was_trans,
                risk=risk,
                rag_answer=draft,
                sources=[],
                report=None,
                decision=dec,
                translation_status="passthrough",
                latency_ms=round((time.time() - t0) * 1000, 2),
                pharmacovigilance_warnings=pv_warnings,
            )
            _audit(res)
            return res

        # Tier-2 Evidence-Guided Clinical Synthesis Fallback
        is_caution = risk.level == "high"
        synth_ans = clinical_fallback_synthesis(q_clean, lang, patient_context, is_caution=is_caution)
        if pv_warnings:
            synth_ans = "\n".join(pv_warnings) + "\n\n" + synth_ans
        action = REVISE if is_caution else ANSWER
        reason_code = "CLINICAL_CAUTION_SYNTHESIS" if is_caution else "CLINICAL_SYNTHESIS_FALLBACK"
        dec = Decision(action, reason_code, synth_ans, [synth_ans], [])
        res = PipelineResult(
            question=q_clean,
            language=lang,
            english_query=eng_q,
            translated=was_trans,
            risk=risk,
            rag_answer=synth_ans,
            sources=[],
            report=None,
            decision=dec,
            translation_status="native",
            latency_ms=round((time.time() - t0) * 1000, 2),
            pharmacovigilance_warnings=pv_warnings,
        )
        _audit(res)
        _PIPELINE_CACHE[cache_key] = res
        return res

    # 8. Post-Generation Claim & Numerical Verification
    report = check_answer(draft, passages)
    dec = final_decision(risk, report, len(passages), draft, cfg=cfg)

    # 9. Responsiveness Check
    if dec.action in (ANSWER, REVISE) and retriever is not None:
        resp = check_responsiveness(q_clean, dec.message)
        dec = check_responsive(dec, risk.level, resp.verdict, resp.clarifying_question)

    # If Tier-1 abstained (e.g., due to HTTP 429 or niche phrasing) and fallback is enabled, use Tier-2 synthesis
    if dec.action == ABSTAIN and allow_clinical_fallback and retriever is not None and risk.category != "out_of_scope":
        is_caut = (risk.level == "high")
        synth_ans = clinical_fallback_synthesis(q_clean, lang, patient_context, is_caution=is_caut)
        act = REVISE if is_caut else ANSWER
        rc = "CLINICAL_CAUTION_SYNTHESIS" if is_caut else "CLINICAL_SYNTHESIS_FALLBACK"
        dec = Decision(act, rc, synth_ans, [synth_ans], [])

    if pv_warnings and dec.action in (ANSWER, REVISE):
        dec.message = "\n".join(pv_warnings) + "\n\n" + dec.message

    msg, t_status = translate_message(dec.message, lang)
    dec.message = msg

    res = PipelineResult(
        question=q_clean,
        language=lang,
        english_query=eng_q,
        translated=was_trans,
        risk=risk,
        rag_answer=draft,
        sources=sources,
        report=report,
        decision=dec,
        translation_status=t_status,
        latency_ms=round((time.time() - t0) * 1000, 2),
        pharmacovigilance_warnings=pv_warnings,
    )
    _audit(res)
    if retriever is not None:
        _PIPELINE_CACHE[cache_key] = res
    return res
