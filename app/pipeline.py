import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.agents.claims import VerificationReport, check_answer
from app.agents.language import detect_language, to_english, translate_message
from app.agents.responsive import judge
from app.agents.risk import RiskResult, assess_risk
from app.baselines import rag_only, clinical_fallback_synthesis, generate_vault_correlation
from app.governance.policy import (
    ABSTAIN, ABSTAIN_MSG, ANSWER, CLARIFY, ESCALATE, REVISE,
    UNAVAILABLE_MSG, CAUTION, Decision, PolicyConfig,
    check_responsive, early_decision, final_decision
)

AUDIT = Path("data/audit/audit.jsonl")
_PIPELINE_CACHE = {}


@dataclass
class PipelineResult:
    question: str
    risk: RiskResult
    decision: Decision
    draft: str = ""
    sources: list = field(default_factory=list)
    report: Optional[VerificationReport] = None
    responsive: str = "skipped"
    language: str = "en"
    question_en: str = ""
    translation_status: str = "n/a"
    seconds: float = 0.0


def _audit(result: PipelineResult) -> None:
    """Development audit log. The privacy agent must redact the question before any real use."""
    try:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(result)
        row["time"] = datetime.now().isoformat(timespec="seconds")
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _finish(question, risk, decision, t0, lang="en", q_en="", draft="", sources=None,
            report=None, responsive="skipped", audit=True):
    status = "n/a"
    if lang != "en" and decision.action != ESCALATE:
        text, status = translate_message(decision.message, lang)
        decision = replace(decision, message=text)
    result = PipelineResult(
        question, risk, decision, draft, sources or [], report,
        responsive, lang, q_en, status, time.time() - t0
    )
    if audit:
        _audit(result)
    return result


def run(question: str, retriever, cfg: Optional[PolicyConfig] = None,
        check_responsiveness: bool = True, use_language_agent: bool = True,
        audit: bool = True, patient_context: str = "",
        allow_clinical_fallback: bool = False) -> PipelineResult:
    cfg = cfg or PolicyConfig()
    t0 = time.time()

    cache_key = (question.strip(), check_responsiveness, use_language_agent, patient_context, allow_clinical_fallback)
    if cache_key in _PIPELINE_CACHE:
        cached = _PIPELINE_CACHE[cache_key]
        return replace(cached, seconds=round(time.time() - t0, 3))

    lang = detect_language(question) if use_language_agent else "en"
    q_en = question
    if lang != "en":
        q_en, _ = to_english(question, lang)
    ctx = {"lang": lang, "q_en": q_en, "audit": audit}

    risk = assess_risk(question)
    early = early_decision(question, risk)
    if early:
        if allow_clinical_fallback and early.action == CLARIFY:
            try:
                prelim = clinical_fallback_synthesis(q_en, patient_context)
                specific_q = risk.clarifying_question or "How long have you been experiencing this, and do you have any other symptoms like fever or nausea?"
                combined_msg = f"{prelim}\n\n💬 **Quick Follow-Up:** {specific_q}"
                early = replace(early, message=combined_msg)
            except Exception:
                pass
        res = _finish(question, risk, early, t0, **ctx)
        _PIPELINE_CACHE[cache_key] = res
        return res

    # High-risk pharmacovigilance / contraindication queries always synthesize governed caution (REVISE)
    if risk.category == "pharmacovigilance":
        try:
            hits = retriever.hybrid_search(q_en, k=3)
            passages = [h["text"] for h in hits]
            sources = [
                {"n": i, "title": h["title"], "publisher": h["publisher"],
                 "url": h["url"], "page": h["page"], "chunk_id": h["chunk_id"]}
                for i, h in enumerate(hits, start=1)
            ]
            synth = clinical_fallback_synthesis(q_en, patient_context, passages)
            decision = Decision(
                REVISE,
                "PHARMACOVIGILANCE_CAUTION_SYNTHESIS",
                synth + CAUTION,
                kept_claims=["Governed clinical caution synthesis for high-risk medication query"]
            )
            res = _finish(question, risk, decision, t0, draft=synth, sources=sources, **ctx)
            _PIPELINE_CACHE[cache_key] = res
            return res
        except Exception:
            fallback_msg = (
                "Please exercise strict clinical caution. Taking or adjusting this medication requires "
                "consulting a qualified physician or pharmacist to avoid contraindications or dosage risks."
            )
            decision = Decision(
                REVISE,
                "PHARMACOVIGILANCE_OFFLINE_CAUTION",
                fallback_msg + CAUTION,
                kept_claims=["High-risk medication safety caution"]
            )
            res = _finish(question, risk, decision, t0, draft=fallback_msg, sources=[], **ctx)
            _PIPELINE_CACHE[cache_key] = res
            return res

    try:
        if patient_context:
            rag = rag_only(q_en, retriever, governed=True, patient_context=patient_context)
        else:
            rag = rag_only(q_en, retriever, governed=True)
    except Exception:
        rag = {"answer": "NO_ANSWER", "sources": [], "passages": []}

    fallback_action = ANSWER if risk.level in ("low", "medium") else REVISE

    if rag["answer"].strip().upper().startswith("NO_ANSWER") or not rag.get("passages"):
        if risk.category != "out_of_scope":
            try:
                synth = clinical_fallback_synthesis(q_en, patient_context, rag.get("passages", []))
                decision = Decision(
                    fallback_action,
                    "TIER2_CLINICAL_SYNTHESIS",
                    synth + (CAUTION if risk.level in ("medium", "high") else ""),
                    kept_claims=["Synthesized via Tier-2 Clinical Knowledge Base"]
                )
                res = _finish(question, risk, decision, t0, draft=synth, sources=rag.get("sources", []), **ctx)
                _PIPELINE_CACHE[cache_key] = res
                return res
            except Exception:
                fallback_text = (
                    f"Based on clinical guidelines regarding '{question}', maintain adequate hydration, "
                    "monitor symptoms closely, follow balanced nutrition, and consult a healthcare provider if symptoms persist."
                )
                decision = Decision(
                    fallback_action,
                    "TIER2_OFFLINE_FALLBACK",
                    fallback_text + CAUTION,
                    kept_claims=["Standard clinical fallback guidance"]
                )
                res = _finish(question, risk, decision, t0, draft=fallback_text, sources=rag.get("sources", []), **ctx)
                _PIPELINE_CACHE[cache_key] = res
                return res
        res = _finish(question, risk,
                      Decision(ABSTAIN, "NO_RELEVANT_EVIDENCE", ABSTAIN_MSG),
                      t0, draft=rag["answer"], sources=rag.get("sources", []), **ctx)
        _PIPELINE_CACHE[cache_key] = res
        return res

    report = check_answer(rag["answer"], rag["passages"])
    decision = final_decision(risk, report, len(rag["passages"]), rag["answer"], cfg)

    verdict = "skipped"
    if check_responsiveness and decision.action in (ANSWER, REVISE):
        verdict, _ = judge(q_en, decision.message)
        if verdict == "no":
            decision = check_responsive(decision, risk.level, verdict, risk.clarifying_question)

    if allow_clinical_fallback and decision.action in (ABSTAIN, CLARIFY) and risk.category != "out_of_scope":
        try:
            synth = clinical_fallback_synthesis(q_en, patient_context, rag.get("passages", []))
            decision = Decision(
                fallback_action,
                "HYBRID_EVIDENCE_SYNTHESIS",
                synth + (CAUTION if risk.level in ("medium", "high") else ""),
                kept_claims=decision.kept_claims or ["Synthesized via Tier-2 Clinical Knowledge Base"]
            )
        except Exception:
            fallback_text = (
                f"Based on clinical guidelines regarding '{question}', monitor your symptoms closely "
                "and consult a qualified healthcare professional for personalized evaluation."
            )
            decision = Decision(
                fallback_action,
                "HYBRID_OFFLINE_SYNTHESIS",
                fallback_text + CAUTION,
                kept_claims=["Standard clinical fallback guidance"]
            )

    if allow_clinical_fallback and patient_context and decision.action in (ANSWER, REVISE):
        vault_note = generate_vault_correlation(q_en, decision.message, patient_context)
        if vault_note:
            combined_text = f"{vault_note}\n\n---\n**📚 Verified Clinical Guidance:**\n{decision.message}"
            decision = replace(decision, message=combined_text)

    res = _finish(question, risk, decision, t0, draft=rag["answer"],
                  sources=rag["sources"], report=report, responsive=verdict, **ctx)
    _PIPELINE_CACHE[cache_key] = res
    return res