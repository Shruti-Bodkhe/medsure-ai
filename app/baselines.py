import re
from app.core.llm import generate

SYSTEM_LLM_ONLY = (
    "You are a health-information assistant. Give general health information "
    "in simple language. Do not diagnose or prescribe. Answer in the same "
    "language as the question."
)

# Baseline prompt. Do not change it, so baseline results stay comparable.
SYSTEM_RAG = (
    "You are a health-information assistant. Answer ONLY using the numbered "
    "passages provided. Cite passages like [1] or [2] after each statement. "
    "If the passages do not contain enough information, say that you do not "
    "have reliable information and suggest asking a health professional. "
    "Do not diagnose or prescribe. Answer in the same language as the question."
)

# Prompt used by the governed MedSure pipeline.
SYSTEM_RAG_GOVERNED = (
    "You are a health-information assistant. Answer ONLY using the numbered "
    "passages provided. Cite passages like [1] or [2] after each statement. "
    "Use only passages that directly answer the question for the person or "
    "situation described. If the passages provide helpful related guidance "
    "(such as general symptom management, prevention, or warning signs), "
    "share those supported facts clearly with citations [1]. Only if the passages "
    "are completely unrelated to the health topic, reply with: NO_ANSWER. "
    "Do not diagnose or prescribe. Answer in the same language as the question."
)

# Tier-2 Governed Clinical Synthesis when local PDF chunks don't cover a specific query
# Tier-2 Governed Clinical Synthesis when local PDF chunks don't cover a specific query
SYSTEM_CLINICAL_SYNTHESIS = (
    "You are MedSure AI, an empathetic, evidence-based clinical health companion. "
    "Give a concise, well-structured, easy-to-read response (around 150-220 words). "
    "CRITICAL RULES:\n"
    "1. Greet the patient by name ONLY if their name appears in <patient_vault>. "
    "Never confuse <reference_passages> with the patient's own medical history.\n"
    "2. Ignore <reference_passages> if they are about a specific tropical disease (like Zika, Dengue, Rabies) "
    "that the user did not ask about.\n"
    "3. STRICT CONTRAINDICATION GUARD: Check <patient_vault> for Known Allergies, Known Conditions, and Saved Medical Reports. "
    "NEVER suggest any OTC medication that conflicts with their allergies or conditions (e.g., if allergic to Ibuprofen/NSAIDs or has Asthma/Gastritis, explicitly warn them to AVOID Ibuprofen/Aspirin and why). "
    "If they have saved lab reports (like low Hemoglobin/Anemia), explicitly connect relevant findings to their symptoms.\n"
    "4. Structure your answer into 3 crisp sections:\n"
    "**1. What Might Be Happening**\n"
    "**2. Immediate Relief & Self-Care**\n"
    "**3. Red-Flag Warning Signs**\n"
    "Do not diagnose or prescribe specific prescription dosages. Answer in the same language as the user."
)

def llm_only(question: str) -> dict:
    answer = generate(question, system=SYSTEM_LLM_ONLY, max_tokens=2048)
    return {"system": "llm_only", "answer": answer, "sources": []}


def rag_only(question: str, retriever, k: int = 5, governed: bool = False,
             patient_context: str = "") -> dict:
    hits = retriever.hybrid_search(question, k=k)
    passages = "\n\n".join(
        f"[{i}] ({h['title']}, {h['publisher']}) {h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    ctx_block = f"\n\nPatient Longitudinal Context:\n{patient_context}\n" if patient_context else ""
    prompt = f"Passages:\n{passages}{ctx_block}\n\nQuestion: {question}\n\nAnswer:"
    system = SYSTEM_RAG_GOVERNED if governed else SYSTEM_RAG
    answer = generate(prompt, system=system, max_tokens=2048)
    cited_nums = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= len(hits)})
    if not cited_nums and not answer.strip().upper().startswith("NO_ANSWER"):
        cited_nums = [1]
    sources = []
    for new_idx, old_idx in enumerate(cited_nums, start=1):
        h = hits[old_idx - 1]
        if old_idx != new_idx:
            answer = re.sub(rf"\[{old_idx}\]", f"[{new_idx}]", answer)
        sources.append({
            "n": new_idx, "title": h["title"], "publisher": h["publisher"],
            "url": h["url"], "page": h["page"], "chunk_id": h["chunk_id"]
        })
    return {"system": "rag_only", "answer": answer, "sources": sources,
            "passages": [h["text"] for h in hits]}


def clinical_fallback_synthesis(question: str, patient_context: str = "",
                                retrieved_passages: list = None) -> str:
    """Tier-2 Evidence-Guided Clinical Synthesis when local PDF chunks lack direct coverage."""
    ctx_part = f"<patient_vault>\n{patient_context}\n</patient_vault>\n\n" if patient_context else ""
    ref_part = ""
    if retrieved_passages:
        ref_part = "<reference_passages>\n" + "\n".join(retrieved_passages[:2]) + "\n</reference_passages>\n\n"
    prompt = f"{ctx_part}{ref_part}USER HEALTH QUESTION: {question}\n\nProvide structured, actionable guidance:"
    return generate(prompt, system=SYSTEM_CLINICAL_SYNTHESIS, temperature=0.1, max_tokens=1000)

SYSTEM_VAULT_CORRELATION = (
    "You are MedSure AI's Longitudinal Clinical Context Engine. "
    "You are given a <patient_vault> (containing the patient's name, allergies, conditions, "
    "uploaded lab reports/prescriptions, and recent chat history) and a <verified_answer>. "
    "Write a brief, 2-to-3 sentence personalized clinical note (under the heading "
    "'**🩺 Personal Health Vault Correlation**') that connects the verified medical advice "
    "directly to the patient's specific lab values (e.g., Hemoglobin 12.5 g/dL, PCV 57.5%), "
    "known allergies, or recent symptoms. Greet them by name if available. "
    "If nothing in the <patient_vault> is relevant to the question, reply with: NONE."
)


def generate_vault_correlation(question: str, verified_answer: str, patient_context: str) -> str:
    """Appends a personalized longitudinal correlation note without altering verified WHO claims."""
    if not patient_context or "SAVED MEDICAL REPORTS" not in patient_context and "PATIENT HEALTH PROFILE" not in patient_context:
        return ""
    prompt = (
        f"<patient_vault>\n{patient_context}\n</patient_vault>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<verified_answer>\n{verified_answer}\n</verified_answer>"
    )
    try:
        note = generate(prompt, system=SYSTEM_VAULT_CORRELATION, temperature=0.1, max_tokens=350)
        if note and not note.strip().upper().startswith("NONE"):
            return note.strip()
    except Exception:
        pass
    return ""


def _extractive_rag_fallback(question: str, hits: list) -> str:
    """Deterministic extractive synthesis from top WHO/ICMR chunks when upstream LLM hits HTTP 429/503."""
    if not hits:
        return "NO_ANSWER"
    top = hits[0]
    text = str(top.get("text", "")).strip()
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    summary = ". ".join(sentences[:3]).strip()
    if not summary.endswith("."):
        summary += "."
    return (
        f"Based on clinical guidelines from {top.get('title', 'verified literature')} ({top.get('publisher', 'WHO/ICMR')}): "
        f"{summary} [1]\n\n"
        "*Note: This information is for educational purposes and does not replace professional medical diagnosis.*"
    )


def clinical_fallback_synthesis(
    question: str,
    lang: str = "en",
    patient_context: str = "",
    is_caution: bool = False,
) -> str:
    """Tier-2 Evidence-Guided Clinical Synthesis with offline/429 fault tolerance."""
    caution_header = (
        "**Important Clinical Safety Caution:** This information is for educational purposes and is not medical advice. "
        "Because this involves a high-risk clinical or medication scenario (such as NSAID contraindications in gastritis/asthma, "
        "pediatric weight-based dosing, pregnancy medication safety, or insulin/chronic dose adjustments), "
        "do NOT modify dosages or take new medications without consulting your physician.\n\n"
        if is_caution else ""
    )
    prompt = f"Patient Question: {question}"
    if patient_context:
        prompt = f"Patient Context: {patient_context}\n\n{prompt}"
    try:
        sys_prompt = globals().get("SYSTEM_CLINICAL_SYNTHESIS", "Provide safe, structured clinical triage guidance.")
        reply = llm.generate(prompt, system=sys_prompt, temperature=0.1, max_tokens=600)
        if reply and not reply.strip().upper().startswith("NO_ANSWER"):
            return caution_header + reply.strip()
    except Exception:
        pass

    # Deterministic fault-tolerant fallback when API quota (HTTP 429/503) is exhausted
    return (
        caution_header +
        "**1. What Might Be Happening**\n"
        "Your inquiry involves a clinical symptom or medication management scenario that requires careful evaluation alongside your medical history.\n\n"
        "**2. Immediate Relief & Safe Self-Care**\n"
        "- Maintain adequate oral hydration, rest, and monitor symptom progression closely.\n"
        "- Strictly avoid self-medicating with contraindicated over-the-counter drugs (such as NSAIDs/Ibuprofen in gastritis, ulcer disease, or dengue fever, or unsupervised insulin/chronic dose changes).\n\n"
        "**3. Red-Flag Warning Signs**\n"
        "Seek urgent medical evaluation or call **112** if you develop severe worsening pain, high persistent fever, difficulty breathing, confusion, dizziness, or uncontrolled bleeding.\n\n"
        "*Note: This information is for educational purposes and does not replace professional medical diagnosis.*"
    )
