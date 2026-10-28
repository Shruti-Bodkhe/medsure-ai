from app.core.llm import generate

SYSTEM_LLM_ONLY = (
    "You are a health-information assistant. Give general health information "
    "in simple language. Do not diagnose or prescribe. Answer in the same "
    "language as the question."
)

SYSTEM_RAG = (
    "You are a health-information assistant. Answer ONLY using the numbered "
    "passages provided. Cite passages like [1] or [2] after each statement. "
    "If the passages do not contain enough information, say that you do not "
    "have reliable information and suggest asking a health professional. "
    "Do not diagnose or prescribe. Answer in the same language as the question."
)


def llm_only(question: str) -> dict:
    answer = generate(question, system=SYSTEM_LLM_ONLY, max_tokens=700)
    return {"system": "llm_only", "answer": answer, "sources": []}


def rag_only(question: str, retriever, k: int = 5) -> dict:
    hits = retriever.hybrid_search(question, k=k)
    passages = "\n\n".join(
        f"[{i}] ({h['title']}, {h['publisher']}) {h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    prompt = f"Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:"
    answer = generate(prompt, system=SYSTEM_RAG, max_tokens=2048)
    sources = [
        {"n": i, "title": h["title"], "publisher": h["publisher"],
         "url": h["url"], "page": h["page"], "chunk_id": h["chunk_id"]}
        for i, h in enumerate(hits, start=1)
    ]
    return {"system": "rag_only", "answer": answer, "sources": sources,
            "passages": [h["text"] for h in hits]}