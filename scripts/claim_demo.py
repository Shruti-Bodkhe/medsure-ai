import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.agents.claims import check_answer
from app.baselines import llm_only
from app.retrieval.search import Retriever

question = " ".join(sys.argv[1:]) or "What should I do after a dog bite?"

print("Loading search index...")
retriever = Retriever()
hits = retriever.hybrid_search(question, k=5)
passages = [h["text"] for h in hits]

print(f"\nQUESTION: {question}")
print("\nEvidence retrieved:")
for i, h in enumerate(hits, start=1):
    print(f"  [{i}] {h['title']} ({h['publisher']}, page {h['page']})")

answer = llm_only(question)["answer"]
print("\n--- LLM-ONLY ANSWER (not shown any evidence) ---\n" + answer)

report = check_answer(answer, passages)
print("\n--- CLAIM CHECK AGAINST THE EVIDENCE ---")
if report.error:
    print("NOTE:", report.error)
mark = {"supported": "OK     ", "contradicted": "WRONG  ",
        "not_found": "NO PROOF", "unverified": "UNCHECKED"}
for c in report.claims:
    print(f"[{mark[c.label]}] {c.text}")
    if c.label != "supported":
        print(f"            why: {c.reason}")
if report.support_rate is not None:
    print(f"\nSupport rate: {report.support_rate:.0%}  "
          f"({len(report.unsupported)} of {len(report.claims)} claims not supported)")