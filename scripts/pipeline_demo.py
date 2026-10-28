import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import run
from app.retrieval.search import Retriever

QUESTIONS = sys.argv[1:] or [
    "How does dengue spread?",
    "What should I do after a dog bite?",
    "Chest pain and my left arm hurts, what should I do?",
    "Can I stop my diabetes medicine if my sugar is normal now?",
    "I have a headache since morning",
    "What is the weather today?",
]

print("Loading search index...")
retriever = Retriever()

for q in QUESTIONS:
    r = run(q, retriever)
    d = r.decision
    print("\n" + "=" * 78)
    print(f"QUESTION: {q}")
    print(f"RISK: {r.risk.level} ({r.risk.source})   ACTION: {d.action}   "
          f"REASON: {d.reason_code}   TIME: {r.seconds:.0f}s")
    print("\nRESPONSE SHOWN TO USER:\n" + d.message)
    if d.dropped_claims:
        print("\nLEFT OUT (not supported by the sources):")
        for c in d.dropped_claims:
            print("  -", c)