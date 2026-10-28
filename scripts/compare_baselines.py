import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.baselines import llm_only, rag_only
from app.retrieval.search import Retriever

QUESTIONS = [
    "How does dengue spread?",
    "What should I do after a dog bite?",
    "Chest pain and my left arm hurts, what should I do?",
    "Can I stop my diabetes medicine if my sugar is normal now?",
    "डेंगू के लक्षण क्या होते हैं?",
]

OUT = Path("experiments/baseline_demo.md")
OUT.parent.mkdir(exist_ok=True)

print("Loading search index (first run takes a minute or two)...")
retriever = Retriever()

lines = []


def say(text=""):
    print(text)
    lines.append(text)


for q in QUESTIONS:
    say("=" * 78)
    say(f"QUESTION: {q}")

    try:
        a = llm_only(q)
        say("\n--- LLM-ONLY ---\n" + a["answer"])
    except Exception as e:
        say(f"\n--- LLM-ONLY FAILED ---\n{e}")
    time.sleep(3)

    try:
        b = rag_only(q, retriever)
        say("\n--- RAG-ONLY ---\n" + b["answer"])
        say("\nSources used:")
        for s in b["sources"]:
            say(f"  [{s['n']}] {s['title']} ({s['publisher']}, page {s['page']})")
    except Exception as e:
        say(f"\n--- RAG-ONLY FAILED ---\n{e}")
    time.sleep(3)
    say()

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {OUT}")