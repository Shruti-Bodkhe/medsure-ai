import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.retrieval.search import Retriever

retriever = Retriever()
print("Ready. Type a health question (or 'quit').\n")
while True:
    query = input("Question: ").strip()
    if query.lower() in {"quit", "exit", ""}:
        break
    for rank, hit in enumerate(retriever.hybrid_search(query, k=5), start=1):
        print(f"\n{rank}. {hit['title']} ({hit['publisher']}, page {hit['page']})  score={hit['score']}")
        print(f"   {hit['text'][:220]}...")
    print()