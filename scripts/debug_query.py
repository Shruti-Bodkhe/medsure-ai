import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.retrieval.search import Retriever

query = " ".join(sys.argv[1:]) or "How does dengue spread?"
r = Retriever()
index_of = {c["chunk_id"]: i for i, c in enumerate(r.chunks)}
bm = {i: rank for rank, (i, _) in enumerate(r.bm25_search(query, 50), 1)}
de = {i: rank for rank, (i, _) in enumerate(r.dense_search(query, 50), 1)}

print(f"\nQUERY: {query}\n\nFINAL TOP 10 (bm25 rank / dense rank):")
for pos, h in enumerate(r.hybrid_search(query, k=10), 1):
    i = index_of[h["chunk_id"]]
    print(f"{pos:2}. {h['chunk_id']:<42} bm25={bm.get(i, '-')}  dense={de.get(i, '-')}")

print("\nDENGUE CHUNKS THAT MENTION 'Aedes' (ranks out of 50):")
for i, c in enumerate(r.chunks):
    if "who_dengue" in c["doc"] and "aedes" in c["text"].lower():
        print(f"  {c['chunk_id']} page={c['page']} bm25={bm.get(i, '-')} dense={de.get(i, '-')}")
        print(f"      {c['text'][:140]}...")