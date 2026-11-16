import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CACHE_DIR = Path("data/processed")
MODEL_NAME = "intfloat/multilingual-e5-small"
# letters, digits and the whole Devanagari block (Hindi/Marathi)
TOKEN = re.compile(r"[A-Za-z0-9\u0900-\u097F]+")
RRF_K = 60

BM25_STOP = {
    "morning", "since", "day", "days", "feel", "feeling", "started", "how", "does", "do", "did", "what", "which", "who", "when", "where", "why",
    "is", "are", "was", "were", "can", "could", "should", "would", "i", "my",
    "me", "you", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    "it", "this", "that", "with", "after", "if", "be", "have", "has",
    "के", "का", "की", "है", "हैं", "में", "से", "को", "और", "पर", "क्या",
    "होते", "होता", "होती", "करें", "कैसे", "मैं", "मेरा", "मेरी",
}

def tokenize(text: str):
    return [t for t in TOKEN.findall(text.lower()) if t not in BM25_STOP]


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        texts = [c["text"] for c in self.chunks]

        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

        self.model = SentenceTransformer(MODEL_NAME)
        digest = hashlib.md5(CHUNKS_PATH.read_bytes()).hexdigest()[:8]
        cache = CACHE_DIR / f"embeddings_{digest}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            print("Embedding chunks (first run only)...")
            emb = self.model.encode(
                ["passage: " + t for t in texts],
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=True,
            ).astype("float32")
            np.save(cache, emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def bm25_search(self, query: str, k: int = 20):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, k: int = 20):
        q = self.model.encode(["query: " + query], normalize_embeddings=True)
        sims, ids = self.index.search(q.astype("float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]

    def hybrid_search(self, query: str, k: int = 5, pool: int = 25):
        q_lower = query.lower()
        fused = {}
        for results in (self.bm25_search(query, pool), self.dense_search(query, pool)):
            for rank, (i, _) in enumerate(results, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

        restricted_keywords = {
            "who_zika": ["zika", "microcephaly", "mosquito"],
            "who_plague": ["plague", "bubonic", "flea"],
            "who_yellow_fever": ["yellow fever", "jaundice"],
            "who_rabies": ["rabies", "dog", "bite", "animal", "scratch"],
            "who_snakebite": ["snake", "snakebite", "venom", "saanp", "????", "????"],
            "who_heart_attack": ["chest", "heart", "cardiac", "arm", "sweating", "angina", "chhaati", "seene", "????", "?????"],
            "who_stroke": ["stroke", "paralysis", "slurred", "droop", "lakwa", "????", "???????"],
        }

        filtered = []
        for i, s in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            chunk = self.chunks[i]
            doc_name = chunk["doc"].lower()
            skip = False
            for doc_prefix, req_terms in restricted_keywords.items():
                if doc_prefix in doc_name and not any(t in q_lower for t in req_terms):
                    skip = True
                    break
            if not skip:
                filtered.append({**chunk, "score": round(s, 4)})
            if len(filtered) >= k:
                break

        return filtered