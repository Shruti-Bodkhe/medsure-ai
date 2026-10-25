import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader

RAW = Path("data/raw")
REGISTRY = Path("data/sources.csv")
OUT = Path("data/processed/chunks.jsonl")
CHUNK_WORDS = 200
OVERLAP = 40
MIN_WORDS = 30


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # join words split across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pages(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            yield i, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def make_chunks(words):
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_WORDS:
            yield " ".join(piece)
        if start + CHUNK_WORDS >= len(words):
            break


def load_registry():
    with open(REGISTRY, encoding="utf-8-sig", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def main():
    registry = load_registry()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    problems = []
    with open(OUT, "w", encoding="utf-8") as out:
        for path in sorted(RAW.iterdir()):
            if path.suffix.lower() not in {".pdf", ".txt"}:
                continue
            meta = registry.get(path.name)
            if meta is None:
                print(f"SKIPPED (not in sources.csv): {path.name}")
                problems.append(path.name)
                continue
            n = 0
            for page_no, raw in read_pages(path):
                words = clean(raw).split()
                for text in make_chunks(words):
                    n += 1
                    record = {
                        "chunk_id": f"{path.stem}_p{page_no}_{n}",
                        "doc": path.name,
                        "title": meta["title"],
                        "publisher": meta["publisher"],
                        "url": meta["url"],
                        "language": meta["language"],
                        "page": page_no,
                        "text": text,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{path.name}: {n} chunks")
            if n == 0:
                problems.append(path.name)
            total += n
    print(f"\nTotal chunks: {total}")
    if problems:
        print("Files needing attention:", ", ".join(problems))


if __name__ == "__main__":
    main()