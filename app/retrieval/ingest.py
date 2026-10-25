import csv
import json
import re
from pathlib import Path

try:
    import pymupdf
except ImportError:  # older PyMuPDF versions
    import fitz as pymupdf

RAW = Path("data/raw")
REGISTRY = Path("data/sources.csv")
OUT = Path("data/processed/chunks.jsonl")
CHUNK_WORDS = 200
OVERLAP = 40
MIN_WORDS = 30

# Browser print header/footer, e.g. "10/25/26, 1:47 PM Animal bites https://www.who.int/... 2/8"
FOOTER = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M[^/]{0,120}?https?://\S+\s+\d+/\d+"
)
URL_PAGE = re.compile(r"https?://\S+\s+\d+/\d+\b")
DATE_ONLY = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M")


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = FOOTER.sub(" ", text)
    text = URL_PAGE.sub(" ", text)
    text = DATE_ONLY.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path: Path):
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(str(path)) as doc:
            for i, page in enumerate(doc, start=1):
                yield i, page.get_text("text")
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")

def trim_tokens(tokens):
    """Drop site navigation at the start and the reference list at the end."""
    # 1. remove leading "Donate ©"
    while tokens and tokens[0][0] in {"Donate", "©"}:
        tokens = tokens[1:]
    # 2. cut at the last "References" heading, if it is in the second half
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i][0] == "References" and i > len(tokens) * 0.5:
            return tokens[:i]
    return tokens

def make_chunks(tokens):
    """tokens: list of (word, page). Returns (text, start_page) chunks."""
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_WORDS:
            yield " ".join(w for w, _ in piece), piece[0][1]
        if start + CHUNK_WORDS >= len(tokens):
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
            tokens = []
            for page_no, raw in read_pages(path):
                tokens.extend((w, page_no) for w in clean(raw).split())
            tokens = trim_tokens(tokens)
            n = 0
            for text, page_no in make_chunks(tokens):
                n += 1
                record = {
                    "chunk_id": f"{path.stem}_{n:03d}",
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