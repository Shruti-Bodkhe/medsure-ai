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
LOG_OUT = Path("data/processed/dropped_chunks.jsonl")
CHUNK_WORDS = 200
OVERLAP = 40
MIN_WORDS = 30

# ---- non-prose filter -------------------------------------------------------
STOP = {
    "the", "of", "and", "to", "in", "a", "is", "are", "for", "that", "with",
    "as", "by", "on", "can", "be", "or", "it", "from", "this", "which",
    "have", "has", "not", "at", "was", "were", "an",
}
MIN_STOP_RATIO = 0.10
JUNK_MARKERS = [
    "Feature story", "Latest publications", "WHO Academy Course",
    "Science in 5", "Disease Outbreak News", "Self-learning course",
    "Fact sheets Related", "Subscribe to our newsletter", "Our work ",
]

# ---- browser print header/footer patterns ----------------------------------
# e.g. "10/25/26, 1:47 PM Animal bites https://www.who.int/... 2/8"
FOOTER = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M[^/]{0,120}?https?://\S+\s+\d+/\d+"
)
URL_PAGE = re.compile(r"https?://\S+\s+\d+/\d+\b")
DATE_ONLY = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M")


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # join words split across lines
    text = re.sub(r"\s+", " ", text)
    text = FOOTER.sub(" ", text)
    text = URL_PAGE.sub(" ", text)
    text = DATE_ONLY.sub(" ", text)
    text = re.sub(r"\[https?://[^\]]*\]", " ", text)  # drop "[https://...]" link markup
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
    while tokens and tokens[0][0] in {"Donate", "©"}:
        tokens = tokens[1:]
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i][0] == "References" and i > len(tokens) * 0.5:
            return tokens[:i]
    return tokens


def make_chunks(tokens):
    """tokens: list of (word, page). Yields (text, start_page)."""
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + CHUNK_WORDS]
        if len(piece) >= MIN_WORDS:
            yield " ".join(w for w, _ in piece), piece[0][1]
        if start + CHUNK_WORDS >= len(tokens):
            break


def screen(text: str):
    """Return (kept_text, status, reason).
    status: 'kept', 'trimmed' (cut at a junk marker) or 'dropped'."""
    status, reason = "kept", ""

    # 1. cut the chunk where website "related content" begins
    cut, hit = len(text), None
    for marker in JUNK_MARKERS:
        i = text.find(marker)
        if i != -1 and i < cut:
            cut, hit = i, marker
    if hit:
        text = text[:cut].strip()
        status, reason = "trimmed", f"cut at marker '{hit}'"

    words = text.lower().split()
    if len(words) < MIN_WORDS:
        return None, "dropped", reason + " -> too short after cut"

    # 2. not enough ordinary sentence words (menus, link titles, lists)
    ratio = sum(w in STOP for w in words) / len(words)
    if ratio < MIN_STOP_RATIO:
        return None, "dropped", f"low prose ratio {ratio:.2f}"

    return text, status, reason


def load_registry():
    with open(REGISTRY, encoding="utf-8-sig", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def main():
    registry = load_registry()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    problems = []
    log = []
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
                kept, status, reason = screen(text)
                if status != "kept":
                    log.append({"doc": path.name, "page": page_no, "status": status,
                                "reason": reason, "text": text[:200]})
                if kept is None:
                    continue
                n += 1
                record = {
                    "chunk_id": f"{path.stem}_{n:03d}",
                    "doc": path.name,
                    "title": meta["title"],
                    "publisher": meta["publisher"],
                    "url": meta["url"],
                    "language": meta["language"],
                    "page": page_no,
                    "text": kept,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{path.name}: {n} chunks")
            if n == 0:
                problems.append(path.name)
            total += n
    print(f"\nTotal chunks: {total}")
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        for d in log:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    dropped = sum(1 for d in log if d["status"] == "dropped")
    print(f"Dropped: {dropped}   Trimmed: {len(log) - dropped}   (see {LOG_OUT})")
    if problems:
        print("Files needing attention:", ", ".join(problems))


if __name__ == "__main__":
    main()