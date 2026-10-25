import csv
import re
from collections import Counter
from pathlib import Path

import httpx
from pypdf import PdfReader

RAW = Path("data/raw")
CSV_PATH = Path("data/sources.csv")
URL_RE = re.compile(r'https?://(?:www\.)?who\.int/[^\s"<>]+')
BASE = "https://www.who.int/news-room/fact-sheets/detail/"


def url_from_pdf(path):
    try:
        reader = PdfReader(str(path))
        text = " ".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return None
    found = [u.rstrip(".,;") for u in URL_RE.findall(text)]
    detail = [u for u in found if "/fact-sheets/detail/" in u]
    if detail:
        return Counter(detail).most_common(1)[0][0]
    return None


def guess_url(filename):
    stem = Path(filename).stem
    if stem.startswith("who_"):
        stem = stem[4:]
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return BASE + slug


def check(url):
    try:
        r = httpx.get(url, follow_redirects=True, timeout=20,
                      headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code
    except Exception:
        return None


def main():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    problems = []
    for row in rows:
        pdf_path = RAW / row["filename"]
        url = url_from_pdf(pdf_path)
        origin = "pdf"
        if not url:
            url = guess_url(row["filename"])
            origin = "guess"
        status = check(url)
        row["url"] = url
        print(f"{status}  [{origin}]  {row['filename']}\n      {url}")
        if status != 200:
            problems.append((row["filename"], url, status))

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== NEED MANUAL CHECK ===")
    if not problems:
        print("None. All URLs returned 200.")
    for name, url, status in problems:
        print(f"{name}  ->  status {status}\n      {url}")


if __name__ == "__main__":
    main()