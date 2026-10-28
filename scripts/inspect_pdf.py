import sys

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

sys.stdout.reconfigure(encoding="utf-8")
path = sys.argv[1]

with pymupdf.open(path) as doc:
    pages = [p.get_text("text") for p in doc]
    for i, (p, t) in enumerate(zip(doc, pages), start=1):
        print(f"page {i}: {len(t.split())} words, {len(p.get_images())} images")

full = " ".join(pages).lower()
print()
for kw in ["transmission", "symptoms", "treatment", "prevention", "aedes", "bite"]:
    print(f"'{kw}' appears {full.count(kw)} times in the PDF text")