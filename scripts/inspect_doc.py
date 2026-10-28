import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
name = sys.argv[1]  # for example: who_dengue

for line in open("data/processed/chunks.jsonl", encoding="utf-8"):
    c = json.loads(line)
    if name in c["doc"]:
        print(f"{c['chunk_id']} (page {c['page']}): {c['text'][:170]}...\n")