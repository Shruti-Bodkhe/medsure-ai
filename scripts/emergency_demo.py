import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.agents.emergency import build_response, detect, guess_lang

print("Type a question (empty line to quit).")
while True:
    q = input("\nQuestion: ").strip()
    if not q or q.lower() in {"quit", "exit"}:
        break
    r = detect(q)
    if r.is_emergency:
        print(f"EMERGENCY [{r.category}] matched: '{r.matched}'")
        print(build_response(r, guess_lang(q)))
    else:
        print("No emergency detected.")