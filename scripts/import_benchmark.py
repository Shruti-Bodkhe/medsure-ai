import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(sys.argv[1])
DEST = Path("data/benchmark/benchmark_test.csv")

ALIASES = {
    "question": ["question", "query", "text", "prompt", "user_query"],
    "language": ["language", "lang"],
    "expected_action": ["expected_action", "action", "expected", "label", "correct_action"],
    "expected_risk": ["expected_risk", "risk", "risk_level", "severity"],
    "notes": ["notes", "note", "category", "comment", "topic"],
}
LANG_MAP = {"english": "en", "en": "en", "hindi": "hi", "hi": "hi", "marathi": "mr",
            "mr": "mr", "hinglish": "hinglish", "code-mixed": "hinglish",
            "code mixed": "hinglish", "codemixed": "hinglish", "romanised": "hinglish",
            "romanized": "hinglish", "mixed": "hinglish"}
ACTIONS = {"ANSWER", "REVISE", "CLARIFY", "ABSTAIN", "ESCALATE"}
ACTION_FIX = {"ANSWER WITH CAUTION": "ANSWER|REVISE", "ANSWER_WITH_CAUTION": "ANSWER|REVISE",
              "CAUTION": "ANSWER|REVISE", "REFUSE": "ABSTAIN", "REFER": "ESCALATE",
              "ASK": "CLARIFY", "ASK CLARIFICATION": "CLARIFY"}

raw = list(csv.DictReader(open(SRC, encoding="utf-8-sig", newline="")))
if not raw:
    raise SystemExit("The CSV is empty.")
cols = {c.strip().lower(): c for c in raw[0].keys()}
pick = {}
for target, names in ALIASES.items():
    pick[target] = next((cols[n] for n in names if n in cols), None)
print("Column mapping:", {k: v for k, v in pick.items()})
for must in ("question", "expected_action"):
    if not pick[must]:
        raise SystemExit(f"No column found for '{must}'. Columns in your file: {list(raw[0].keys())}")

out, bad = [], []
for i, r in enumerate(raw, start=1):
    q = (r[pick["question"]] or "").strip()
    if not q:
        continue
    act = (r[pick["expected_action"]] or "").strip().upper().replace("/", "|").replace(",", "|")
    parts = [ACTION_FIX.get(p.strip(), p.strip()) for p in act.split("|") if p.strip()]
    acts = "|".join("|".join(p.split("|")) for p in parts)
    if not acts or not set(acts.split("|")) <= ACTIONS:
        bad.append((i, q[:50], act))
        continue
    lang_raw = (r[pick["language"]] if pick["language"] else "").strip().lower()
    out.append({
        "id": 1000 + i, "question": q,
        "language": LANG_MAP.get(lang_raw, lang_raw or "unknown"),
        "expected_risk": (r[pick["expected_risk"]] if pick["expected_risk"] else "").strip().lower(),
        "expected_action": acts, "split": "test",
        "notes": (r[pick["notes"]] if pick["notes"] else ""),
    })

DEST.parent.mkdir(parents=True, exist_ok=True)
with open(DEST, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "question", "language", "expected_risk",
                                      "expected_action", "split", "notes"])
    w.writeheader()
    w.writerows(out)

print(f"\nImported {len(out)} questions to {DEST}")
print("By language:", dict(Counter(o["language"] for o in out)))
print("By action:  ", dict(Counter(o["expected_action"] for o in out)))
if bad:
    print(f"\n{len(bad)} rows skipped (unreadable action). First few:")
    for b in bad[:10]:
        print("  row", b[0], "|", b[1], "| action =", repr(b[2]))
        