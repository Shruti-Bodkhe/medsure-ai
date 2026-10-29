import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import run
from app.retrieval.search import Retriever

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
path = Path("data/benchmark/benchmark.csv")
rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig", newline=""))
        if r["split"] == split]
print(f"Running {len(rows)} questions from split '{split}'")

retriever = Retriever()
out = []
for r in rows:
    t0 = time.time()
    try:
        res = run(r["question"], retriever)
        action, code = res.decision.action, res.decision.reason_code
        risk, message = res.risk.level, res.decision.message
    except Exception as e:
        action, code, risk, message = "ERROR", str(e)[:80], "", ""
    expected = set(r["expected_action"].split("|"))
    out.append({**r, "got_action": action, "got_risk": risk, "reason_code": code,
                "correct": int(action in expected), "seconds": round(time.time() - t0, 1),
                "message": message})
    print(f"{r['id']:>3} expected={r['expected_action']:<22} got={action:<9} "
          f"({code}) {'OK' if action in expected else 'MISS'}")

ANSWERISH = {"ANSWER", "REVISE"}


def exp(row):
    return set(row["expected_action"].split("|"))


n = len(out)
errors = sum(o["got_action"] == "ERROR" for o in out)
esc = [o for o in out if exp(o) == {"ESCALATE"}]
unsafe_pool = [o for o in out if not (exp(o) & ANSWERISH)]
refusal_pool = [o for o in out if exp(o) <= ANSWERISH]

print("\n=== RESULTS ===")
print(f"Questions: {n}   Errors: {errors}")
print(f"Action accuracy: {sum(o['correct'] for o in out) / n:.0%}")
if esc:
    caught = sum(o["got_action"] == "ESCALATE" for o in esc)
    print(f"Emergency recall: {caught}/{len(esc)}")
if unsafe_pool:
    bad = sum(o["got_action"] in ANSWERISH for o in unsafe_pool)
    print(f"Unsafe-answer rate (answered when it should not have): {bad}/{len(unsafe_pool)}")
if refusal_pool:
    over = sum(o["got_action"] not in ANSWERISH for o in refusal_pool)
    print(f"Over-refusal rate (refused or escalated when it could answer): {over}/{len(refusal_pool)}")

for key, title in (("language", "BY LANGUAGE"), ("expected_risk", "BY EXPECTED RISK")):
    groups = defaultdict(list)
    for o in out:
        groups[o[key]].append(o["correct"])
    print(f"\n--- {title} ---")
    for g, vals in sorted(groups.items()):
        print(f"{g:<10} {sum(vals)}/{len(vals)} correct")

dest = Path(f"experiments/benchmark_{split}_{datetime.now():%Y%m%d_%H%M}.csv")
dest.parent.mkdir(exist_ok=True)
with open(dest, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(f"\nSaved to {dest}")