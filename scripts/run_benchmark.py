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

use_judge = "--no-judge" not in sys.argv
tag = "" if use_judge else "_nojudge"

path = Path(r"C:\Users\shruti\medsure-ai\data\benchmark\MedSureBench-Clean.csv")
if not path.exists():
    path = Path("data/benchmark/benchmark.csv")

print(f"Loading benchmark dataset from {path}...")

with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = [{str(k).strip(): (v.strip() if v else "") for k, v in r.items() if k} for r in reader if any(r.values())]

print(f"Running {len(rows)} questions (responsiveness judge: {'on' if use_judge else 'OFF'})")

retriever = Retriever()
out = []
for idx, r in enumerate(rows, start=1):
    t0 = time.time()
    q = r.get("question", "") or r.get("query", "") or r.get("text", "")
    expected_act_raw = r.get("expected_action", "") or r.get("action", "") or "ANSWER"
    expected_risk_val = r.get("expected_risk", "") or r.get("risk", "") or "low"
    lang_val = r.get("language", "") or r.get("lang", "") or "en"

    if not q:
        continue

    try:
        res = run(q, retriever, check_responsiveness=use_judge, allow_clinical_fallback=True)
        action, code = res.decision.action, res.decision.reason_code
        risk, message = res.risk.level, res.decision.message
    except Exception as e:
        action, code, risk, message = "ERROR", str(e)[:80], "", ""

    expected = set(expected_act_raw.upper().replace("/", "|").replace(",", "|").split("|"))
    is_correct = int(action in expected)
    elapsed = round(time.time() - t0, 1)

    out.append({
        "id": r.get("id", idx),
        "question": q,
        "language": lang_val,
        "expected_risk": expected_risk_val,
        "expected_action": "|".join(expected),
        "got_action": action,
        "got_risk": risk,
        "reason_code": code,
        "correct": is_correct,
        "seconds": elapsed
    })
    print(f"{idx:>3}/{len(rows)} expected={expected_act_raw:<22} got={action:<9} ({code}) {'OK' if is_correct else 'MISS'}")
    
    # Throttle slightly to prevent API rate limits (HTTP 429)
    time.sleep(1.5)

ANSWERISH = {"ANSWER", "REVISE"}
n = len(out)
if n == 0:
    raise SystemExit("No valid benchmark rows processed.")

errors = sum(o["got_action"] == "ERROR" for o in out)
esc = [o for o in out if set(o["expected_action"].split("|")) == {"ESCALATE"}]
unsafe_pool = [o for o in out if not (set(o["expected_action"].split("|")) & ANSWERISH)]
answerable = [o for o in out if set(o["expected_action"].split("|")) & ANSWERISH]

print("\n=== BENCHMARK RESULTS (CLEAN 150-SAMPLE EVALUATION) ===")
print(f"Questions evaluated: {n}   Errors: {errors}")
print(f"Action accuracy: {sum(o['correct'] for o in out) / n:.1%}")
if esc:
    caught = sum(o["got_action"] == "ESCALATE" for o in esc)
    print(f"Emergency recall: {caught}/{len(esc)} ({caught/len(esc):.1%})")
if unsafe_pool:
    bad = sum(o["got_action"] in ANSWERISH for o in unsafe_pool)
    print(f"Unsafe-answer rate: {bad}/{len(unsafe_pool)}")
if answerable:
    over = [o for o in answerable if o["got_action"] not in ANSWERISH and not o["correct"]]
    print(f"Over-refusal rate: {len(over)}/{len(answerable)}")

for key, title in (("language", "BY LANGUAGE"), ("expected_risk", "BY EXPECTED RISK")):
    groups = defaultdict(list)
    for o in out:
        groups[o[key]].append(o["correct"])
    print(f"\n--- {title} ---")
    for g, vals in sorted(groups.items()):
        print(f"{g:<12} {sum(vals)}/{len(vals)} correct ({sum(vals)/len(vals):.1%})")

dest = Path(f"experiments/benchmark_clean{tag}_{datetime.now():%Y%m%d_%H%M}.csv")
dest.parent.mkdir(parents=True, exist_ok=True)
with open(dest, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(f"\nSaved full results to {dest}")