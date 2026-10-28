import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.agents.claims import VerificationReport, check_answer
from app.agents.risk import RiskResult, assess_risk
from app.baselines import rag_only
from app.governance.policy import (ABSTAIN, UNAVAILABLE_MSG, Decision, PolicyConfig,
                                   early_decision, final_decision)

AUDIT = Path("data/audit/audit.jsonl")


@dataclass
class PipelineResult:
    question: str
    risk: RiskResult
    decision: Decision
    draft: str = ""
    sources: list = field(default_factory=list)
    report: Optional[VerificationReport] = None
    seconds: float = 0.0


def _audit(result: PipelineResult) -> None:
    """Development audit log. The privacy agent must redact the question before any real use."""
    try:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(result)
        row["time"] = datetime.now().isoformat(timespec="seconds")
        with open(AUDIT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run(question: str, retriever, cfg: Optional[PolicyConfig] = None) -> PipelineResult:
    cfg = cfg or PolicyConfig()
    t0 = time.time()
    risk = assess_risk(question)

    early = early_decision(question, risk)
    if early:
        result = PipelineResult(question, risk, early, seconds=time.time() - t0)
        _audit(result)
        return result

    try:
        rag = rag_only(question, retriever)
    except Exception:
        decision = Decision(ABSTAIN, "ANSWER_GENERATION_FAILED", UNAVAILABLE_MSG)
        result = PipelineResult(question, risk, decision, seconds=time.time() - t0)
        _audit(result)
        return result

    report = check_answer(rag["answer"], rag["passages"])
    decision = final_decision(risk, report, len(rag["passages"]), rag["answer"], cfg)
    result = PipelineResult(question, risk, decision, rag["answer"], rag["sources"],
                            report, time.time() - t0)
    _audit(result)
    return result