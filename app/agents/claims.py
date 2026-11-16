import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from app.core import llm

VALID_LABELS = {"supported", "contradicted", "not_found", "unverified"}
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and",
    "in", "on", "for", "with", "as", "by", "at", "from", "or", "that", "this", "it",
    "can", "should", "may", "might", "will", "if", "your", "you", "have", "has"
}

@dataclass
class ClaimResult:
    text: str
    verdict: str = "supported"
    passages: List[int] = field(default_factory=list)
    reason: str = ""

    @property
    def label(self) -> str:
        return self.verdict

    @property
    def claim(self) -> str:
        return self.text

@dataclass
class VerificationReport:
    claims: List[ClaimResult] = field(default_factory=list)
    overall_score: float = 1.0
    passed: bool = True
    error: Optional[str] = None

    @property
    def support_rate(self) -> Optional[float]:
        if not self.claims:
            return None
        sup = sum(1 for c in self.claims if c.verdict == "supported")
        return round(sup / len(self.claims), 4)

    @property
    def unsupported(self) -> List[ClaimResult]:
        return [c for c in self.claims if c.verdict != "supported"]

def parse_claims(text: str) -> Optional[List[str]]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        raw = data.get("claims", [])
        if not isinstance(raw, list):
            return None
        return [str(c).strip() for c in raw if str(c).strip()]
    except Exception:
        return None

def parse_verdicts(text: str, max_id: int) -> Dict[int, dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        results = data.get("results", [])
        out = {}
        for item in results:
            cid = item.get("id")
            label = str(item.get("label", "")).strip().lower()
            if isinstance(cid, int) and 1 <= cid <= max_id and label in VALID_LABELS:
                out[cid] = {
                    "label": label,
                    "passages": item.get("passages", []),
                    "reason": str(item.get("reason", ""))
                }
        return out
    except Exception:
        return {}

def missing_numbers(claim_text: str, passage_text: str) -> List[str]:
    cleaned_claim = re.sub(r"\[\d+\]", "", claim_text)
    claim_nums = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_claim)
    passage_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", passage_text))
    return [n for n in claim_nums if n not in passage_nums]

def extract_claims(answer: str) -> List[str]:
    cleaned = re.sub(r"\*Note:.*$", "", answer, flags=re.DOTALL).strip()
    lines = cleaned.replace("\n", ".").split(".")
    parts = [re.sub(r"^[-*\d.)\s]+", "", s).strip() for s in lines if len(s.strip()) > 12]
    return parts[:8]

def verify_claims(claims_list: List[str], passages: List[str]) -> Dict[int, dict]:
    joined_passages = " ".join(passages).lower()
    p_tokens = set(re.findall(r"[a-z0-9\u0900-\u097F]{3,}", joined_passages)) - STOPWORDS
    out = {}
    for idx, c in enumerate(claims_list, start=1):
        c_tokens = set(re.findall(r"[a-z0-9\u0900-\u097F]{3,}", c.lower())) - STOPWORDS
        if not c_tokens:
            out[idx] = {"label": "supported", "passages": [1], "reason": "general transitional phrase"}
            continue
        overlap = len(c_tokens & p_tokens) / max(len(c_tokens), 1)
        has_citation = bool(re.search(r"\[\d+\]", c))
        if overlap >= 0.18 or has_citation:
            out[idx] = {"label": "supported", "passages": [1], "reason": f"lexical evidence overlap ({overlap:.0%})"}
        else:
            out[idx] = {"label": "not_found", "passages": [], "reason": "insufficient overlap with retrieved passages"}
    return out

def check_answer(answer: str, passages: List[str]) -> VerificationReport:
    raw_claims = extract_claims(answer)
    if not raw_claims:
        return VerificationReport(claims=[], overall_score=1.0, passed=True)

    joined_passages = " ".join(passages if isinstance(passages, list) else [str(passages)])
    try:
        verdict_map = verify_claims(raw_claims, passages)
    except Exception as e:
        unv = [ClaimResult(text=c, verdict="unverified", reason=str(e)) for c in raw_claims]
        return VerificationReport(claims=unv, overall_score=0.0, passed=False, error=str(e))

    results = []
    for idx, c_text in enumerate(raw_claims, start=1):
        v = verdict_map.get(idx, {"label": "unverified", "passages": [], "reason": "not checked"})
        label = v["label"]
        reason = v["reason"]
        if label == "supported":
            miss = missing_numbers(c_text, joined_passages)
            if miss:
                label = "not_found"
                reason = f"Numeric value(s) {miss} not present in retrieved clinical evidence"
        results.append(ClaimResult(text=c_text, verdict=label, passages=v.get("passages", []), reason=reason))

    sup_count = sum(1 for r in results if r.verdict == "supported")
    score = round(sup_count / len(results), 4) if results else 1.0
    return VerificationReport(claims=results, overall_score=score, passed=(score >= 0.65))
