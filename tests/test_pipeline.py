from app import pipeline
from app.agents.risk import RiskResult


def test_no_answer_token_becomes_abstain(monkeypatch):
    monkeypatch.setattr(pipeline, "assess_risk",
                        lambda q: RiskResult("high", "medicine", False, "t", "llm"))
    monkeypatch.setattr(pipeline, "rag_only",
                        lambda *a, **k: {"answer": "NO_ANSWER", "sources": [], "passages": ["x"]})

    def must_not_run(*a, **k):
        raise AssertionError("claim checking must be skipped when there is no answer")
    monkeypatch.setattr(pipeline, "check_answer", must_not_run)
    monkeypatch.setattr(pipeline, "_audit", lambda r: None)

    result = pipeline.run("Can I stop my medicine?", retriever=None)
    assert result.decision.action == "ABSTAIN"
    assert result.decision.reason_code == "NO_RELEVANT_EVIDENCE"