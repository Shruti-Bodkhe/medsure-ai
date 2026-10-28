from app.agents.risk import assess_risk, parse_llm_json


def test_parse_good_json():
    out = parse_llm_json('{"level": "HIGH", "category": "medicine", '
                         '"needs_clarification": "true", "reason": "dose change"}')
    assert out["level"] == "high"
    assert out["needs_clarification"] is True


def test_parse_json_inside_text():
    out = parse_llm_json('Sure! {"level": "low", "category": "general_info", '
                         '"needs_clarification": false, "reason": "definition"} Done.')
    assert out["level"] == "low"


def test_parse_rejects_bad_level():
    assert parse_llm_json('{"level": "banana"}') is None


def test_parse_rejects_garbage():
    assert parse_llm_json("I cannot help with that") is None
    assert parse_llm_json("") is None


def test_rules_short_circuit_without_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called for rule-detected emergencies")
    monkeypatch.setattr("app.agents.risk.llm.generate", boom)
    result = assess_risk("my father has chest pain")
    assert result.level == "emergency" and result.source == "rules"


def test_llm_failure_is_unknown_not_low(monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr("app.agents.risk.llm.generate", fail)
    result = assess_risk("How does dengue spread?")
    assert result.level == "unknown"
    assert result.needs_clarification is True