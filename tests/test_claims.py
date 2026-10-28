from app.agents import claims
from app.agents.claims import (check_answer, missing_numbers, parse_claims,
                               parse_verdicts)

PASSAGES = ["Wash the wound with soap and running water for 15 minutes."]


def test_parse_claims():
    assert parse_claims('{"claims": ["A", " B ", ""]}') == ["A", "B"]
    assert parse_claims("nonsense") is None


def test_parse_verdicts_validates():
    text = ('{"results": [{"id": 1, "label": "supported", "passages": [1], "reason": "ok"},'
            '{"id": 2, "label": "banana"}, {"id": 9, "label": "supported"}]}')
    out = parse_verdicts(text, 2)
    assert list(out) == [1]


def test_missing_numbers():
    assert missing_numbers("Wash for 5 to 10 minutes", PASSAGES[0]) == ["5", "10"]
    assert missing_numbers("Wash for 15 minutes", PASSAGES[0]) == []


def test_number_check_overrides_a_wrong_supported(monkeypatch):
    monkeypatch.setattr(claims, "extract_claims", lambda a: ["Wash the wound for 5 minutes"])
    monkeypatch.setattr(claims, "verify_claims", lambda c, p: {
        1: {"label": "supported", "passages": [1], "reason": "washing"}})
    report = check_answer("anything", PASSAGES)
    assert report.claims[0].label == "not_found"


def test_verifier_failure_is_unverified_not_supported(monkeypatch):
    monkeypatch.setattr(claims, "extract_claims", lambda a: ["Some claim"])

    def boom(c, p):
        raise RuntimeError("503")
    monkeypatch.setattr(claims, "verify_claims", boom)
    report = check_answer("anything", PASSAGES)
    assert report.claims[0].label == "unverified"
    assert report.error