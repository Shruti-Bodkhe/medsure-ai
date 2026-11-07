import pytest

from app.agents import language
from app.agents.language import (detect_language, numbers_match, to_english,
                                 translate_message)


@pytest.mark.parametrize("text,lang", [
    ("How does dengue spread?", "en"),
    ("What should I do after a dog bite?", "en"),
    ("डेंगू के लक्षण क्या होते हैं?", "hi"),
    ("मला श्वास घ्यायला त्रास होतोय", "mr"),
    ("डेंग्यू कसा पसरतो?", "mr"),
    ("dengue ke symptoms kya hain", "hinglish"),
    ("mala shwas ghyayla tras hotoy", "mr"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_numbers_match():
    assert numbers_match("Wash for 15 minutes. Call 112 [1]", "15 मिनट धोएं। 112 पर कॉल करें [1]")
    assert numbers_match("wash 15 minutes", "१५ मिनट धुवा")
    assert not numbers_match("wash 15 minutes", "5 मिनट धुवा")
    assert not numbers_match("Call 112", "Call")


def test_translate_ok(monkeypatch):
    monkeypatch.setattr(language.llm, "generate",
                        lambda *a, **k: '{"translation": "१५ मिनिटे धुवा [1]"}')
    text, status = translate_message("Wash for 15 minutes [1]", "mr")
    assert status == "translated" and "धुवा" in text


def test_translate_number_mismatch_falls_back(monkeypatch):
    monkeypatch.setattr(language.llm, "generate",
                        lambda *a, **k: '{"translation": "५ मिनिटे धुवा [1]"}')
    original = "Wash for 15 minutes [1]"
    text, status = translate_message(original, "mr")
    assert status == "number_mismatch" and text == original


def test_translate_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(language.llm, "generate", boom)
    text, status = translate_message("Call 112", "hi")
    assert status == "failed" and text == "Call 112"


def test_to_english_failure_keeps_original(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(language.llm, "generate", boom)
    assert to_english("डेंगू क्या है", "hi") == ("डेंगू क्या है", False)
    assert to_english("hello", "en") == ("hello", True)