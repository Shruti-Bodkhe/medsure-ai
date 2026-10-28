import pytest

from app.agents.emergency import detect

SHOULD_TRIGGER = [
    ("Chest pain and my left arm hurts, what should I do?", "cardiac"),
    ("my father has chest tightness and is sweating", "cardiac"),
    ("seene me dard ho raha hai", "cardiac"),
    ("मुझे सीने में दर्द हो रहा है", "cardiac"),
    ("मला छातीत दुखतंय", "cardiac"),
    ("I can't breathe properly", "breathing"),
    ("मुझे सांस लेने में तकलीफ है", "breathing"),
    ("What should I do after a snake bite?", "snakebite"),
    ("saanp ne kata hai", "snakebite"),
    ("my child drank kerosene", "poisoning"),
    ("bleeding won't stop", "severe_bleeding"),
    ("she is unconscious", "unconscious_seizure"),
    ("I want to die", "self_harm"),
    ("What should I do, my mother has chest pain", "cardiac"),
    ("my chest feels heavy and I am sweating", "cardiac"),
    ("my husband is not waking up", "unconscious_seizure"),
    ("I feel like ending it all", "self_harm"),
    ("How can I help my father, he is unconscious", "unconscious_seizure"),
    ("मैं आत्महत्या के बारे में सोच रहा हूँ", "self_harm"),
]

SHOULD_NOT_TRIGGER = [
    "How does dengue spread?",
    "What is a heart attack?",
    "How can I prevent snake bites?",
    "What are the symptoms of a stroke?",
    "Can I stop my diabetes medicine if my sugar is normal now?",
    "डेंगू के लक्षण क्या होते हैं?",
    "What should I do after a dog bite?",
    "What is suicide prevention?",
]


@pytest.mark.parametrize("text,category", SHOULD_TRIGGER)
def test_triggers(text, category):
    result = detect(text)
    assert result.is_emergency, f"missed: {text}"
    assert result.category == category, f"{text} -> {result.category}"


@pytest.mark.parametrize("text", SHOULD_NOT_TRIGGER)
def test_no_false_alarm(text):
    assert not detect(text).is_emergency, f"false alarm: {text}"