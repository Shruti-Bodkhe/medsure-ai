from app.agents.responsive import parse_verdict


def test_parse_verdict_ok():
    assert parse_verdict('{"verdict": "YES", "reason": "fits"}') == ("yes", "fits")


def test_parse_verdict_inside_text():
    assert parse_verdict('Sure {"verdict": "no", "reason": "adults"} ok')[0] == "no"


def test_parse_verdict_rejects_bad():
    assert parse_verdict('{"verdict": "maybe"}') is None
    assert parse_verdict("nothing") is None