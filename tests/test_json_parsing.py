import pytest

from maner.llm.parsing import JsonParseError, parse_llm_json


def test_parse_pure_json() -> None:
    text = '{"a": 1, "b": [2, 3]}'
    parsed = parse_llm_json(text)
    assert parsed["a"] == 1
    assert parsed["b"] == [2, 3]


def test_parse_json_with_wrapping_text() -> None:
    text = 'analysis... {"ok": true, "data": {"x": 1}} trailing text'
    parsed = parse_llm_json(text)
    assert parsed["ok"] is True
    assert parsed["data"]["x"] == 1


def test_parse_invalid_without_json_should_fail() -> None:
    text = "there is no json payload"
    with pytest.raises(JsonParseError):
        parse_llm_json(text)


def test_parse_json_with_invalid_then_valid_object() -> None:
    text = "prefix {invalid json} middle {\"ok\": true, \"v\": 2} suffix"
    parsed = parse_llm_json(text)
    assert parsed["ok"] is True
    assert parsed["v"] == 2


def test_parse_json_with_code_fence_and_trailing_comma() -> None:
    text = "```json\n{\"ok\": true, \"arr\": [1, 2,],}\n```"
    parsed = parse_llm_json(text)
    assert parsed["ok"] is True
    assert parsed["arr"] == [1, 2]
