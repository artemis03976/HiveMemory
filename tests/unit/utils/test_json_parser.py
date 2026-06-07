import pytest
from pydantic import BaseModel

from hivememory.utils.json_parser import (
    JSONParseError,
    LLMJSONParser,
    parse_llm_json,
    parse_llm_json_many,
    safe_parse_llm_json,
)


class UserModel(BaseModel):
    name: str
    age: int


def test_parse_plain_json_object():
    assert LLMJSONParser().parse('{"name": "Alice", "age": 30}') == {
        "name": "Alice",
        "age": 30,
    }


def test_parse_markdown_json_block():
    raw = """
    Here is the result:
    ```json
    {"ok": true, "items": [1, 2]}
    ```
    """

    assert LLMJSONParser().parse(raw) == {"ok": True, "items": [1, 2]}


def test_parse_extracts_nested_object_from_text():
    raw = 'prefix {"outer": {"inner": "value"}, "items": [1, 2]} suffix'

    assert LLMJSONParser().parse(raw) == {
        "outer": {"inner": "value"},
        "items": [1, 2],
    }


def test_parse_accepts_python_style_dict_when_not_strict():
    raw = "{'ok': True, 'value': None}"

    assert LLMJSONParser().parse(raw) == {"ok": True, "value": None}


def test_strict_parser_rejects_python_style_dict():
    parser = LLMJSONParser(strict=True)

    with pytest.raises(JSONParseError):
        parser.parse("{'ok': True}")


def test_parse_returns_default_on_failure():
    assert LLMJSONParser().parse("not json", default={}) == {}


def test_parse_converts_to_pydantic_model():
    user = LLMJSONParser().parse('{"name": "Bob", "age": 25}', as_model=UserModel)

    assert user == UserModel(name="Bob", age=25)


def test_parse_many_returns_all_json_candidates():
    raw = """
    ```json
    {"id": 1}
    ```
    and then [{"id": 2}, {"id": 3}]
    """

    assert LLMJSONParser().parse_many(raw) == [{"id": 1}, [{"id": 2}, {"id": 3}]]


def test_safe_parse_returns_none_on_failure():
    assert safe_parse_llm_json("not json") is None


def test_convenience_functions_use_default_parser():
    assert parse_llm_json('{"ok": true}') == {"ok": True}
    assert parse_llm_json("bad", default={"fallback": True}) == {"fallback": True}
    assert parse_llm_json_many('{"id": 1}') == [{"id": 1}]
