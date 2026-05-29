"""Tests for hivememory.i18n.types module."""

import pytest

from hivememory.i18n.types import (
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    Language,
    normalize_language,
)


class TestLanguageEnum:
    def test_zh_value(self):
        assert Language.ZH == "zh"

    def test_en_value(self):
        assert Language.EN == "en"

    def test_is_str(self):
        assert isinstance(Language.ZH, str)
        assert isinstance(Language.EN, str)


class TestConstants:
    def test_default_language(self):
        assert DEFAULT_LANGUAGE == Language.ZH

    def test_fallback_language(self):
        assert FALLBACK_LANGUAGE == Language.EN


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("zh", Language.ZH),
            ("zh-cn", Language.ZH),
            ("zh-hans", Language.ZH),
            ("cn", Language.ZH),
            ("chinese", Language.ZH),
            ("en", Language.EN),
            ("en-us", Language.EN),
            ("en-gb", Language.EN),
            ("english", Language.EN),
        ],
    )
    def test_known_aliases(self, input_val, expected):
        assert normalize_language(input_val) == expected

    @pytest.mark.parametrize("input_val", ["ZH", "Zh", "EN", "En", "CHINESE", "English"])
    def test_case_insensitive(self, input_val):
        result = normalize_language(input_val)
        assert result is not None

    def test_none_input(self):
        assert normalize_language(None) is None

    def test_empty_string(self):
        assert normalize_language("") is None

    def test_whitespace_only(self):
        assert normalize_language("   ") is None

    @pytest.mark.parametrize("input_val", ["fr", "ja", "de", "unknown", "123"])
    def test_unknown_returns_none(self, input_val):
        assert normalize_language(input_val) is None

    def test_language_enum_passthrough(self):
        assert normalize_language(Language.ZH) == Language.ZH
        assert normalize_language(Language.EN) == Language.EN

    def test_whitespace_trimmed(self):
        assert normalize_language("  zh  ") == Language.ZH
        assert normalize_language(" en ") == Language.EN
