"""Tests for hivememory.i18n.types module."""

import pytest

from hivememory.i18n.types import (
    Language,
    normalize_language,
)


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

    @pytest.mark.parametrize("input_val", ["ZH", "Zh", "CHINESE"])
    def test_case_insensitive_zh(self, input_val):
        assert normalize_language(input_val) == Language.ZH

    @pytest.mark.parametrize("input_val", ["EN", "En", "English"])
    def test_case_insensitive_en(self, input_val):
        assert normalize_language(input_val) == Language.EN

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
