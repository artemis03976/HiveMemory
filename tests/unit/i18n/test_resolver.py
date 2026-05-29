"""Tests for hivememory.i18n.resolver module."""

import pytest

from hivememory.i18n import Language, resolve_language


class TestResolveLanguage:
    def test_all_none_returns_default_fallback(self):
        assert resolve_language() == Language.ZH

    def test_custom_fallback(self):
        assert resolve_language(fallback=Language.EN) == Language.EN

    def test_explicit_wins(self):
        result = resolve_language(
            explicit="en",
            profile_language="zh",
            component_language="zh",
            default_language="zh",
        )
        assert result == Language.EN

    def test_profile_over_component(self):
        result = resolve_language(
            profile_language="en",
            component_language="zh",
            default_language="zh",
        )
        assert result == Language.EN

    def test_component_over_default(self):
        result = resolve_language(
            component_language="en",
            default_language="zh",
        )
        assert result == Language.EN

    def test_default_over_fallback(self):
        result = resolve_language(
            default_language="en",
            fallback=Language.ZH,
        )
        assert result == Language.EN

    def test_skips_none_values(self):
        result = resolve_language(
            explicit=None,
            profile_language=None,
            component_language="en",
        )
        assert result == Language.EN

    def test_skips_invalid_values(self):
        result = resolve_language(
            explicit="fr",
            profile_language="unknown",
            component_language="en",
        )
        assert result == Language.EN

    def test_normalizes_aliases(self):
        result = resolve_language(explicit="zh-cn")
        assert result == Language.ZH

        result = resolve_language(profile_language="english")
        assert result == Language.EN

    def test_all_invalid_returns_fallback(self):
        result = resolve_language(
            explicit="fr",
            profile_language="ja",
            component_language="de",
            default_language="unknown",
            fallback=Language.EN,
        )
        assert result == Language.EN
