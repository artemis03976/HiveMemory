"""Tests for hivememory.i18n.resolver module."""

import pytest

from hivememory.i18n import Language, get_default_language, resolve_language, set_default_language


@pytest.fixture(autouse=True)
def reset_i18n_default_language():
    set_default_language("zh")
    yield
    set_default_language("zh")


class TestResolveLanguage:
    def test_all_none_returns_default_fallback(self):
        assert resolve_language() == Language.ZH

    def test_module_default_over_custom_fallback(self):
        assert resolve_language(fallback=Language.EN) == Language.ZH

    def test_explicit_wins(self):
        result = resolve_language(
            explicit="en",
            profile_language="zh",
            component_language="zh",
        )
        assert result == Language.EN

    def test_profile_over_component(self):
        result = resolve_language(
            profile_language="en",
            component_language="zh",
        )
        assert result == Language.EN

    def test_component_over_default(self):
        result = resolve_language(
            component_language="en",
        )
        assert result == Language.EN

    def test_module_default_over_fallback(self):
        set_default_language("en")

        assert get_default_language() == Language.EN
        assert resolve_language(fallback=Language.ZH) == Language.EN

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

    def test_all_invalid_returns_module_default(self):
        result = resolve_language(
            explicit="fr",
            profile_language="ja",
            component_language="de",
            fallback=Language.EN,
        )
        assert result == Language.ZH
