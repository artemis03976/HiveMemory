"""Tests for I18nConfig integration with HiveMemoryConfig."""

import pytest

from hivememory.system.config import HiveMemoryConfig, I18nConfig


class TestI18nConfig:
    def test_defaults(self):
        config = I18nConfig()
        assert config.default_language == "zh"
        assert config.fallback_language == "en"
        assert config.supported_languages == ["zh", "en"]

    def test_override(self):
        config = I18nConfig(default_language="en", fallback_language="zh")
        assert config.default_language == "en"
        assert config.fallback_language == "zh"


class TestHiveMemoryConfigI18n:
    def test_default_i18n_field(self):
        config = HiveMemoryConfig()
        assert config.i18n.default_language == "zh"
        assert config.i18n.fallback_language == "en"
        assert config.i18n.supported_languages == ["zh", "en"]

    def test_override_via_constructor(self):
        config = HiveMemoryConfig(i18n={"default_language": "en"})
        assert config.i18n.default_language == "en"
        assert config.i18n.fallback_language == "en"
