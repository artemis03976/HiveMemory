"""Tests for HiveMemoryConfig i18n 集成。"""

import pytest

from hivememory.i18n import get_default_language, set_default_language
from hivememory.system.config import HiveMemoryConfig


@pytest.fixture(autouse=True)
def _reset_default_language():
    set_default_language("zh")
    yield
    set_default_language("zh")


class TestHiveMemoryConfigI18n:
    def test_syncs_default_language_to_global_resolver(self):
        config = HiveMemoryConfig(i18n={"default_language": "en"})
        assert get_default_language() == "en"
        assert config.i18n.default_language == "en"

    def test_keeps_default_language_zh(self):
        config = HiveMemoryConfig()
        assert get_default_language() == "zh"
