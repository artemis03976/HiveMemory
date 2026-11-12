"""语言配置 ``i18n.language``：只接受 zh 与 en，旧键指向新键，构造配置不改变进程的语言。"""

import pytest
from pydantic import ValidationError

from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import Language, get_language, set_language


def test_constructing_config_keeps_process_language():
    set_language(Language.EN)

    config = HiveMemoryConfig(i18n={"language": "zh"})

    assert config.i18n.language == Language.ZH
    assert get_language() == Language.EN


def test_unknown_language_is_rejected():
    with pytest.raises(ValidationError):
        HiveMemoryConfig(i18n={"language": "fr"})


def test_old_default_language_key_points_to_the_new_key(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__I18N__DEFAULT_LANGUAGE", "en")

    with pytest.raises(ValidationError, match="i18n.language"):
        HiveMemoryConfig()
