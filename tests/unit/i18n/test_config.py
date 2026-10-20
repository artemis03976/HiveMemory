"""构造根配置不改变进程默认语言。"""

from hivememory.config.app import HiveMemoryConfig
from hivememory.i18n import get_default_language, set_default_language


def test_constructing_config_keeps_process_default_language():
    set_default_language("en")

    config = HiveMemoryConfig(i18n={"default_language": "zh"})

    assert config.i18n.default_language == "zh"
    assert get_default_language() == "en"
