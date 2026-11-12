"""组合根的装配：进程默认语言，共享 LLM 服务与各组件独立的请求参数，以及插件代为安装的开关。"""

from types import SimpleNamespace

import pytest

from hivememory.config.app import HiveMemoryConfig
from hivememory.core.models.model_definition import ModelSelection
from hivememory.i18n import get_default_language, set_default_language
from hivememory.system.assembler import SystemAssembler


def test_assemble_sets_process_default_language_from_runtime_config():
    set_default_language("zh")
    config = HiveMemoryConfig(runtime_events={"enabled": False}, i18n={"default_language": "en"})

    SystemAssembler(config).assemble()

    assert get_default_language() == "en"


def test_gateway_and_librarian_get_request_params_from_their_own_config():
    """两个组件的生成参数分别来自 ``gateway.llm`` 与 ``patchouli.llm``，覆盖模型默认值。"""
    config = HiveMemoryConfig(runtime_events={"enabled": False})
    config.gateway.llm = ModelSelection(model_id="deepseek-chat", temperature=0.0, max_tokens=512)
    config.patchouli.llm = ModelSelection(
        model_id="deepseek-chat", temperature=0.3, max_tokens=8192
    )
    assembler = SystemAssembler(config)

    llm = assembler._build_llm(assembler._build_registries())

    assert llm.gateway_params is not None
    assert llm.librarian_params is not None
    assert (llm.gateway_params.temperature, llm.gateway_params.max_tokens) == (0.0, 512)
    assert (llm.librarian_params.temperature, llm.librarian_params.max_tokens) == (0.3, 8192)


def test_registry_credentials_reach_request_params_without_changing_the_config(monkeypatch):
    """注册表补齐的提供商密钥只进入组件的请求参数，配置对象仍是配置文件的内容。

    主配置接口读写的就是这份配置对象：写回请求参数会让密钥进入 config.yaml。
    """
    monkeypatch.setenv("HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY", "sk-provider-from-env")
    config = HiveMemoryConfig(runtime_events={"enabled": False})
    config.gateway.llm = ModelSelection(model_id="deepseek-chat", max_tokens=512)
    assembler = SystemAssembler(config)

    llm = assembler._build_llm(assembler._build_registries())

    assert llm.gateway_params is not None
    assert llm.gateway_params.api_key == "sk-provider-from-env"
    assert config.gateway.llm == ModelSelection(model_id="deepseek-chat", max_tokens=512)


def test_component_without_model_gets_no_request_params():
    """llm 配置为 None 的组件不取得请求参数，由组件按无 LLM 降级而不是装配失败。"""
    config = HiveMemoryConfig(runtime_events={"enabled": False})
    config.gateway.llm = None
    assembler = SystemAssembler(config)

    llm = assembler._build_llm(assembler._build_registries())

    assert llm.gateway_params is None
    assert llm.librarian_params is not None


@pytest.mark.parametrize("enabled", [True, False])
def test_plugin_installation_follows_the_switch(monkeypatch, enabled):
    """只有打开开关时插件服务才能代为安装：开发实例代为安装会改写日常实例的 harness 绑定与凭据。"""
    # 只替换组合根模块中的 sys：改全局的 sys.platform 会让 shutil.which 在 Linux 上走 Windows 分支。
    monkeypatch.setattr("hivememory.system.assembler.sys", SimpleNamespace(platform="win32"))
    config = HiveMemoryConfig(
        runtime_events={"enabled": False}, system={"plugin_installation_enabled": enabled}
    )

    system = SystemAssembler(config).assemble()

    assert {status.available for status in system.plugin_service.statuses()} == {enabled}
