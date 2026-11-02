import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import hivememory.config as config_package
from hivememory.config.app import HiveMemoryConfig, provider_credentials_from_env
from hivememory.config.memory_compiler import CascadeContextStrategyConfig
from hivememory.config.shared import ProviderCredentials, SharedConfig


def _config_section_models() -> list[type[BaseModel]]:
    models = []
    for module_info in pkgutil.iter_modules(config_package.__path__):
        module = importlib.import_module(f"{config_package.__name__}.{module_info.name}")
        models.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == module.__name__
            and obj is not HiveMemoryConfig
        )
    return models


def test_every_config_section_rejects_unknown_keys():
    """写错或已删除的键必须让装配失败，而不是被静默丢弃。

    根配置除外：它同时接收环境变量与 .env，其中有不属于配置树的变量。
    """
    lenient = [
        model.__qualname__
        for model in _config_section_models()
        if model.model_config.get("extra") != "forbid"
    ]

    assert lenient == []


def test_config_file_rejects_unknown_top_level_sections(tmp_path, monkeypatch):
    """配置文件中已删除或写错的顶层段让加载失败，不随根配置的宽松规则被丢弃。"""
    path = tmp_path / "stale.yaml"
    path.write_text("passive_ingress:\n  dedup_ttl_seconds: 300\n", encoding="utf-8")
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(path))

    with pytest.raises(ValueError, match="passive_ingress"):
        HiveMemoryConfig()


def test_shared_llm_section_fails_with_its_new_locations():
    """组件模型配置移到了各自子系统下；旧位置让加载失败，并指出两段的新位置。"""
    with pytest.raises(ValueError, match=r"gateway\.llm.*patchouli\.llm"):
        SharedConfig.model_validate({"llm": {"gateway": {"model_id": "deepseek-chat"}}})


def test_config_file_rejects_saved_secrets(tmp_path, monkeypatch):
    """配置文件受版本控制：写入了密钥的字段让加载失败并指出字段，留空的不受影响。"""
    path = tmp_path / "secrets.yaml"
    path.write_text(
        "gateway:\n  llm:\n    api_key: sk-in-file\n"
        "patchouli:\n  storage:\n    api_key: qdrant-in-file\n"
        "    host: 127.0.0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(path))

    with pytest.raises(ValueError) as raised:
        HiveMemoryConfig()

    assert "gateway.llm.api_key" in str(raised.value)
    assert "patchouli.storage.api_key" in str(raised.value)
    assert "sk-in-file" not in str(raised.value)


def test_config_file_accepts_empty_secret_fields(tmp_path, monkeypatch):
    """留空的 api_key 不算写入密钥，照常加载（运行时的值仍可来自环境变量）。"""
    path = tmp_path / "empty-secrets.yaml"
    path.write_text("patchouli:\n  storage:\n    api_key: ''\n", encoding="utf-8")
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(path))

    HiveMemoryConfig()


def test_provider_credentials_come_from_environment(monkeypatch):
    """提供商凭证按 provider 名小写归组，交给 Provider 注册表，不进入主配置。"""
    monkeypatch.setenv("HIVEMEMORY__PROVIDERS__PROBEPROV__API_KEY", "sk-probe")
    monkeypatch.setenv("HIVEMEMORY__PROVIDERS__PROBEPROV__API_BASE", "https://probe.invalid")

    credentials = provider_credentials_from_env()

    assert credentials["probeprov"] == ProviderCredentials(
        api_key="sk-probe", api_base="https://probe.invalid"
    )
    assert "providers" not in HiveMemoryConfig().shared.model_dump()


def test_memory_compiler_config_accepts_cascade_strategy():
    config = HiveMemoryConfig(
        memory_compiler={
            "retrieval_context": {
                "strategy": {
                    "type": "cascade",
                    "max_memory_tokens": 123,
                    "full_payload_count": 2,
                }
            }
        }
    )

    # discriminated union 能解析 cascade 分支且字段落位
    strategy = config.memory_compiler.retrieval_context.strategy
    assert isinstance(strategy, CascadeContextStrategyConfig)
    assert strategy.type == "cascade"


def test_nested_env_vars_override_legacy_aliases(monkeypatch):
    """旧式别名映射到组件配置的新位置，同名的嵌套环境变量优先。"""
    monkeypatch.setenv("HIVEMEMORY__LLM__LIBRARIAN__MAX_TOKENS", "4096")
    monkeypatch.setenv("HIVEMEMORY__LLM__GATEWAY__TEMPERATURE", "0.5")
    monkeypatch.setenv("HIVEMEMORY__GATEWAY__LLM__TEMPERATURE", "0.2")

    config = HiveMemoryConfig()

    assert config.patchouli.llm.max_tokens == 4096
    assert config.gateway.llm.temperature == 0.2


def test_runtime_budget_fields_are_loaded_from_configuration(tmp_path, monkeypatch):
    """发布配置中的 runtime 段同时承载循环上限与输入预算参数。"""
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "alice:\n  runtime:\n    max_loop_iterations: 4\n"
        "    default_context_window: 20000\n    safety_margin: 500\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(path))
    config = HiveMemoryConfig().alice.runtime

    assert config.max_loop_iterations == 4
    assert config.default_context_window == 20000
    assert config.safety_margin == 500
