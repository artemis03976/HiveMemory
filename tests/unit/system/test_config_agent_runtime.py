import importlib
import inspect
import pkgutil

from pydantic import BaseModel

import hivememory.config as config_package
from hivememory.config.app import HiveMemoryConfig
from hivememory.config.memory_compiler import CascadeContextStrategyConfig


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


def test_new_llm_env_vars_override_legacy_aliases(monkeypatch):
    monkeypatch.setenv("HIVEMEMORY__LLM__GATEWAY__API_KEY", "legacy-gateway-key")
    monkeypatch.setenv("HIVEMEMORY__SHARED__LLM__GATEWAY__API_KEY", "new-gateway-key")

    config = HiveMemoryConfig()

    assert config.shared.llm.gateway.api_key == "new-gateway-key"


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
