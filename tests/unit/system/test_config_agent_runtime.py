import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

import hivememory.config as config_package
from hivememory.config.app import (
    HiveMemoryConfig,
    config_differences,
    load_default_config,
    provider_credentials_from_env,
)
from hivememory.config.files import DATA_ROOT_ENV
from hivememory.config.memory_compiler import CascadeContextStrategyConfig
from hivememory.config.patchouli import MemoryRetrievalConfig
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


@pytest.mark.parametrize("file_name", ["config.yaml", "config.local.yaml"])
def test_config_file_rejects_unknown_top_level_sections(tmp_path, monkeypatch, file_name):
    """配置文件与个人覆盖文件中已删除或写错的顶层段让加载失败，不随根配置的宽松规则被丢弃。"""
    (tmp_path / file_name).write_text("passive_ingress:\n  dedup_ttl_seconds: 300\n", "utf-8")
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(tmp_path / "config.yaml"))

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


def test_local_overrides_reject_saved_secrets(tmp_path, monkeypatch):
    """个人覆盖文件同样不保存密钥。"""
    (tmp_path / "config.local.yaml").write_text(
        "patchouli:\n  storage:\n    api_key: qdrant-in-file\n", encoding="utf-8"
    )
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(tmp_path / "config.yaml"))

    with pytest.raises(ValueError, match=r"patchouli\.storage\.api_key"):
        HiveMemoryConfig()


def test_local_overrides_merge_between_config_file_and_environment(tmp_path, monkeypatch):
    """覆盖文件按层级合并：只替换写出的叶子，同段未写的字段仍取 config.yaml，环境变量仍优先。

    默认值不含覆盖文件。
    """
    (tmp_path / "config.yaml").write_text(
        "alice:\n  runtime:\n    max_loop_iterations: 4\n    default_context_window: 20000\n",
        encoding="utf-8",
    )
    (tmp_path / "config.local.yaml").write_text(
        "alice:\n  runtime:\n    max_loop_iterations: 6\nlogging:\n  level: DEBUG\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HIVEMEMORY_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("HIVEMEMORY__LOGGING__LEVEL", "WARNING")

    config = HiveMemoryConfig()

    assert config.alice.runtime.max_loop_iterations == 6
    assert config.alice.runtime.default_context_window == 20000
    assert config.logging.level == "WARNING"
    assert load_default_config().alice.runtime.max_loop_iterations == 4


def test_local_overrides_are_read_from_the_data_root(tmp_path, monkeypatch):
    """指定数据根时，个人覆盖文件在数据根的 configs 下，默认值仍取随程序发布的 config.yaml。"""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.local.yaml").write_text(
        "alice:\n  runtime:\n    max_loop_iterations: 7\n", encoding="utf-8"
    )
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.delenv("HIVEMEMORY_CONFIG_PATH", raising=False)

    assert HiveMemoryConfig().alice.runtime.max_loop_iterations == 7
    assert load_default_config().alice.runtime.max_loop_iterations != 7


def test_config_differences_write_a_switched_strategy_whole():
    """切换了类型的策略整段计入差异：只写变化的叶子会让旧类型的字段与新类型合并而装载失败。"""
    base = HiveMemoryConfig().model_dump(mode="json")
    target = HiveMemoryConfig(
        memory_compiler={"retrieval_context": {"strategy": {"type": "cascade"}}}
    ).model_dump(mode="json")

    overrides = config_differences(base, target)

    strategy = overrides["memory_compiler"]["retrieval_context"]["strategy"]
    assert strategy == target["memory_compiler"]["retrieval_context"]["strategy"]
    assert HiveMemoryConfig(**overrides).memory_compiler.retrieval_context.strategy.type == (
        "cascade"
    )


def test_config_differences_keep_the_type_of_a_changed_strategy():
    """类型不变的策略只写变化的叶子，但带上 type：下层配置缺少这一段时，覆盖仍能确定结构。"""
    base = HiveMemoryConfig().model_dump(mode="json")
    target = HiveMemoryConfig(
        patchouli={"retrieval": {"retriever": {"type": "hybrid", "top_k": 8}}}
    ).model_dump(mode="json")

    overrides = config_differences(base, target)

    assert overrides == {"patchouli": {"retrieval": {"retriever": {"type": "hybrid", "top_k": 8}}}}
    retrieval = MemoryRetrievalConfig.model_validate(overrides["patchouli"]["retrieval"])
    assert retrieval.retriever.top_k == 8


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

    # discriminated union 能解析 cascade 分支且字段取到对应的值
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
