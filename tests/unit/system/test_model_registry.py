"""
ModelRegistry 单元测试

覆盖范围：
- YAML 加载与持久化（原子写入）
- CRUD：add_model / update_model / delete_model / get_model / list_models
- 默认模型不变量（有且仅有一条 is_default=True）
- 凭证解析优先级：model 显式 > ProviderRegistry > provider_credentials 静态字典 > None
- resolve() / resolve_for_llm_config() / to_llm_config()
- 异常路径：ModelNotFoundError / DuplicateModelIdError
"""

import pytest
import yaml

from hivememory.core.models.model_definition import ModelDefinition
from hivememory.system.config.shared import LLMConfig, ProviderCredentials
from hivememory.system.model_registry import (
    DuplicateModelIdError,
    ModelNotFoundError,
    ModelRegistry,
)


# ---------------------------------------------------------------------------
# 测试 fixtures / 辅助函数
# ---------------------------------------------------------------------------

def _make_model(
    model_id: str = "test-model",
    display_name: str = "Test Model",
    litellm_model: str = "deepseek/deepseek-chat",
    is_default: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 1.0,
    api_key: str | None = None,
    api_base: str | None = None,
    provider: str = "",
) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        display_name=display_name,
        litellm_model=litellm_model,
        is_default=is_default,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_key=api_key,
        api_base=api_base,
        provider=provider,
    )


def _make_registry(
    tmp_path,
    models: list[ModelDefinition] | None = None,
    provider_credentials: dict | None = None,
    provider_registry=None,
) -> ModelRegistry:
    """在临时目录创建 ModelRegistry，可选预填模型。"""
    registry_path = tmp_path / "models.yaml"
    if models is not None:
        data = {"models": [m.model_dump(mode="json") for m in models]}
        registry_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    registry = ModelRegistry(
        registry_path=registry_path,
        provider_credentials=provider_credentials or {},
        provider_registry=provider_registry,
    )
    return registry


# ---------------------------------------------------------------------------
# 加载测试
# ---------------------------------------------------------------------------

class TestLoad:
    def test_empty_registry_when_file_missing(self, tmp_path):
        """文件不存在时以空注册表启动，不抛异常。"""
        registry = ModelRegistry(registry_path=tmp_path / "nonexistent.yaml")
        assert len(registry) == 0

    def test_load_models_from_yaml(self, tmp_path):
        """从 YAML 文件正确加载模型列表。"""
        models = [
            _make_model("m1", is_default=True),
            _make_model("m2"),
        ]
        registry = _make_registry(tmp_path, models=models)
        assert len(registry) == 2
        assert registry.get_model("m1") is not None
        assert registry.get_model("m2") is not None

    def test_provider_auto_derived_from_litellm_model(self, tmp_path):
        """provider 留空时从 litellm_model 前缀自动推导。"""
        model = _make_model("ds", litellm_model="deepseek/deepseek-chat", provider="")
        registry = _make_registry(tmp_path, models=[model])
        loaded = registry.get_model("ds")
        assert loaded is not None
        assert loaded.provider == "deepseek"

    def test_invalid_model_entry_skipped(self, tmp_path):
        """无效的模型定义被跳过，注册表仍可正常运行。"""
        registry_path = tmp_path / "models.yaml"
        # 写入一条缺少必填字段 id 的无效条目 + 一条合法条目
        data = {
            "models": [
                {"display_name": "Bad", "litellm_model": "x"},  # 缺少 id
                {"id": "good", "display_name": "Good", "litellm_model": "openai/gpt-4o"},
            ]
        }
        registry_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        registry = ModelRegistry(registry_path=registry_path)
        assert len(registry) == 1
        assert registry.get_model("good") is not None


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------

class TestQuery:
    def test_get_model_found(self, tmp_path):
        model = _make_model("m1")
        registry = _make_registry(tmp_path, models=[model])
        result = registry.get_model("m1")
        assert result is not None
        assert result.id == "m1"

    def test_get_model_not_found(self, tmp_path):
        registry = _make_registry(tmp_path)
        assert registry.get_model("nonexistent") is None

    def test_list_models_preserves_order(self, tmp_path):
        models = [_make_model(f"m{i}") for i in range(3)]
        registry = _make_registry(tmp_path, models=models)
        ids = [m.id for m in registry.list_models()]
        assert ids == ["m0", "m1", "m2"]

    def test_get_default_model_explicit(self, tmp_path):
        """is_default=True 的模型被正确返回。"""
        models = [
            _make_model("m1"),
            _make_model("m2", is_default=True),
        ]
        registry = _make_registry(tmp_path, models=models)
        default = registry.get_default_model()
        assert default is not None
        assert default.id == "m2"

    def test_get_default_model_falls_back_to_first(self, tmp_path):
        """无 is_default 时退化返回第一条记录。"""
        models = [_make_model("m1"), _make_model("m2")]
        registry = _make_registry(tmp_path, models=models)
        default = registry.get_default_model()
        assert default is not None
        assert default.id == "m1"

    def test_get_default_model_empty_registry(self, tmp_path):
        registry = _make_registry(tmp_path)
        assert registry.get_default_model() is None


# ---------------------------------------------------------------------------
# 写入接口（CRUD）
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_add_model_new(self, tmp_path):
        registry = _make_registry(tmp_path)
        model = _make_model("new-model", temperature=0.3)
        registry.add_model(model)

        loaded = registry.get_model("new-model")
        assert loaded is not None
        assert loaded.temperature == 0.3
        assert loaded.display_name == "Test Model"

    def test_add_model_duplicate_raises(self, tmp_path):
        model = _make_model("m1")
        registry = _make_registry(tmp_path, models=[model])
        with pytest.raises(DuplicateModelIdError):
            registry.add_model(_make_model("m1"))

    def test_add_model_sets_only_one_default(self, tmp_path):
        """新增 is_default=True 的模型时，旧默认模型的标记被清除。"""
        old_default = _make_model("old", is_default=True)
        registry = _make_registry(tmp_path, models=[old_default])

        new_default = _make_model("new", is_default=True)
        registry.add_model(new_default)

        assert registry.get_model("old").is_default is False
        assert registry.get_model("new").is_default is True

    def test_update_model_fields(self, tmp_path):
        model = _make_model("m1", temperature=0.5)
        registry = _make_registry(tmp_path, models=[model])
        updated = registry.update_model("m1", {"temperature": 0.9, "display_name": "New Name"})
        assert updated.temperature == 0.9
        assert updated.display_name == "New Name"

    def test_update_model_not_found_raises(self, tmp_path):
        registry = _make_registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            registry.update_model("ghost", {"temperature": 0.1})

    def test_update_model_set_default_clears_others(self, tmp_path):
        """update 将某模型设为 default 时，其他模型的标记被清除。"""
        models = [_make_model("m1", is_default=True), _make_model("m2")]
        registry = _make_registry(tmp_path, models=models)
        registry.update_model("m2", {"is_default": True})
        assert registry.get_model("m1").is_default is False
        assert registry.get_model("m2").is_default is True

    def test_delete_model_success(self, tmp_path):
        model = _make_model("m1")
        registry = _make_registry(tmp_path, models=[model])
        registry.delete_model("m1")
        assert registry.get_model("m1") is None
        assert len(registry) == 0

    def test_delete_model_not_found_raises(self, tmp_path):
        registry = _make_registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            registry.delete_model("ghost")


# ---------------------------------------------------------------------------
# 持久化测试
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_changes_persisted_to_yaml(self, tmp_path):
        """add_model 后重新加载注册表，新模型应仍然存在。"""
        registry = _make_registry(tmp_path)
        registry.add_model(_make_model("persisted"))

        reloaded = ModelRegistry(registry_path=tmp_path / "models.yaml")
        assert reloaded.get_model("persisted") is not None

    def test_delete_persisted_to_yaml(self, tmp_path):
        model = _make_model("to-delete")
        registry = _make_registry(tmp_path, models=[model])
        registry.delete_model("to-delete")

        reloaded = ModelRegistry(registry_path=tmp_path / "models.yaml")
        assert reloaded.get_model("to-delete") is None

    def test_update_persisted_to_yaml(self, tmp_path):
        model = _make_model("m1", temperature=0.5)
        registry = _make_registry(tmp_path, models=[model])
        registry.update_model("m1", {"temperature": 1.5})

        reloaded = ModelRegistry(registry_path=tmp_path / "models.yaml")
        assert reloaded.get_model("m1").temperature == 1.5


# ---------------------------------------------------------------------------
# 凭证解析
# ---------------------------------------------------------------------------

class TestCredentialResolution:
    def test_model_explicit_api_key_takes_priority(self, tmp_path):
        """模型自身的 api_key 优先于 provider 凭证。"""
        model = _make_model(
            "m1",
            litellm_model="deepseek/deepseek-chat",
            api_key="model-explicit-key",
        )
        creds = {"deepseek": ProviderCredentials(api_key="provider-key")}
        registry = _make_registry(tmp_path, models=[model], provider_credentials=creds)

        llm_config = registry.to_llm_config("m1")
        assert llm_config.api_key == "model-explicit-key"

    def test_provider_credentials_used_when_model_key_is_none(self, tmp_path):
        """model.api_key=None 时从 provider 凭证中补齐。"""
        model = _make_model("m1", litellm_model="openai/gpt-4o", api_key=None)
        creds = {"openai": ProviderCredentials(api_key="provider-openai-key", api_base="https://api.openai.com")}
        registry = _make_registry(tmp_path, models=[model], provider_credentials=creds)

        llm_config = registry.to_llm_config("m1")
        assert llm_config.api_key == "provider-openai-key"
        assert llm_config.api_base == "https://api.openai.com"

    def test_no_credentials_returns_none(self, tmp_path):
        """既无 model key 又无 provider 凭证时，返回 None（litellm 自行从环境变量读取）。"""
        model = _make_model("m1", litellm_model="deepseek/deepseek-chat", api_key=None)
        registry = _make_registry(tmp_path, models=[model])

        llm_config = registry.to_llm_config("m1")
        assert llm_config.api_key is None
        assert llm_config.api_base is None

    def test_provider_registry_injected(self, tmp_path):
        """注入 ProviderRegistry 时，_resolve_credentials 通过它动态查询。"""
        from unittest.mock import MagicMock

        model = _make_model("m1", litellm_model="anthropic/claude-3", api_key=None)

        mock_pr = MagicMock()
        mock_pr.get.return_value = ProviderCredentials(api_key="dynamic-key", api_base=None)
        registry = _make_registry(
            tmp_path,
            models=[model],
            provider_registry=mock_pr,
        )

        llm_config = registry.to_llm_config("m1")
        assert llm_config.api_key == "dynamic-key"


# ---------------------------------------------------------------------------
# resolve() / resolve_for_llm_config()
# ---------------------------------------------------------------------------

class TestResolve:
    def test_resolve_by_id(self, tmp_path):
        model = _make_model("m1", temperature=0.8, max_tokens=2048, top_p=0.9)
        registry = _make_registry(tmp_path, models=[model])

        llm_config, display_name = registry.resolve("m1")
        assert llm_config.model == model.litellm_model
        assert llm_config.temperature == 0.8
        assert llm_config.max_tokens == 2048
        assert llm_config.top_p == 0.9
        assert display_name == model.display_name

    def test_resolve_default(self, tmp_path):
        model = _make_model("m1", is_default=True)
        registry = _make_registry(tmp_path, models=[model])

        llm_config, _ = registry.resolve("default")
        assert llm_config.model == model.litellm_model

    def test_resolve_with_overrides(self, tmp_path):
        """temperature/max_tokens/top_p 覆盖参数优先于模型默认值。"""
        model = _make_model("m1", temperature=0.7, max_tokens=4096, top_p=1.0)
        registry = _make_registry(tmp_path, models=[model])

        llm_config, _ = registry.resolve(
            "m1",
            temperature_override=0.1,
            max_tokens_override=512,
            top_p_override=0.5,
        )
        assert llm_config.temperature == 0.1
        assert llm_config.max_tokens == 512
        assert llm_config.top_p == 0.5

    def test_resolve_not_found_raises(self, tmp_path):
        registry = _make_registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            registry.resolve("ghost")

    def test_resolve_default_empty_registry_raises(self, tmp_path):
        registry = _make_registry(tmp_path)
        with pytest.raises(ModelNotFoundError):
            registry.resolve("default")

    def test_resolve_for_llm_config_with_model_id(self, tmp_path):
        """model_id 存在时补齐 model/api_key/api_base，保留组件的 temperature/max_tokens。"""
        model = _make_model("m1", litellm_model="deepseek/deepseek-chat")
        creds = {"deepseek": ProviderCredentials(api_key="sk-test", api_base="https://api.deepseek.com")}
        registry = _make_registry(tmp_path, models=[model], provider_credentials=creds)

        llm_in = LLMConfig(model_id="m1", temperature=0.1, max_tokens=512)
        llm_out = registry.resolve_for_llm_config(llm_in)

        assert llm_out.model == "deepseek/deepseek-chat"
        assert llm_out.api_key == "sk-test"
        assert llm_out.api_base == "https://api.deepseek.com"
        # 组件的 temperature/max_tokens 被保留
        assert llm_out.temperature == 0.1
        assert llm_out.max_tokens == 512

    def test_resolve_for_llm_config_passthrough_when_no_model_id(self, tmp_path):
        """model_id 为空时直接返回原 LLMConfig（向后兼容路径）。"""
        registry = _make_registry(tmp_path)
        llm = LLMConfig(model="openai/gpt-4o", api_key="direct-key")
        result = registry.resolve_for_llm_config(llm)
        assert result is llm  # 同一对象，未修改

    def test_resolve_for_llm_config_default_model_id(self, tmp_path):
        model = _make_model("m1", is_default=True)
        registry = _make_registry(tmp_path, models=[model])
        llm_in = LLMConfig(model_id="default")
        llm_out = registry.resolve_for_llm_config(llm_in)
        assert llm_out.model == model.litellm_model

    def test_resolve_for_llm_config_unknown_model_id_raises(self, tmp_path):
        model = _make_model("m1")
        registry = _make_registry(tmp_path, models=[model])
        llm_in = LLMConfig(model_id="does-not-exist")
        with pytest.raises(ModelNotFoundError):
            registry.resolve_for_llm_config(llm_in)
