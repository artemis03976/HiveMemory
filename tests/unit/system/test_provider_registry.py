"""
ProviderRegistry 单元测试

覆盖范围：
- YAML 加载（文件存在/缺失）
- env 层与 yaml 层的合并及优先级
- get() / list_all() / has_provider()
- upsert() / delete() 及持久化
- 删除 env-only 提供商抛出 ProviderNotFoundError
"""

import pytest
import yaml

from hivememory.system.config.shared import ProviderCredentials
from hivememory.system.provider_registry import ProviderNotFoundError, ProviderRegistry


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _make_registry(
    tmp_path,
    yaml_providers: dict[str, dict] | None = None,
    env_providers: dict[str, ProviderCredentials] | None = None,
) -> ProviderRegistry:
    secrets_path = tmp_path / "providers.secrets.yaml"
    if yaml_providers is not None:
        data = {"providers": yaml_providers}
        secrets_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return ProviderRegistry(
        secrets_path=secrets_path,
        env_providers=env_providers or {},
    )


def _cred(api_key: str | None = None, api_base: str | None = None) -> ProviderCredentials:
    return ProviderCredentials(api_key=api_key, api_base=api_base)


# ---------------------------------------------------------------------------
# 加载测试
# ---------------------------------------------------------------------------

class TestLoad:
    def test_empty_when_file_missing(self, tmp_path):
        """文件不存在时 yaml 层为空，不抛异常。"""
        registry = ProviderRegistry(secrets_path=tmp_path / "no.yaml")
        assert len(registry) == 0

    def test_loads_yaml_layer(self, tmp_path):
        """从 yaml 文件正确加载 provider 凭证。"""
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "sk-yaml", "api_base": "https://api.deepseek.com"}},
        )
        assert len(registry) == 1
        cred = registry.get("deepseek")
        assert cred is not None
        assert cred.api_key == "sk-yaml"

    def test_provider_names_lowercased(self, tmp_path):
        """Provider 名称统一小写存储。"""
        registry = _make_registry(
            tmp_path,
            yaml_providers={"DEEPSEEK": {"api_key": "sk-upper"}},
        )
        assert registry.get("deepseek") is not None
        assert registry.get("DEEPSEEK") is not None

    def test_invalid_entry_skipped(self, tmp_path):
        """格式不合法的条目被跳过，注册表仍可运行。"""
        secrets_path = tmp_path / "p.yaml"
        # 写入一个 value 为非 dict 的非法条目
        secrets_path.write_text(
            "providers:\n  bad_entry: not_a_dict\n  good: {api_key: sk-good}\n",
            encoding="utf-8",
        )
        registry = ProviderRegistry(secrets_path=secrets_path)
        assert len(registry) == 1
        assert registry.get("good") is not None

    def test_env_providers_loaded_on_init(self, tmp_path):
        """env_providers 在初始化时加载到 env 层。"""
        env = {"openai": _cred(api_key="sk-env")}
        registry = _make_registry(tmp_path, env_providers=env)
        cred = registry.get("openai")
        assert cred is not None
        assert cred.api_key == "sk-env"


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------

class TestQuery:
    def test_get_env_provider(self, tmp_path):
        env = {"openai": _cred(api_key="env-key")}
        registry = _make_registry(tmp_path, env_providers=env)
        assert registry.get("openai").api_key == "env-key"

    def test_get_yaml_provider(self, tmp_path):
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "yaml-key"}},
        )
        assert registry.get("deepseek").api_key == "yaml-key"

    def test_get_returns_none_when_not_found(self, tmp_path):
        registry = _make_registry(tmp_path)
        assert registry.get("nonexistent") is None

    def test_env_takes_priority_over_yaml(self, tmp_path):
        """同名提供商 env 层优先于 yaml 层。"""
        env = {"deepseek": _cred(api_key="env-key")}
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "yaml-key"}},
            env_providers=env,
        )
        assert registry.get("deepseek").api_key == "env-key"

    def test_list_all_returns_all_providers(self, tmp_path):
        """list_all 返回 env + yaml 合并结果。"""
        env = {"openai": _cred(api_key="env-openai")}
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "yaml-ds"}},
            env_providers=env,
        )
        items = registry.list_all()
        assert len(items) == 2
        names = {name for name, _, _ in items}
        assert "openai" in names
        assert "deepseek" in names

    def test_list_all_is_from_env_flag(self, tmp_path):
        """list_all 中 is_from_env 标记准确。"""
        env = {"openai": _cred(api_key="e")}
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "y"}},
            env_providers=env,
        )
        flags = {name: from_env for name, _, from_env in registry.list_all()}
        assert flags["openai"] is True
        assert flags["deepseek"] is False

    def test_list_all_env_overrides_yaml_entry(self, tmp_path):
        """同名 provider 在 list_all 中 env 层覆盖 yaml 层（is_from_env=True）。"""
        env = {"deepseek": _cred(api_key="env-key")}
        registry = _make_registry(
            tmp_path,
            yaml_providers={"deepseek": {"api_key": "yaml-key"}},
            env_providers=env,
        )
        items = registry.list_all()
        assert len(items) == 1
        name, cred, from_env = items[0]
        assert from_env is True
        assert cred.api_key == "env-key"

    def test_has_provider_true_for_env(self, tmp_path):
        env = {"openai": _cred()}
        registry = _make_registry(tmp_path, env_providers=env)
        assert registry.has_provider("openai") is True

    def test_has_provider_true_for_yaml(self, tmp_path):
        registry = _make_registry(tmp_path, yaml_providers={"deepseek": {}})
        assert registry.has_provider("deepseek") is True

    def test_has_provider_false(self, tmp_path):
        registry = _make_registry(tmp_path)
        assert registry.has_provider("ghost") is False


# ---------------------------------------------------------------------------
# 写入接口
# ---------------------------------------------------------------------------

class TestWrite:
    def test_upsert_creates_new_provider(self, tmp_path):
        registry = _make_registry(tmp_path)
        registry.upsert("anthropic", _cred(api_key="sk-ant"))
        assert registry.get("anthropic").api_key == "sk-ant"

    def test_upsert_updates_existing_provider(self, tmp_path):
        registry = _make_registry(
            tmp_path, yaml_providers={"deepseek": {"api_key": "old-key"}}
        )
        registry.upsert("deepseek", _cred(api_key="new-key"))
        assert registry.get("deepseek").api_key == "new-key"

    def test_upsert_lowercases_name(self, tmp_path):
        registry = _make_registry(tmp_path)
        registry.upsert("OPENAI", _cred(api_key="sk-upper"))
        assert registry.get("openai").api_key == "sk-upper"

    def test_delete_yaml_provider(self, tmp_path):
        registry = _make_registry(
            tmp_path, yaml_providers={"deepseek": {"api_key": "sk"}}
        )
        registry.delete("deepseek")
        assert registry.get("deepseek") is None
        assert len(registry) == 0

    def test_delete_env_only_provider_raises(self, tmp_path):
        """env-only provider 无法通过 delete() 删除。"""
        env = {"openai": _cred(api_key="env-key")}
        registry = _make_registry(tmp_path, env_providers=env)
        with pytest.raises(ProviderNotFoundError):
            registry.delete("openai")

    def test_delete_nonexistent_raises(self, tmp_path):
        registry = _make_registry(tmp_path)
        with pytest.raises(ProviderNotFoundError):
            registry.delete("ghost")


# ---------------------------------------------------------------------------
# 持久化测试
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_upsert_persisted_to_file(self, tmp_path):
        registry = _make_registry(tmp_path)
        registry.upsert("deepseek", _cred(api_key="sk-persist", api_base="https://api.deepseek.com"))

        # 重新加载，数据应仍然存在
        reloaded = ProviderRegistry(secrets_path=tmp_path / "providers.secrets.yaml")
        cred = reloaded.get("deepseek")
        assert cred is not None
        assert cred.api_key == "sk-persist"
        assert cred.api_base == "https://api.deepseek.com"

    def test_delete_persisted_to_file(self, tmp_path):
        registry = _make_registry(
            tmp_path, yaml_providers={"deepseek": {"api_key": "sk"}}
        )
        registry.delete("deepseek")

        reloaded = ProviderRegistry(secrets_path=tmp_path / "providers.secrets.yaml")
        assert reloaded.get("deepseek") is None

    def test_env_layer_not_persisted(self, tmp_path):
        """env 层凭证不应被写入 yaml 文件。"""
        env = {"openai": _cred(api_key="env-key")}
        registry = _make_registry(tmp_path, env_providers=env)
        # 触发一次 upsert 以确保文件被写入
        registry.upsert("deepseek", _cred(api_key="yaml-key"))

        secrets_path = tmp_path / "providers.secrets.yaml"
        data = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
        yaml_providers = data.get("providers", {})
        # openai 是 env 层，不应出现在 yaml 文件中
        assert "openai" not in yaml_providers
        assert "deepseek" in yaml_providers

    def test_save_creates_file_if_not_exists(self, tmp_path):
        """yaml 文件不存在时，第一次 upsert 会自动创建。"""
        secrets_path = tmp_path / "new_secrets.yaml"
        registry = ProviderRegistry(secrets_path=secrets_path)
        registry.upsert("openai", _cred(api_key="sk-new"))
        assert secrets_path.exists()
