"""
Providers 路由单元测试

覆盖范围：
- GET /providers — 列出所有提供商（api_key 已脱敏，is_from_env 标记正确）
- PUT /providers/{name} — 创建 / 更新 / api_key=null 时保留已有值
- DELETE /providers/{name} — 成功 / yaml 层不存在 → 404
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from hivememory.server.routers.providers import router
from hivememory.system.config.shared import ProviderCredentials
from hivememory.system.provider_registry import ProviderNotFoundError, ProviderRegistry
from hivememory.server import deps


# ---------------------------------------------------------------------------
# 测试 App 工厂
# ---------------------------------------------------------------------------

def _create_app(registry: ProviderRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_provider_registry] = lambda: registry
    return TestClient(app)


def _cred(api_key: str | None = None, api_base: str | None = None) -> ProviderCredentials:
    return ProviderCredentials(api_key=api_key, api_base=api_base)


# ---------------------------------------------------------------------------
# GET /providers
# ---------------------------------------------------------------------------

class TestListProviders:
    def test_returns_empty_list(self, tmp_path):
        from hivememory.system.provider_registry import ProviderRegistry
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        client = _create_app(registry)

        resp = client.get("/api/v1/providers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_providers(self, tmp_path):
        import yaml
        secrets = tmp_path / "p.yaml"
        secrets.write_text(
            yaml.safe_dump({"providers": {"deepseek": {"api_key": "sk-abc123456789"}}}),
            encoding="utf-8",
        )
        registry = ProviderRegistry(
            secrets_path=secrets,
            env_providers={"openai": _cred(api_key="sk-env-openai")},
        )
        client = _create_app(registry)

        resp = client.get("/api/v1/providers")
        assert resp.status_code == 200
        data = {p["name"]: p for p in resp.json()}
        assert "deepseek" in data
        assert "openai" in data

    def test_api_key_masked_in_list(self, tmp_path):
        """列表中 api_key 已脱敏，不暴露明文。"""
        import yaml
        secrets = tmp_path / "p.yaml"
        secrets.write_text(
            yaml.safe_dump({"providers": {"deepseek": {"api_key": "sk-abcdef1234567890"}}}),
            encoding="utf-8",
        )
        registry = ProviderRegistry(secrets_path=secrets)
        client = _create_app(registry)

        resp = client.get("/api/v1/providers")
        item = resp.json()[0]
        assert "api_key" not in item or item.get("api_key") is None
        masked = item.get("api_key_masked", "")
        assert "abcdef" not in masked
        assert "sk-" in masked

    def test_is_from_env_flag_in_list(self, tmp_path):
        registry = ProviderRegistry(
            secrets_path=tmp_path / "p.yaml",
            env_providers={"openai": _cred(api_key="e")},
        )
        # 也写一个 yaml 层 provider
        registry.upsert("deepseek", _cred(api_key="y"))
        client = _create_app(registry)

        resp = client.get("/api/v1/providers")
        data = {p["name"]: p for p in resp.json()}
        assert data["openai"]["is_from_env"] is True
        assert data["deepseek"]["is_from_env"] is False


# ---------------------------------------------------------------------------
# PUT /providers/{name}
# ---------------------------------------------------------------------------

class TestUpsertProvider:
    def test_create_new_provider(self, tmp_path):
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        client = _create_app(registry)

        resp = client.put(
            "/api/v1/providers/anthropic",
            json={"api_key": "sk-ant-new", "api_base": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "anthropic"
        assert data["api_key_masked"] is not None
        assert registry.get("anthropic") is not None

    def test_update_existing_provider(self, tmp_path):
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        registry.upsert("deepseek", _cred(api_key="old-key"))
        client = _create_app(registry)

        resp = client.put(
            "/api/v1/providers/deepseek",
            json={"api_key": "new-key", "api_base": "https://new.base"},
        )
        assert resp.status_code == 200
        assert registry.get("deepseek").api_key == "new-key"
        assert registry.get("deepseek").api_base == "https://new.base"

    def test_null_api_key_preserves_existing_key(self, tmp_path):
        """api_key=null 且提供商已存在时，保留原有 api_key（只更新 api_base）。"""
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        registry.upsert("deepseek", _cred(api_key="preserved-key"))
        client = _create_app(registry)

        resp = client.put(
            "/api/v1/providers/deepseek",
            json={"api_key": None, "api_base": "https://updated.base"},
        )
        assert resp.status_code == 200
        # api_key 应被保留
        assert registry.get("deepseek").api_key == "preserved-key"
        # api_base 被更新
        assert registry.get("deepseek").api_base == "https://updated.base"

    def test_provider_name_normalized_to_lowercase(self, tmp_path):
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        client = _create_app(registry)

        resp = client.put("/api/v1/providers/OPENAI", json={"api_key": "sk-upper"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "openai"
        assert registry.get("openai") is not None

    def test_env_provider_shown_as_from_env_after_yaml_write(self, tmp_path):
        """即使在 yaml 层写入了同名 provider，get() 仍以 env 优先，is_from_env=True。"""
        registry = ProviderRegistry(
            secrets_path=tmp_path / "p.yaml",
            env_providers={"openai": _cred(api_key="env-key")},
        )
        client = _create_app(registry)

        resp = client.put("/api/v1/providers/openai", json={"api_key": "yaml-key"})
        assert resp.status_code == 200
        data = resp.json()
        # 响应来自 registry.get()，env 优先 → api_key 脱敏后应是 env-key 的形式
        assert data["is_from_env"] is True


# ---------------------------------------------------------------------------
# DELETE /providers/{name}
# ---------------------------------------------------------------------------

class TestDeleteProvider:
    def test_delete_yaml_provider_success(self, tmp_path):
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        registry.upsert("deepseek", _cred(api_key="sk"))
        client = _create_app(registry)

        resp = client.delete("/api/v1/providers/deepseek")
        assert resp.status_code == 204
        assert registry.get("deepseek") is None

    def test_delete_nonexistent_returns_404(self, tmp_path):
        registry = ProviderRegistry(secrets_path=tmp_path / "p.yaml")
        client = _create_app(registry)

        resp = client.delete("/api/v1/providers/ghost")
        assert resp.status_code == 404

    def test_delete_env_only_provider_returns_404(self, tmp_path):
        """env-only provider 不在 yaml 层，delete 应返回 404。"""
        registry = ProviderRegistry(
            secrets_path=tmp_path / "p.yaml",
            env_providers={"openai": _cred(api_key="env")},
        )
        client = _create_app(registry)

        resp = client.delete("/api/v1/providers/openai")
        assert resp.status_code == 404
