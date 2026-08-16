"""
Models 路由单元测试

覆盖范围：
- GET /models — 列出所有模型（api_key 已脱敏）
- POST /models — 创建成功 / 重复 ID → 409
- GET /models/{id} — 找到 / 不存在 → 404
- PUT /models/{id} — 更新成功 / 不存在 → 404 / 空 body → 400
- DELETE /models/{id} — 删除成功 / 不存在 → 404
- api_key 脱敏规则验证
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from hivememory.core.models.model_definition import ModelDefinition
from hivememory.server.routers.models import router
from hivememory.system.model_registry import DuplicateModelIdError, ModelNotFoundError
from hivememory.server import deps


# ---------------------------------------------------------------------------
# 测试 App 工厂
# ---------------------------------------------------------------------------

def _create_app(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_model_registry] = lambda: registry
    return TestClient(app)


def _make_model(
    model_id: str = "test-model",
    display_name: str = "Test Model",
    litellm_model: str = "deepseek/deepseek-chat",
    is_default: bool = False,
    api_key: str | None = None,
    provider: str = "deepseek",
) -> ModelDefinition:
    return ModelDefinition(
        id=model_id,
        display_name=display_name,
        litellm_model=litellm_model,
        is_default=is_default,
        api_key=api_key,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_empty_list(self):
        registry = MagicMock()
        registry.list_models.return_value = []
        client = _create_app(registry)

        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_models(self):
        registry = MagicMock()
        registry.list_models.return_value = [
            _make_model("m1", "Model 1"),
            _make_model("m2", "Model 2"),
        ]
        client = _create_app(registry)

        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "m1"
        assert data[1]["id"] == "m2"

    def test_api_key_masked_in_response(self):
        """明文 api_key 不应出现在响应中，只返回脱敏形式。"""
        registry = MagicMock()
        registry.list_models.return_value = [
            _make_model("m1", api_key="sk-verylongapikey1234"),
        ]
        client = _create_app(registry)

        resp = client.get("/api/v1/models")
        model = resp.json()[0]
        assert "api_key" not in model
        masked = model.get("api_key_masked", "")
        assert "verylongapikey" not in masked
        assert masked.startswith("sk-")

    def test_api_key_none_returns_null_masked(self):
        registry = MagicMock()
        registry.list_models.return_value = [_make_model("m1", api_key=None)]
        client = _create_app(registry)

        resp = client.get("/api/v1/models")
        assert resp.json()[0]["api_key_masked"] is None


# ---------------------------------------------------------------------------
# POST /models
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_create_success(self):
        registry = MagicMock()
        registry.add_model.return_value = None
        client = _create_app(registry)

        payload = {
            "id": "new-model",
            "display_name": "New Model",
            "litellm_model": "openai/gpt-4o",
            "is_default": False,
        }
        resp = client.post("/api/v1/models", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "new-model"
        assert data["display_name"] == "New Model"
        registry.add_model.assert_called_once()

    def test_create_duplicate_returns_409(self):
        registry = MagicMock()
        registry.add_model.side_effect = DuplicateModelIdError("already exists")
        client = _create_app(registry)

        payload = {
            "id": "dup",
            "display_name": "Dup",
            "litellm_model": "openai/gpt-4o",
        }
        resp = client.post("/api/v1/models", json=payload)
        assert resp.status_code == 409

    def test_provider_auto_derived(self):
        """provider 留空时从 litellm_model 推导。"""
        registry = MagicMock()
        registry.add_model.return_value = None
        client = _create_app(registry)

        payload = {
            "id": "ds",
            "display_name": "DS",
            "litellm_model": "deepseek/deepseek-chat",
            "provider": "",
        }
        resp = client.post("/api/v1/models", json=payload)
        assert resp.status_code == 201
        # ModelDefinition.__init__ 会自动推导 provider
        assert resp.json()["provider"] == "deepseek"


# ---------------------------------------------------------------------------
# GET /models/{model_id}
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_get_existing_model(self):
        registry = MagicMock()
        registry.get_model.return_value = _make_model("m1", "Model 1")
        client = _create_app(registry)

        resp = client.get("/api/v1/models/m1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "m1"

    def test_get_nonexistent_returns_404(self):
        registry = MagicMock()
        registry.get_model.return_value = None
        client = _create_app(registry)

        resp = client.get("/api/v1/models/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /models/{model_id}
# ---------------------------------------------------------------------------

class TestUpdateModel:
    def test_update_success(self):
        updated = _make_model("m1", display_name="Updated Name")
        registry = MagicMock()
        registry.update_model.return_value = updated
        client = _create_app(registry)

        resp = client.put("/api/v1/models/m1", json={"display_name": "Updated Name"})
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Updated Name"
        registry.update_model.assert_called_once_with("m1", {"display_name": "Updated Name"})

    def test_update_not_found_returns_404(self):
        registry = MagicMock()
        registry.update_model.side_effect = ModelNotFoundError("not found")
        client = _create_app(registry)

        resp = client.put("/api/v1/models/ghost", json={"display_name": "X"})
        assert resp.status_code == 404

    def test_update_empty_body_returns_400(self):
        """请求体中没有可更新字段时返回 400。"""
        registry = MagicMock()
        client = _create_app(registry)

        # 发送空对象，to_updates_dict() 返回空 dict → 400
        resp = client.put("/api/v1/models/m1", json={})
        assert resp.status_code == 400
        registry.update_model.assert_not_called()

    def test_update_api_key_empty_string_clears_key(self):
        """api_key="" 被转换为 None（清除密钥）。"""
        updated = _make_model("m1", api_key=None)
        registry = MagicMock()
        registry.update_model.return_value = updated
        client = _create_app(registry)

        resp = client.put("/api/v1/models/m1", json={"api_key": ""})
        assert resp.status_code == 200
        # to_updates_dict 把 "" 转为 None
        call_updates = registry.update_model.call_args[0][1]
        assert call_updates.get("api_key") is None


# ---------------------------------------------------------------------------
# DELETE /models/{model_id}
# ---------------------------------------------------------------------------

class TestDeleteModel:
    def test_delete_success(self):
        registry = MagicMock()
        registry.delete_model.return_value = None
        client = _create_app(registry)

        resp = client.delete("/api/v1/models/m1")
        assert resp.status_code == 204
        registry.delete_model.assert_called_once_with("m1")

    def test_delete_not_found_returns_404(self):
        registry = MagicMock()
        registry.delete_model.side_effect = ModelNotFoundError("not found")
        client = _create_app(registry)

        resp = client.delete("/api/v1/models/ghost")
        assert resp.status_code == 404
