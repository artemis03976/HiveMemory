"""
Agents 路由单元测试

路由经真实的 root API 与 Patchouli Profile 管理服务读写临时目录中的 Profile 文件；
只有 HTTP 访问依赖替换为测试组合。
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.memory_library.agent_profiles import AgentProfileStore
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.server import deps
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.routers.agents import router
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from tests.helpers.workspace import make_server_access_overrides


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")

    bus = GlobalSystemBus()
    management = AgentProfileManagementService(store=AgentProfileStore(tmp_path), bus=AsyncMock())
    for route, handler in (
        (PatchouliRoutes.AGENT_PROFILE_CREATE, management.create_agent_profile),
        (PatchouliRoutes.AGENT_PROFILE_UPDATE, management.update_agent_profile),
        (PatchouliRoutes.AGENT_PROFILE_DELETE, management.delete_agent_profile),
        (PatchouliRoutes.AGENT_PROFILE_LIST, management.list_agent_profiles),
    ):
        bus.register(route, handler)
    overrides, composition = make_server_access_overrides()
    service = RootAgentProfileService(global_bus=bus, operation_authorizer=composition.authorizer)
    app.dependency_overrides[deps.get_root_agent_service] = lambda: service
    app.dependency_overrides.update(overrides)
    return TestClient(app)


def _create(client, agent_id="coder_doll", **fields):
    return client.post("/api/v1/agents", json={"agent_id": agent_id, **fields})


def test_created_agent_is_listed_with_its_save_time(client):
    created = _create(
        client, title="写代码的", summary="Python 开发", persona="严谨", temperature=0.2
    )

    assert created.status_code == 201
    listed = client.get("/api/v1/agents")
    assert listed.status_code == 200
    (agent,) = listed.json()
    assert {key: value for key, value in agent.items() if key != "updated_at"} == {
        "agent_id": "coder_doll",
        "title": "写代码的",
        "summary": "Python 开发",
        "persona": "严谨",
        "model": "default",
        "temperature": 0.2,
        "top_p": None,
        "enabled": True,
    }
    assert agent["updated_at"] == created.json()["updated_at"]


def test_create_with_taken_agent_id_returns_409(client):
    _create(client, title="原来的")

    response = _create(client, title="新的")

    assert response.status_code == 409
    assert response.json()["error"] == "workspace.agent_profile.exists"
    assert client.get("/api/v1/agents").json()[0]["title"] == "原来的"


@pytest.mark.parametrize(
    ("agent_id", "fields", "reason"),
    [
        ("Coder Doll", {}, "invalid_agent_id"),
        ("omni_doll", {}, "invalid_agent_id"),
        ("coder_doll", {"temperature": 3.0}, "invalid_value"),
    ],
)
def test_invalid_agent_is_rejected_with_422(client, agent_id, fields, reason):
    response = _create(client, agent_id, **fields)

    assert response.status_code == 422
    assert response.json()["error"] == "workspace.agent_profile.invalid"
    assert response.json()["reason"] == reason
    assert client.get("/api/v1/agents").json() == []


def test_patch_changes_only_given_fields(client):
    _create(client, title="写代码的", persona="严谨", temperature=0.2)

    response = client.patch(
        "/api/v1/agents/coder_doll", json={"enabled": False, "temperature": None}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["title"], body["persona"], body["enabled"], body["temperature"]) == (
        "写代码的",
        "严谨",
        False,
        None,
    )


def test_agent_id_cannot_be_changed(client):
    _create(client, title="写代码的")

    response = client.patch("/api/v1/agents/coder_doll", json={"agent_id": "renamed"})

    assert response.status_code == 422
    assert [agent["agent_id"] for agent in client.get("/api/v1/agents").json()] == ["coder_doll"]


def test_patch_and_delete_of_missing_agent_return_404(client):
    assert client.patch("/api/v1/agents/nobody", json={"title": "x"}).status_code == 404
    assert client.delete("/api/v1/agents/nobody").status_code == 404


def test_deleted_agent_is_no_longer_listed(client):
    _create(client, title="写代码的")

    assert client.delete("/api/v1/agents/coder_doll").status_code == 204
    assert client.get("/api/v1/agents").json() == []
