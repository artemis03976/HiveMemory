"""
Agents 路由单元测试
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import MemoryAliasConflictError
from hivememory.server import deps
from hivememory.server.routers.agents import router
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from tests.helpers.workspace import (
    make_server_access_overrides,
    make_workspace_runtime,
)


def _create_test_app(storage):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    bus = GlobalSystemBus()
    management = _AgentProfileManagementStub(storage)
    bus.register(
        GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE,
        management.create_agent_profile,
    )
    bus.register(
        GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
        management.list_agent_profiles,
    )
    # 管理用例的 operation 授权（management.memory）在能力层执行：服务与
    # 访问依赖共享同一组合的操作授权者，context 才能通过签发校验。
    overrides, composition = make_server_access_overrides()
    service = AgentApplicationService(
        global_bus=bus,
        operation_authorizer=composition.authorizer,
        profile_reader=make_workspace_runtime(bus).profiles,
    )
    app.dependency_overrides[deps.get_agent_service] = lambda: service
    app.dependency_overrides.update(overrides)

    return app


class _AgentProfileManagementStub:
    def __init__(self, storage):
        self.storage = storage

    # 对齐 Patchouli 路由当前契约：只接收操作授权者返回的 scope，不接收 access。
    async def create_agent_profile(self, identity_scope, atom):
        self.storage.upsert_memory(atom)
        return atom

    async def list_agent_profiles(self, *, identity_scope, limit=100):
        return self.storage.get_all_memories(
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )


def test_list_agents_returns_200_and_passes_limit():
    storage = MagicMock()
    storage.get_all_memories.return_value = []

    app = _create_test_app(storage)
    client = TestClient(app)

    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    # limit=100 由 router/service 层透传；stub 内部的 filter 属于 stub 自身行为，不在此处断言
    assert storage.get_all_memories.call_args.kwargs["limit"] == 100


def test_create_agent_rejects_blank_title_with_422():
    """Agent 名称去除空白后必填：返回 422 与字段原因，而不是 500，且不写入。"""
    storage = MagicMock()
    client = TestClient(_create_test_app(storage))

    response = client.post(
        "/api/v1/agents",
        json={"title": "  ", "alias": "reviewer_doll"},
    )

    assert response.status_code == 422
    assert "title" in response.json()["detail"]
    storage.upsert_memory.assert_not_called()


def test_create_agent_alias_conflict_returns_409():
    """Agent alias（即 agent_id）已被占用时返回 409，而不是 500。"""
    storage = MagicMock()
    storage.upsert_memory.side_effect = MemoryAliasConflictError(
        "alias 已被同一 Workspace 内的其他记忆占用",
        details={"alias": "reviewer_doll", "reason": "alias_occupied"},
    )
    client = TestClient(_create_test_app(storage))

    response = client.post(
        "/api/v1/agents",
        json={"title": "Reviewer", "alias": "reviewer_doll"},
    )

    assert response.status_code == 409


def test_list_agents_passes_target_workspace_and_access():
    """路由只取得请求级 context 与目标 workspace，并以此调用能力层。"""
    service = MagicMock()
    service.list_agent_profiles = AsyncMock(return_value=[])

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_agent_service] = lambda: service
    overrides, _ = make_server_access_overrides(users=["u1"])
    app.dependency_overrides.update(overrides)
    client = TestClient(app)

    response = client.get("/api/v1/agents", headers={"x-user-id": "u1"})

    assert response.status_code == 200
    assert response.json() == []
    kwargs = service.list_agent_profiles.await_args.kwargs
    assert kwargs["target_workspace"].owner_user_id == "u1"
    assert kwargs["target_workspace"].workspace_id == "main_workspace"
    assert kwargs["limit"] == 100
    assert isinstance(kwargs["access"], WorkspaceAccessContext)
    # 声明只作为认证输入：路由处理函数不取得 identity_scope 或 claims
    assert "identity_scope" not in kwargs
    assert "claims" not in kwargs
