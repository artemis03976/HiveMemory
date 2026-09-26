"""
Agents 路由单元测试
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.server.routers.agents import router
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.workspace.access import WorkspaceAccessGuard
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.registry import WorkspaceActorAccessRegistry
from tests.helpers.workspace import make_workspace_runtime


def _create_test_app(storage):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps

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
    # 管理路由不经 Profile 读取 resolver 与 operation 守卫：注入真实但空白的依赖。
    service = AgentApplicationService(
        global_bus=bus,
        config=MagicMock(),
        access_guard=WorkspaceAccessGuard(WorkspaceActorAccessRegistry([])),
        profile_reader=make_workspace_runtime(bus).profiles,
    )
    app.dependency_overrides[deps.get_agent_service] = lambda: service

    return app


class _AgentProfileManagementStub:
    def __init__(self, storage):
        self.storage = storage

    async def create_agent_profile(self, identity_scope, atom, access=None):
        self.storage.upsert_memory(atom)
        return atom

    async def list_agent_profiles(self, *, identity_scope, limit=100, access=None):
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
