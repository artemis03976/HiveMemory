"""
Agents 路由单元测试
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.server.routers.agents import router


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
    service = AgentApplicationService(
        global_bus=bus,
        config=MagicMock(),
    )
    app.dependency_overrides[deps.get_agent_service] = lambda: service

    return app


class _AgentProfileManagementStub:
    def __init__(self, storage):
        self.storage = storage

    async def create_agent_profile(self, atom):
        self.storage.upsert_memory(atom)
        return atom

    async def list_agent_profiles(self, *, limit=100):
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
