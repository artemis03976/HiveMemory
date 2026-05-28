"""
Agents 路由单元测试
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.server.routers.agents import router


def _create_test_app(storage):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    service = AgentApplicationService(
        global_bus=GlobalSystemBus(),
        config=MagicMock(),
        storage=storage,
    )
    app.dependency_overrides[deps.get_agent_service] = lambda: service

    return app


def test_list_agents_uses_index_memory_type_filter():
    storage = MagicMock()
    storage.get_all_memories.return_value = []

    app = _create_test_app(storage)
    client = TestClient(app)

    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    storage.get_all_memories.assert_called_once_with(
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )
