"""
Agents 路由单元测试
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.agents import router


def _create_test_app(mock_system):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_system] = lambda: mock_system

    return app


def test_list_agents_uses_index_memory_type_filter():
    mock_system = MagicMock()
    mock_system.patchouli.storage.get_all_memories.return_value = []

    app = _create_test_app(mock_system)
    client = TestClient(app)

    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    mock_system.patchouli.storage.get_all_memories.assert_called_once_with(
        filters={"index.memory_type": "AGENT_PROFILE"},
        limit=100,
    )
