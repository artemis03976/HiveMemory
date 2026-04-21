"""
Memories 路由单元测试
"""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.models import (
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
)
from hivememory.server.routers.memories import router


def _create_test_app(mock_system):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_system] = lambda: mock_system

    return app


def _make_atom(title="Test", user_id="u1"):
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestMemoriesRouter:
    def test_list_memories_no_query(self):
        atom = _make_atom()
        mock_system = MagicMock()
        mock_system.storage.get_all_memories.return_value = [atom]

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/memories?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["memories"][0]["title"] == "Test"

    def test_list_memories_with_query(self):
        atom = _make_atom()
        mock_system = MagicMock()
        mock_system.storage.search_memories.return_value = [{"memory": atom, "score": 0.9}]

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/memories?query=test&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_list_memories_filters_map_to_payload_paths(self):
        mock_system = MagicMock()
        mock_system.storage.get_all_memories.return_value = []

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/memories?user_id=u1&memory_type=FACT&limit=10")
        assert response.status_code == 200
        mock_system.storage.get_all_memories.assert_called_once_with(
            filters={"meta.user_id": "u1", "index.memory_type": "FACT"},
            limit=10,
        )

    def test_get_memory_by_id(self):
        atom = _make_atom()
        mock_system = MagicMock()
        mock_system.storage.get_memory.return_value = atom

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get(f"/api/v1/memories/{atom.id}")
        assert response.status_code == 200
        assert response.json()["id"] == str(atom.id)

    def test_get_memory_not_found(self):
        mock_system = MagicMock()
        mock_system.storage.get_memory.return_value = None

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get(f"/api/v1/memories/{uuid4()}")
        assert response.status_code == 404

    def test_get_memory_invalid_id(self):
        mock_system = MagicMock()

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/memories/not-a-uuid")
        assert response.status_code == 400

    def test_delete_memory(self):
        mid = uuid4()
        mock_system = MagicMock()
        mock_system.storage.delete_memory.return_value = True

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.delete(f"/api/v1/memories/{mid}")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_delete_memory_not_found(self):
        mock_system = MagicMock()
        mock_system.storage.delete_memory.return_value = False

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.delete(f"/api/v1/memories/{uuid4()}")
        assert response.status_code == 404
