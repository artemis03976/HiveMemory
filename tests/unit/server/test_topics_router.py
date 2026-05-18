"""
Topics 路由单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.topics import router


def _create_test_app(mock_system):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_system] = lambda: mock_system
    app.dependency_overrides[deps.get_user_id] = lambda: "test_user"

    return app


def _make_snapshot(topic_id="t1", title="Test Topic"):
    s = MagicMock()
    s.topic_id = topic_id
    s.title = title
    s.state_summary = "summary"
    s.last_turn = {"user": "hi", "assistant": "hello"}
    s.total_tokens = 100
    return s


class TestTopicsRouter:
    def test_list_topics(self):
        mock_system = MagicMock()
        mock_system.kernel.librarian_core.get_active_topics_snapshots.return_value = [
            _make_snapshot("t1", "Topic 1"),
            _make_snapshot("t2", "Topic 2"),
        ]

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/topics")
        assert response.status_code == 200
        data = response.json()
        assert len(data["topics"]) == 2
        assert data["topics"][0]["topic_id"] == "t1"

    def test_list_topics_empty(self):
        mock_system = MagicMock()
        mock_system.kernel.librarian_core.get_active_topics_snapshots.return_value = []

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.get("/api/v1/topics")
        assert response.status_code == 200
        assert response.json()["topics"] == []

    def test_archive_topic(self):
        mock_system = MagicMock()
        mock_system.manual_archive_topic = AsyncMock(return_value={
            "success": True,
            "topic_id": "t1",
            "message": "Topic settled",
            "blocks_archived": 5,
        })

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.post("/api/v1/topics/t1/archive")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["blocks_archived"] == 5
