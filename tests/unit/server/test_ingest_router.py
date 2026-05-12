"""
Ingest 路由单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.ingest import router


def _create_test_app(mock_system):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_system] = lambda: mock_system

    return app


class TestIngestRouter:
    def test_ingest_user_message(self):
        mock_system = MagicMock()
        mock_system.ingest_event = AsyncMock(return_value={
            "intent": "Chat",
            "rewritten": "hello rewritten",
            "keywords": ["hello"],
            "worth_saving": True,
            "memory": None,
        })

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.post(
            "/api/v1/ingest",
            json={"role": "user", "content": "hello", "user_id": "u1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "Chat"
        assert data["worth_saving"] is True

        mock_system.ingest_event.assert_called_once()

    def test_ingest_assistant_message(self):
        mock_system = MagicMock()
        mock_system.ingest_event = AsyncMock(return_value={
            "intent": "buffered",
            "rewritten": None,
            "keywords": [],
            "worth_saving": True,
            "memory": None,
        })

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.post(
            "/api/v1/ingest",
            json={"role": "assistant", "content": "hi there", "user_id": "u1"},
        )
        assert response.status_code == 200
        assert response.json()["intent"] == "buffered"

    def test_ingest_tool_call(self):
        mock_system = MagicMock()
        mock_system.ingest_event = AsyncMock(return_value={
            "intent": "buffered",
            "rewritten": None,
            "keywords": [],
            "worth_saving": True,
            "memory": None,
        })

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.post(
            "/api/v1/ingest",
            json={
                "role": "tool_call",
                "content": "get_weather(city='北京')",
                "user_id": "u1",
                "action_id": "a1",
                "tool_name": "weather_api",
                "tool_kind": "function_call",
                "tool_args": {"city": "北京"},
            },
        )
        assert response.status_code == 200
        assert response.json()["intent"] == "buffered"

        call_kwargs = mock_system.ingest_event.call_args.kwargs
        event = call_kwargs["event"]
        assert event.role == "tool_call"
        assert event.action_id == "a1"
        assert event.tool_name == "weather_api"

    def test_ingest_tool_result(self):
        mock_system = MagicMock()
        mock_system.ingest_event = AsyncMock(return_value={
            "intent": "buffered",
            "rewritten": None,
            "keywords": [],
            "worth_saving": True,
            "memory": None,
        })

        app = _create_test_app(mock_system)
        client = TestClient(app)

        response = client.post(
            "/api/v1/ingest",
            json={
                "role": "tool_result",
                "content": "北京 25°C",
                "user_id": "u1",
                "action_id": "a1",
                "status": "success",
            },
        )
        assert response.status_code == 200
        assert response.json()["intent"] == "buffered"

        call_kwargs = mock_system.ingest_event.call_args.kwargs
        event = call_kwargs["event"]
        assert event.role == "tool_result"
        assert event.action_id == "a1"
        assert event.status == "success"
