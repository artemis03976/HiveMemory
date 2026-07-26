"""
Ingest 路由单元测试
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.ingest import router

SOURCE = "claude_code"
CONVERSATION = "sess-1"


def _create_test_app(mock_service):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_ingress_service] = lambda: mock_service

    return app


def _mock_service(status: str = "buffered", memory=None):
    mock_service = MagicMock()
    mock_service.ingest_event = AsyncMock(return_value={
        "status": status,
        "external_event_id": "evt-1",
        "memory": memory,
    })
    return mock_service


def _body(**overrides):
    body = {
        "source": SOURCE,
        "external_conversation_id": CONVERSATION,
        "user_id": "u1",
    }
    body.update(overrides)
    return body


class TestIngestRouter:
    def test_ingest_user_message(self):
        mock_service = _mock_service(status="accepted", memory="[memory]")
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(role="user", content="hello"),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["memory"] == "[memory]"
        assert data["external_event_id"] == "evt-1"

        mock_service.ingest_event.assert_called_once()
        event = mock_service.ingest_event.call_args.kwargs["event"]
        assert event.source == SOURCE
        assert event.external_conversation_id == CONVERSATION
        # 未显式提供时由服务端生成幂等键
        assert event.external_event_id

    def test_ingest_assistant_message(self):
        mock_service = _mock_service()
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(role="assistant", content="hi there"),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "buffered"

    def test_ingest_preserves_external_event_id_and_final_flag(self):
        mock_service = _mock_service()
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(
                role="assistant",
                content="done",
                external_event_id="ext-42",
                turn_id="turn-7",
                sequence=3,
                is_final=True,
            ),
        )
        assert response.status_code == 200

        event = mock_service.ingest_event.call_args.kwargs["event"]
        assert event.external_event_id == "ext-42"
        assert event.turn_id == "turn-7"
        assert event.sequence == 3
        assert event.is_final is True

    def test_ingest_requires_source_and_conversation(self):
        mock_service = _mock_service()
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json={"role": "user", "content": "hello", "user_id": "u1"},
        )
        assert response.status_code == 422

    def test_ingest_duplicate_status_passthrough(self):
        mock_service = _mock_service(status="duplicate")
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(role="user", content="hello", external_event_id="ext-1"),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "duplicate"

    def test_ingest_tool_call(self):
        mock_service = _mock_service()
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(
                role="tool_call",
                content="get_weather(city='北京')",
                action_id="a1",
                tool_name="weather_api",
                tool_kind="function_call",
                tool_args={"city": "北京"},
            ),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "buffered"

        event = mock_service.ingest_event.call_args.kwargs["event"]
        assert event.role == "tool_call"
        assert event.action_id == "a1"
        assert event.tool_name == "weather_api"
        assert event.tool_args == {"city": "北京"}

    def test_ingest_tool_result(self):
        mock_service = _mock_service()
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest",
            json=_body(
                role="tool_result",
                content="北京 25°C",
                action_id="a1",
                status="success",
            ),
        )
        assert response.status_code == 200
        assert response.json()["status"] == "buffered"

        event = mock_service.ingest_event.call_args.kwargs["event"]
        assert event.role == "tool_result"
        assert event.action_id == "a1"
        assert event.status == "success"


class TestIngestFlushRouter:
    def test_flush_conversation(self):
        mock_service = MagicMock()
        mock_service.flush_conversation = AsyncMock(return_value=True)
        client = TestClient(_create_test_app(mock_service))

        response = client.post(
            "/api/v1/ingest/flush",
            json={
                "source": SOURCE,
                "external_conversation_id": CONVERSATION,
                "user_id": "u1",
            },
        )
        assert response.status_code == 200
        assert response.json()["submitted"] is True

        kwargs = mock_service.flush_conversation.call_args.kwargs
        assert kwargs["source"] == SOURCE
        assert kwargs["external_conversation_id"] == CONVERSATION
