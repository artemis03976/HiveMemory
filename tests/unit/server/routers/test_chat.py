"""
Chat 路由单元测试

测试覆盖:
    1. 正常对话 — SSE 事件序列: topic_info → token → done
    2. MTP 对话 — SSE 事件序列: topic_info → token → mtp_start → mtp_result → token → done
    3. 异常处理 — SSE error 事件
"""

import json
import asyncio
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.server.routers.chat import router
from hivememory.server.routers.chat import chat
from hivememory.server.models.chat import ChatRequest


def _create_test_app(mock_service):
    """创建测试用 FastAPI 应用"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    from hivememory.server import deps
    app.dependency_overrides[deps.get_chat_service] = lambda: mock_service

    return app


def _parse_sse_events(response_text: str):
    """解析 SSE 文本为事件列表"""
    events = []
    current_event = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = json.loads(line[len("data:"):].strip())
    if current_event:
        events.append(current_event)
    return events


class TestChatRouter:
    def test_runtime_generation_options_are_forwarded(self):
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {"event": "done", "data": {"final_text": "ok", "mtp_iterations": 0, "total_iterations": 1}}

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "hello",
                "user_id": "test",
                "generation_options": {
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 1024,
                },
            },
        )
        assert response.status_code == 200
        mock_service.chat_stream.assert_called_once()
        call_kwargs = mock_service.chat_stream.call_args.kwargs
        assert call_kwargs["generation_options"] == {
            "model": "gpt-4o",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 1024,
        }

    def test_normal_chat_sse_events(self):
        """正常对话: topic_info → token → done"""
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {"event": "topic_info", "data": {"topic_id": "t1", "is_new": False}}
            yield {"event": "token", "data": {"content": "Hello "}}
            yield {"event": "token", "data": {"content": "world!"}}
            yield {
                "event": "done",
                "data": {
                    "final_text": "Hello world!",
                    "mtp_iterations": 0,
                    "total_iterations": 1,
                },
            }

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "user_id": "test"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert "topic_info" in event_types
        assert "done" in event_types
        # token 事件应该存在
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) >= 1

    def test_mtp_chat_sse_events(self):
        """MTP 对话: topic_info → token → mtp_start → mtp_result → token → done"""
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {"event": "topic_info", "data": {"topic_id": "t1", "is_new": False}}
            yield {"event": "token", "data": {"content": "Let me search. "}}
            yield {"event": "mtp_start", "data": {"verb": "SEARCH", "iteration": 1}}
            yield {"event": "mtp_result", "data": {"verb": "SEARCH", "status": "success", "iteration": 1}}
            yield {"event": "token", "data": {"content": "Found it!"}}
            yield {
                "event": "done",
                "data": {
                    "final_text": "Let me search. Found it!",
                    "mtp_iterations": 1,
                    "total_iterations": 2,
                },
            }

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "search something", "user_id": "test"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert "mtp_start" in event_types
        assert "mtp_result" in event_types

    def test_error_event(self):
        """异常: error 事件"""
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {"event": "error", "data": {"message": "LLM 调用失败"}}

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "user_id": "test"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "LLM" in error_events[0]["data"]["message"]

    def test_command_result_sse_event_is_forwarded_as_own_event(self):
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {
                "event": "command_result",
                "data": {
                    "command_id": "system.clear",
                    "status": "completed",
                    "message": "cleared",
                    "client_action": {"type": "clear_chat"},
                },
            }
            yield {
                "event": "done",
                "data": {
                    "status": "completed",
                    "command_id": "system.clear",
                },
            }

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "/clear", "user_id": "test"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]
        assert event_types == ["command_result", "done"]
        assert "token" not in event_types
        assert events[0]["data"]["message"] == "cleared"
        assert events[0]["data"]["client_action"] == {"type": "clear_chat"}

    def test_uuid_payload_is_serializable(self):
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {
                "event": "memory_refs",
                "data": {
                    "memories": [{"id": uuid4(), "content": "hello"}],
                },
            }

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))

        app = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "user_id": "test"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        memory_events = [e for e in events if e["event"] == "memory_refs"]
        assert len(memory_events) == 1
        memory_id = memory_events[0]["data"]["memories"][0]["id"]
        assert isinstance(memory_id, str)

    @pytest.mark.asyncio
    async def test_disconnect_while_waiting_for_next_event_cancels_generation(self):
        mock_service = MagicMock()
        blocker = asyncio.Event()

        async def fake_stream(**kwargs):
            try:
                yield {"event": "generation_id", "data": {"generation_id": "gen-1"}}
                await blocker.wait()
                yield {"event": "done", "data": {"final_text": "late"}}
            finally:
                blocker.set()

        disconnect_checks = 0

        class FakeRequest:
            async def is_disconnected(self):
                nonlocal disconnect_checks
                disconnect_checks += 1
                return disconnect_checks >= 3

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))
        mock_service.cancel_generation = MagicMock()

        response = await chat(
            request=FakeRequest(),
            body=ChatRequest(message="hello", user_id="test"),
            service=mock_service,
        )

        first_chunk = await response.body_iterator.__anext__()
        assert first_chunk["event"] == "generation_id"
        with pytest.raises(StopAsyncIteration):
            await response.body_iterator.__anext__()

        generation_id = mock_service.chat_stream.call_args.kwargs["generation_id"]
        assert generation_id
        mock_service.cancel_generation.assert_called_once_with(
            generation_id,
            reason="client_disconnected",
        )

    @pytest.mark.asyncio
    async def test_disconnect_before_generation_id_event_cancels_generation(self):
        mock_service = MagicMock()
        stream_started = asyncio.Event()
        blocker = asyncio.Event()

        async def fake_stream(**kwargs):
            stream_started.set()
            await blocker.wait()
            yield {"event": "generation_id", "data": {"generation_id": kwargs["generation_id"]}}

        disconnect_checks = 0

        class FakeRequest:
            async def is_disconnected(self):
                nonlocal disconnect_checks
                disconnect_checks += 1
                return disconnect_checks >= 2

        mock_service.chat_stream = MagicMock(side_effect=lambda **kw: fake_stream(**kw))
        mock_service.cancel_generation = MagicMock()

        response = await chat(
            request=FakeRequest(),
            body=ChatRequest(message="hello", user_id="test"),
            service=mock_service,
        )

        with pytest.raises(StopAsyncIteration):
            await response.body_iterator.__anext__()

        assert stream_started.is_set()
        generation_id = mock_service.chat_stream.call_args.kwargs["generation_id"]
        assert generation_id
        mock_service.cancel_generation.assert_called_once_with(
            generation_id,
            reason="client_disconnected",
        )
