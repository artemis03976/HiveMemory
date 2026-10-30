"""
Chat 路由单元测试

测试覆盖:
    1. 注册入口 — 先经网关认证并注册进程，再开始流式运行；注册失败
       （AdmissionDeniedError）返回 403 且不创建进程、不运行
    2. 正常对话 — SSE 事件序列: token → done
    3. MTP 对话 — SSE 事件序列: token → mtp_start → mtp_result → token → done
    4. 异常处理 — SSE error 事件
    5. 断连 — 响应被取消时交付生成器随之关闭（退订并以 client_disconnected
       请求停止由注册入口负责）；生成器从未开始迭代时经 close_process(handle)
       关闭进程
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.errors import AdmissionDeniedError
from hivememory.server import deps
from hivememory.server.deps import RequestIdentitySelection
from hivememory.server.error_handlers import install_error_handlers
from hivememory.server.models.chat import ChatRequest
from hivememory.server.routers.chat import chat, router
from tests.helpers.workspace import make_server_access_overrides


def _registered_process(*, process_id: str):
    """构造 ``register_process`` 返回的进程句柄 stub（只暴露 process_id）。"""
    return SimpleNamespace(process_id=process_id)


def _create_test_app(mock_service, access=None):
    """创建测试用 FastAPI 应用；``access`` 为可选共享的 (overrides, composition)。"""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(router, prefix="/api/v1")

    app.dependency_overrides[deps.get_process_service] = lambda: mock_service
    # /chat 与 /chat/stop 经统一认证网关取得访问 context（进程绑定/请求级）；
    # 注册失败的 403 映射复用生产异常处理器。

    if access is None:
        access = make_server_access_overrides()
    overrides, composition = access
    app.dependency_overrides.update(overrides)

    return app, composition


def _wired_chat_service(stream_factory):
    """构造按注册→运行编排的 mock 任务进程服务。

    ``register_process`` 返回只暴露 ``process_id`` 的进程句柄 stub；
    ``run_process`` 与生产签名一致地只接收进程句柄。
    """
    mock_service = MagicMock()
    mock_service.register_process = AsyncMock(
        side_effect=lambda **kwargs: _registered_process(process_id=kwargs["process_id"])
    )
    mock_service.run_process = MagicMock(
        side_effect=lambda *args, **kw: stream_factory(*args, **kw)
    )
    mock_service.close_process = AsyncMock()
    return mock_service


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
            current_event["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_event["data"] = json.loads(line[len("data:") :].strip())
    if current_event:
        events.append(current_event)
    return events


class TestChatRegistration:
    def test_session_id_is_accepted_without_entering_actor_claims(self):
        """旧客户端可以继续发送 session_id，但它不进入进程注册的 actor 声明。"""
        service = _wired_chat_service(
            lambda *args, **kwargs: _simple_stream({"event": "done", "data": {"final_text": "ok"}})
        )
        app, _ = _create_test_app(service)

        response = TestClient(app).post(
            "/api/v1/chat",
            headers={"x-user-id": "user-a"},
            json={"message": "hello", "agent_id": "test_agent", "session_id": "old-session"},
        )

        assert response.status_code == 200
        assert _parse_sse_events(response.text) == [{"event": "done", "data": {"final_text": "ok"}}]
        # 注册载荷是该入口的出站契约；与 HTTP 终态一起验证，避免仅断言 mock 调用。
        assert service.register_process.call_args.kwargs["actor"].model_dump() == {
            "user_id": "user-a",
            "agent_id": "test_agent",
            "team_id": None,
        }

    def test_register_failure_returns_403_and_never_runs(self):
        """注册认证失败直接上抛：HTTP 403，不创建进程也不运行。"""
        mock_service = MagicMock()
        mock_service.register_process = AsyncMock(
            side_effect=AdmissionDeniedError(
                "该 Actor 在目标 Workspace 没有有效的访问登记",
                details={"reason": "actor_not_admitted"},
            )
        )
        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )

        assert response.status_code == 403
        assert response.json()["reason"] == "actor_not_admitted"
        mock_service.register_process.assert_called_once()
        mock_service.run_process.assert_not_called()
        mock_service.close_process.assert_not_called()

    def test_reserved_system_agent_returns_400_without_registration(self):
        """Chat 必须由具体 Agent 执行：保留 system 作为 agent_id 在认证前以 400 拒绝。"""
        mock_service = MagicMock()
        mock_service.register_process = AsyncMock()
        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "system"},
        )

        assert response.status_code == 400
        mock_service.register_process.assert_not_called()

    def test_runtime_generation_options_are_forwarded_to_registration(self):
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {
                    "event": "done",
                    "data": {"final_text": "ok", "mtp_iterations": 0, "total_iterations": 1},
                }
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "hello",
                "agent_id": "test_agent",
                "generation_options": {
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 1024,
                },
            },
        )
        assert response.status_code == 200
        register_kwargs = mock_service.register_process.call_args.kwargs
        assert register_kwargs["generation_options"] == {
            "model": "gpt-4o",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 1024,
        }
        # run_process 收到注册返回的不透明进程句柄
        run_args, _ = mock_service.run_process.call_args
        assert run_args[0].process_id == register_kwargs["process_id"]

    def test_registration_precedes_stream_run(self):
        """正常路径先完成注册（认证+登记），再开始流式运行。"""
        order: list[str] = []

        async def fake_register(**kwargs):
            order.append("register")
            return _registered_process(process_id=kwargs["process_id"])

        def fake_run(handle, **_kwargs):
            order.append("run")

            async def gen():
                yield {"event": "done", "data": {"final_text": "ok"}}

            return gen()

        mock_service = MagicMock()
        mock_service.register_process = AsyncMock(side_effect=fake_register)
        mock_service.run_process = MagicMock(side_effect=fake_run)
        mock_service.close_process = AsyncMock()

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )

        assert response.status_code == 200
        assert order == ["register", "run"]

    def test_server_freezes_process_id_before_registration(self):
        """process_id 由 server 入口在注册前生成并冻结（process_ 前缀）。"""
        captured: list[str] = []

        async def fake_register(**kwargs):
            captured.append(kwargs["process_id"])
            return _registered_process(process_id=kwargs["process_id"])

        def fake_run(handle, **_kwargs):
            async def gen():
                yield {"event": "done", "data": {"final_text": "ok"}}

            return gen()

        mock_service = MagicMock()
        mock_service.register_process = AsyncMock(side_effect=fake_register)
        mock_service.run_process = MagicMock(side_effect=fake_run)
        mock_service.close_process = AsyncMock()

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )

        assert response.status_code == 200
        assert len(captured) == 1
        assert captured[0].startswith("process_")
        assert captured[0] == mock_service.run_process.call_args.args[0].process_id


class TestChatRouter:
    def test_normal_chat_sse_events(self):
        """正常对话: token → done"""
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {"event": "token", "data": {"content": "Hello "}},
                {"event": "token", "data": {"content": "world!"}},
                {
                    "event": "done",
                    "data": {
                        "final_text": "Hello world!",
                        "mtp_iterations": 0,
                        "total_iterations": 1,
                    },
                },
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        # 完整事件顺序（router 透传 service 事件原样）
        assert event_types == ["token", "token", "done"]

    def test_mtp_chat_sse_events(self):
        """MTP 对话: token → mtp_start → mtp_result → token → done"""
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {"event": "token", "data": {"content": "Let me search. "}},
                {"event": "mtp_start", "data": {"verb": "SEARCH", "iteration": 1}},
                {
                    "event": "mtp_result",
                    "data": {"verb": "SEARCH", "status": "success", "iteration": 1},
                },
                {"event": "token", "data": {"content": "Found it!"}},
                {
                    "event": "done",
                    "data": {
                        "final_text": "Let me search. Found it!",
                        "mtp_iterations": 1,
                        "total_iterations": 2,
                    },
                },
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "search something", "agent_id": "test_agent"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert event_types == [
            "token",
            "mtp_start",
            "mtp_result",
            "token",
            "done",
        ]

    def test_stream_exception_emits_error_event(self):
        """流式中途抛异常时，router 应产出 error 事件"""
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {"event": "token", "data": {"content": "partial"}},
                exc=RuntimeError("LLM 调用失败"),
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "系统错误" in error_events[0]["data"]["message"]

    def test_command_result_sse_event_is_forwarded_as_own_event(self):
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {
                    "event": "command_result",
                    "data": {
                        "command_id": "system.clear",
                        "status": "completed",
                        "message": "cleared",
                        "client_action": {"type": "clear_chat"},
                    },
                },
                {
                    "event": "done",
                    "data": {
                        "status": "completed",
                        "command_id": "system.clear",
                    },
                },
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "/clear", "agent_id": "test_agent"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]
        assert event_types == ["command_result", "done"]
        assert "token" not in event_types
        assert events[0]["data"]["message"] == "cleared"
        assert events[0]["data"]["client_action"] == {"type": "clear_chat"}

    def test_stop_route_projects_cancel_result(self):
        mock_service = MagicMock()
        mock_service.cancel_process.return_value = MagicMock(
            process_id="process-1",
            cancelled=False,
            status="not_found",
            reason="user_requested",
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stop",
            json={"process_id": "process-1"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "process_id": "process-1",
            "cancelled": False,
            "status": "not_found",
            "reason": "user_requested",
        }
        # 取消以请求级 access context 发起（经网关认证签发）
        cancel_args, cancel_kwargs = mock_service.cancel_process.call_args
        assert cancel_args == ("process-1",)
        assert isinstance(cancel_kwargs["access"], WorkspaceAccessContext)

    def test_uuid_payload_is_serializable(self):
        mock_service = _wired_chat_service(
            lambda *args, **kw: _simple_stream(
                {
                    "event": "memory_refs",
                    "data": {
                        "memories": [{"id": uuid4(), "content": "hello"}],
                    },
                }
            )
        )

        app, _ = _create_test_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "test_agent"},
        )
        assert response.status_code == 200

        events = _parse_sse_events(response.text)
        memory_events = [e for e in events if e["event"] == "memory_refs"]
        assert len(memory_events) == 1
        memory_id = memory_events[0]["data"]["memories"][0]["id"]
        assert isinstance(memory_id, str)


def _simple_stream(*events, exc: Exception | None = None):
    """构造依次产出 events、可选抛出 exc 的异步事件流。"""

    async def gen():
        for event in events:
            yield event
        if exc is not None:
            raise exc

    return gen()


class TestChatDisconnect:
    @staticmethod
    def _direct_call(mock_service):
        """绕过 HTTP 栈直接调用 chat 路由（使用真实网关组合完成认证）。

        chat 路由只声明 service 与 principal_id 依赖：注册入口的认证在
        ``register_process`` 内完成，路由不再注入网关。
        """
        _, composition = make_server_access_overrides()
        return chat(
            body=ChatRequest(message="hello", agent_id="test_agent"),
            selection=RequestIdentitySelection(user_id=None, workspace_id=None),
            service=mock_service,
            principal_id=composition.principal.principal_id,
        )

    @staticmethod
    def _http_scope():
        """直接调用响应时使用的 http scope。

        ASGI 规定 http scope 必须带 ``method``；SSE 响应据此识别 HEAD 请求，
        缺少该键会在开始响应前失败。chat 路由只接受 POST。
        """
        return {"type": "http", "method": "POST"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cancelled_while", ["awaiting_event", "awaiting_send"])
    async def test_disconnect_closes_delivery_without_close_path(self, cancelled_while):
        """客户端断连：交付生成器被关闭（注册入口据此以断连原因请求停止），路由不调用关闭路径。

        两种取消时机：响应在等待下一项时被取消（取消直接穿过生成器），以及
        在等待 send 时被取消（生成器停在 yield，由响应收尾关闭）。
        """
        delivery_closed = asyncio.Event()

        async def delivery():
            try:
                yield {"event": "process_id", "data": {}}
                await asyncio.Event().wait()
            finally:
                delivery_closed.set()

        mock_service = _wired_chat_service(lambda *args, **kw: delivery())
        response = await self._direct_call(mock_service)
        first_chunk_sent = asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk_sent.set()
                if cancelled_while == "awaiting_send":
                    await asyncio.Event().wait()

        async def receive():
            await first_chunk_sent.wait()
            return {"type": "http.disconnect"}

        await asyncio.wait_for(response(self._http_scope(), receive, send), timeout=2)

        assert delivery_closed.is_set()
        mock_service.close_process.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trigger", ["shutdown_before_streaming", "send_raises_on_disconnect"])
    async def test_response_ending_before_iteration_still_closes_process(
        self, trigger, monkeypatch
    ):
        """注册成功、但响应在开始迭代生成器前就结束：进程仍经关闭路径收口。

        生成器体从未执行时其 ``finally`` 不会运行，关闭必须挂在响应自身的
        收尾上。两种触发：关停信号在注册期间到达（sse_starlette 在开始迭代
        前取消响应）；服务器对已断开连接的 send 抛出 ``OSError``（ASGI 2.4
        允许的行为）。
        """
        mock_service = _wired_chat_service(lambda *args, **kw: _simple_stream())
        response = await self._direct_call(mock_service)

        async def receive():
            # 客户端不发送断开消息：响应只由触发条件结束。
            await asyncio.Event().wait()

        if trigger == "shutdown_before_streaming":
            monkeypatch.setattr(AppStatus, "should_exit", True)

            async def send(message):
                await asyncio.sleep(0)

            await response(self._http_scope(), receive, send)
        else:

            async def send(message):
                raise OSError("client disconnected")

            with pytest.raises(OSError, match="client disconnected"):
                await response(self._http_scope(), receive, send)

        # 生成器体从未执行（未发起运行），进程仍被关闭且只关闭一次。
        mock_service.run_process.assert_not_called()
        mock_service.close_process.assert_awaited_once()
        closed = mock_service.close_process.await_args.args[0]
        assert closed.process_id == (mock_service.register_process.call_args.kwargs["process_id"])
