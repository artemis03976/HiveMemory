"""server 入口层统一身份解析测试（v0.6.2 B1 + A1 访问边界返工）。

覆盖 ``resolve_request_identity_claims`` 的合并/冲突规则与各 HTTP 入口的
身份上下文行为：user_id + workspace_id 构造认证前声明（不产出
``IdentityScope``）、body/query/header 冲突显式失败、未知 Workspace 拒绝、
非 Agent action 注入 system actor、Chat 缺失具体 Agent 显式失败；
``resolve_request_identity_scope`` 作为 /ingest 已知例外仍产出完整 scope。
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.server import deps
from hivememory.server.deps import (
    RequestIdentityClaims,
    RequestIdentitySelection,
    resolve_request_identity_claims,
    resolve_request_identity_scope,
)
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.topics import router as topics_router
from tests.helpers.workspace import make_server_access_overrides

# ─── 解析器单元行为 ──────────────────────────────────────────────────────────


class TestResolveRequestIdentityClaims:
    def test_header_selection_builds_claims(self):
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
        )
        assert claims.actor.user_id == "u1"
        # 非 Agent action 注入保留 system actor
        assert claims.actor.agent_id == SYSTEM_AGENT_ID
        assert claims.workspace.owner_user_id == "u1"
        assert claims.workspace.workspace_id == "main_workspace"

    def test_same_owner_constraint_holds_for_public_entry(self):
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id="u1", workspace_id="main_workspace"),
        )
        assert claims.actor.user_id == claims.workspace.owner_user_id

    def test_missing_selection_falls_back_at_single_entry(self):
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id=None, workspace_id=None),
        )
        assert claims.actor.user_id == "default"
        assert claims.workspace.workspace_id == "main_workspace"

    def test_session_id_travels_with_actor_claims(self):
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
            require_agent=True,
            agent_id="omni_doll",
            session_id="session-1",
        )
        assert claims.actor.agent_id == "omni_doll"
        assert claims.actor.session_id == "session-1"

    def test_claims_resolution_does_not_assemble_identity_scope(self):
        """不变量 1：认证前的声明解析不产出 IdentityScope。"""
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
        )
        assert isinstance(claims, RequestIdentityClaims)
        assert not isinstance(claims, IdentityScope)
        # 声明载体只有 actor 与进入 workspace 两个坐标，没有冻结的 scope
        assert {f.name for f in dataclasses.fields(RequestIdentityClaims)} == {
            "actor",
            "workspace",
        }
        assert isinstance(claims.workspace, WorkspaceIdentity)
        assert not hasattr(claims, "workspace_identity")

    def test_resolve_request_identity_scope_still_builds_scope_for_ingest(self):
        """/ingest 已知例外：scope 解析入口仍一次性组装完整 IdentityScope。"""
        scope = resolve_request_identity_scope(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
        )
        assert isinstance(scope, IdentityScope)
        assert scope.actor_identity.user_id == "u1"
        assert scope.workspace_identity.workspace_id == "main_workspace"

    def test_header_body_conflict_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="header-user", workspace_id=None),
                explicit_user_id="body-user",
            )
        assert exc_info.value.status_code == 409

    def test_workspace_conflict_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="u1", workspace_id="main_workspace"),
                explicit_workspace_id="other_workspace",
            )
        assert exc_info.value.status_code == 409

    def test_unknown_workspace_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="u1", workspace_id="ghost_workspace"),
            )
        assert exc_info.value.status_code == 404

    def test_agent_action_requires_concrete_agent(self):
        claims = resolve_request_identity_claims(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
            require_agent=True,
            agent_id="omni_doll",
        )
        assert claims.actor.agent_id == "omni_doll"

    def test_agent_action_missing_agent_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="u1", workspace_id=None),
                require_agent=True,
                agent_id=None,
            )
        assert exc_info.value.status_code == 400

    def test_agent_action_blank_agent_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="u1", workspace_id=None),
                require_agent=True,
                agent_id="   ",
            )
        assert exc_info.value.status_code == 400

    def test_agent_action_explicit_system_agent_rejected(self):
        """Agent action 显式使用保留 system：它表示"没有具体 Agent"，解析即拒绝（400）。

        注册入口保留同一检查（由 chat 路由测试覆盖）；此处钉住声明解析层的契约。
        """
        with pytest.raises(HTTPException) as exc_info:
            resolve_request_identity_claims(
                RequestIdentitySelection(user_id="u1", workspace_id=None),
                require_agent=True,
                agent_id=SYSTEM_AGENT_ID,
            )
        assert exc_info.value.status_code == 400


# ─── HTTP 入口行为 ───────────────────────────────────────────────────────────


def _registered_process_stub(**register_kwargs):
    """构造 ``register_process`` 返回的进程句柄 stub（只暴露 process_id）。"""
    return SimpleNamespace(process_id=register_kwargs["process_id"])


def _create_chat_app(mock_service, access=None):
    """创建 Chat 测试应用；``access`` 为可选共享的 (overrides, composition)。

    需要在请求后检查网关/guard 状态的测试必须复用同一组合对象（每次
    ``make_server_access_overrides()`` 都是新网关）。
    """
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: mock_service
    # /chat 与 /chat/stop 经统一认证网关认证：覆盖注入真实组合的网关。
    if access is None:
        access = make_server_access_overrides()
    overrides, composition = access
    app.dependency_overrides.update(overrides)
    return app, composition


def _create_topics_app():
    app = FastAPI()
    app.include_router(topics_router, prefix="/api/v1")

    app.dependency_overrides[deps.get_topic_service] = lambda: _TopicServiceStub()
    # topics 路由经统一认证网关取得请求级 context：覆盖注入真实组合的网关。
    overrides, _ = make_server_access_overrides(users=["u1", "query-user", "header-user"])
    app.dependency_overrides.update(overrides)
    return app


class _TopicServiceStub:
    async def list_active_topics(self, *, target_workspace, access):
        return []


class TestChatEntryIdentity:
    def test_missing_agent_id_fails_explicitly(self):
        """agent_id 字段缺失在 DTO 校验层即显式失败（422）。"""
        client = TestClient(_create_chat_app(MagicMock())[0])
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
        )
        assert response.status_code == 422

    def test_blank_agent_id_fails_explicitly(self):
        """空 agent_id 通过 DTO 校验后在身份解析入口显式失败（400）。"""
        client = TestClient(_create_chat_app(MagicMock())[0])
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "  "},
        )
        assert response.status_code == 400
        assert "agent_id" in response.json()["detail"]

    def test_identity_selection_comes_from_headers_only(self):
        """Chat body 不携带基础身份选择：actor/workspace 声明完全来自统一请求头。"""
        mock_service = MagicMock()
        mock_service.register_process = AsyncMock(
            side_effect=lambda **kwargs: _registered_process_stub(**kwargs)
        )

        async def fake_stream(handle, *, stream=True):
            yield {"event": "done", "data": {"final_text": "ok"}}

        mock_service.run_process = MagicMock(
            side_effect=lambda *args, **kw: fake_stream(*args, **kw)
        )
        mock_service.close_process = AsyncMock()
        app, _ = _create_chat_app(mock_service)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "omni_doll"},
            headers={"x-user-id": "u1", "x-workspace-id": "main_workspace"},
        )
        assert response.status_code == 200
        register_kwargs = mock_service.register_process.call_args.kwargs
        assert register_kwargs["actor"].user_id == "u1"
        assert register_kwargs["actor"].agent_id == "omni_doll"
        assert register_kwargs["workspace"].workspace_id == "main_workspace"

    def test_unknown_workspace_via_header_rejected(self):
        client = TestClient(_create_chat_app(MagicMock())[0])
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "omni_doll"},
            headers={"x-workspace-id": "ghost_workspace"},
        )
        assert response.status_code == 404


class TestStopEntryIdentity:
    @staticmethod
    def _stop_service_mock(composition, *, cancelled, status):
        """取消发生时快照请求级 context 的授予摘要；请求结束后它已失效。

        返回 ``(mock_service, summaries)``：``summaries`` 记录 cancel_process
        被调用时刻经认证网关取回的授予摘要（此时 context 尚未随请求收尾失效）。
        """

        summaries: list = []
        mock_service = MagicMock()

        def fake_cancel(process_id, *, access):
            summaries.append(composition.gateway.describe_context(access))
            return MagicMock(
                process_id=process_id,
                cancelled=cancelled,
                status=status,
                reason="user_requested",
            )

        mock_service.cancel_process = MagicMock(side_effect=fake_cancel)
        return mock_service, summaries

    def test_stop_without_identity_selection_uses_single_fallback(self):
        overrides, composition = make_server_access_overrides()
        mock_service, summaries = self._stop_service_mock(
            composition, cancelled=False, status="not_found"
        )
        app, _ = _create_chat_app(mock_service, (overrides, composition))
        client = TestClient(app)

        response = client.post("/api/v1/chat/stop", json={"process_id": "process-1"})

        assert response.status_code == 200
        assert response.json()["cancelled"] is False
        # stop 不是 Agent action：请求以缺省回退的 (default, system) 声明认证
        assert len(summaries) == 1
        assert summaries[0].actor_user_id == "default"
        assert summaries[0].agent_id == SYSTEM_AGENT_ID
        # 请求结束后请求级 context 已失效
        access = mock_service.cancel_process.call_args.kwargs["access"]
        assert composition.gateway.describe_context(access) is None

    def test_stop_uses_header_selection_for_ownership_check(self):
        overrides, composition = make_server_access_overrides(users=["u1"])
        mock_service, summaries = self._stop_service_mock(
            composition, cancelled=True, status="stop_requested"
        )
        app, _ = _create_chat_app(mock_service, (overrides, composition))
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stop",
            json={"process_id": "process-1"},
            headers={"x-user-id": "u1", "x-workspace-id": "main_workspace"},
        )

        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        # 取消发生时请求级 context 仍有效，且以 header 选择的 (u1, system) 声明签发
        assert len(summaries) == 1
        assert summaries[0].actor_user_id == "u1"
        assert summaries[0].agent_id == SYSTEM_AGENT_ID
        # 请求结束后请求级 context 已失效
        access = mock_service.cancel_process.call_args.kwargs["access"]
        assert composition.gateway.describe_context(access) is None


class TestTopicsEntryIdentity:
    def test_query_user_id_without_header_is_respected(self):
        app = _create_topics_app()
        client = TestClient(app)

        response = client.get("/api/v1/topics", params={"user_id": "query-user"})

        assert response.status_code == 200

    def test_query_user_id_conflicting_with_header_rejected(self):
        app = _create_topics_app()
        client = TestClient(app)

        response = client.get(
            "/api/v1/topics",
            params={"user_id": "query-user"},
            headers={"x-user-id": "header-user"},
        )

        assert response.status_code == 409


class TestRequestAccessResolvedOncePerRequest:
    """同一请求内声明与请求级 context 只解析/认证一次，服务收到同一坐标。"""

    @pytest.mark.asyncio
    async def test_single_request_access_per_request(self):
        captured: list[SimpleNamespace] = []

        app = FastAPI()
        app.include_router(topics_router, prefix="/api/v1")

        class AccessCapturingStub:
            async def list_active_topics(self, *, target_workspace, access):
                captured.append(SimpleNamespace(target_workspace=target_workspace, access=access))
                return []

        app.dependency_overrides[deps.get_topic_service] = lambda: AccessCapturingStub()

        overrides, _ = make_server_access_overrides(users=["u1"])
        app.dependency_overrides.update(overrides)

        client = TestClient(app)
        client.get("/api/v1/topics", headers={"x-user-id": "u1"})

        assert len(captured) == 1
        assert captured[0].target_workspace.owner_user_id == "u1"
        assert captured[0].target_workspace.workspace_id == "main_workspace"
        assert isinstance(captured[0].access, WorkspaceAccessContext)
