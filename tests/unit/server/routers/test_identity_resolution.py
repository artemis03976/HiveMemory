"""server 入口层统一身份解析测试（v0.6.2 B1）。

覆盖 ``resolve_request_identity_scope`` 的合并/冲突规则与各 HTTP 入口的
身份上下文行为：user_id + workspace_id 构造 scope、same-owner 校验、
未知 Workspace 拒绝、body/query/header 冲突显式失败、非 Agent action
注入 system actor、Chat 缺失具体 Agent 显式失败。
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.server import deps
from hivememory.server.deps import (
    RequestIdentitySelection,
    resolve_request_identity_scope,
)
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.topics import router as topics_router


# ─── 解析器单元行为 ──────────────────────────────────────────────────────────


class TestResolveRequestIdentityScope:
    def test_header_selection_builds_scope(self):
        scope = resolve_request_identity_scope(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
        )
        assert scope.actor_identity.user_id == "u1"
        # 非 Agent action 注入保留 system actor
        assert scope.actor_identity.agent_id == SYSTEM_AGENT_ID
        assert scope.workspace_identity.owner_user_id == "u1"
        assert scope.workspace_identity.workspace_id == "main_workspace"

    def test_same_owner_constraint_holds_for_public_entry(self):
        scope = resolve_request_identity_scope(
            RequestIdentitySelection(user_id="u1", workspace_id="main_workspace"),
        )
        assert scope.actor_identity.user_id == scope.workspace_identity.owner_user_id

    def test_missing_selection_falls_back_at_single_entry(self):
        scope = resolve_request_identity_scope(
            RequestIdentitySelection(user_id=None, workspace_id=None),
        )
        assert scope.actor_identity.user_id == "default"
        assert scope.workspace_identity.workspace_id == "main_workspace"

    def test_header_body_conflict_rejected(self):
        with pytest.raises(Exception) as exc_info:
            resolve_request_identity_scope(
                RequestIdentitySelection(user_id="header-user", workspace_id=None),
                explicit_user_id="body-user",
            )
        assert exc_info.value.status_code == 409

    def test_workspace_conflict_rejected(self):
        with pytest.raises(Exception) as exc_info:
            resolve_request_identity_scope(
                RequestIdentitySelection(user_id="u1", workspace_id="main_workspace"),
                explicit_workspace_id="other_workspace",
            )
        assert exc_info.value.status_code == 409

    def test_unknown_workspace_rejected(self):
        with pytest.raises(Exception) as exc_info:
            resolve_request_identity_scope(
                RequestIdentitySelection(user_id="u1", workspace_id="ghost_workspace"),
            )
        assert exc_info.value.status_code == 404

    def test_agent_action_requires_concrete_agent(self):
        scope = resolve_request_identity_scope(
            RequestIdentitySelection(user_id="u1", workspace_id=None),
            require_agent=True,
            agent_id="omni_doll",
        )
        assert scope.actor_identity.agent_id == "omni_doll"

    def test_agent_action_missing_agent_rejected(self):
        with pytest.raises(Exception) as exc_info:
            resolve_request_identity_scope(
                RequestIdentitySelection(user_id="u1", workspace_id=None),
                require_agent=True,
                agent_id=None,
            )
        assert exc_info.value.status_code == 400

    def test_agent_action_blank_agent_rejected(self):
        with pytest.raises(Exception) as exc_info:
            resolve_request_identity_scope(
                RequestIdentitySelection(user_id="u1", workspace_id=None),
                require_agent=True,
                agent_id="   ",
            )
        assert exc_info.value.status_code == 400


# ─── HTTP 入口行为 ───────────────────────────────────────────────────────────


def _create_chat_app(mock_service) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    app.dependency_overrides[deps.get_chat_service] = lambda: mock_service
    return app


def _create_topics_app() -> FastAPI:
    app = FastAPI()
    app.include_router(topics_router, prefix="/api/v1")

    bus = MagicMock()
    handler = AsyncMock(return_value=[])
    from hivememory.system.contracts.routes import GlobalRoutes

    bus.request = handler
    app.dependency_overrides[deps.get_topic_service] = lambda: _TopicServiceStub(bus)
    return app


class _TopicServiceStub:
    def __init__(self, bus):
        self._bus = bus

    async def list_active_topics(self, *, identity_scope):
        return await self._bus.request(
            "topic.list_active", identity_scope=identity_scope
        )


class TestChatEntryIdentity:
    def test_missing_agent_id_fails_explicitly(self):
        """agent_id 字段缺失在 DTO 校验层即显式失败（422）。"""
        client = TestClient(_create_chat_app(MagicMock()))
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
        )
        assert response.status_code == 422

    def test_blank_agent_id_fails_explicitly(self):
        """空 agent_id 通过 DTO 校验后在身份解析入口显式失败（400）。"""
        client = TestClient(_create_chat_app(MagicMock()))
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "  "},
        )
        assert response.status_code == 400
        assert "agent_id" in response.json()["detail"]

    def test_identity_selection_comes_from_headers_only(self):
        """Chat body 不携带基础身份选择：scope 完全由统一请求头冻结。"""
        mock_service = MagicMock()

        async def fake_stream(**kwargs):
            yield {"event": "done", "data": {"final_text": "ok"}}

        mock_service.chat_stream_scoped = MagicMock(
            side_effect=lambda **kw: fake_stream(**kw)
        )
        client = TestClient(_create_chat_app(mock_service))

        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "omni_doll"},
            headers={"x-user-id": "u1", "x-workspace-id": "main_workspace"},
        )
        assert response.status_code == 200
        scope = mock_service.chat_stream_scoped.call_args.kwargs["identity_scope"]
        assert scope.actor_identity.user_id == "u1"
        assert scope.actor_identity.agent_id == "omni_doll"
        assert scope.workspace_identity.workspace_id == "main_workspace"

    def test_unknown_workspace_via_header_rejected(self):
        client = TestClient(_create_chat_app(MagicMock()))
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "agent_id": "omni_doll"},
            headers={"x-workspace-id": "ghost_workspace"},
        )
        assert response.status_code == 404


class TestStopEntryIdentity:
    def test_stop_without_identity_selection_uses_single_fallback(self):
        mock_service = MagicMock()
        mock_service.cancel_generation_scoped.return_value = MagicMock(
            generation_id="gen-1",
            cancelled=False,
            status="not_found",
            reason="user_requested",
        )
        client = TestClient(_create_chat_app(mock_service))

        response = client.post("/api/v1/chat/stop", json={"generation_id": "gen-1"})

        assert response.status_code == 200
        scope = mock_service.cancel_generation_scoped.call_args.kwargs["identity_scope"]
        # stop 不是 Agent action：actor 为保留 system
        assert scope.actor_identity.agent_id == SYSTEM_AGENT_ID
        assert scope.actor_identity.user_id == "default"

    def test_stop_uses_header_selection_for_ownership_check(self):
        mock_service = MagicMock()
        mock_service.cancel_generation_scoped.return_value = MagicMock(
            generation_id="gen-1",
            cancelled=True,
            status="stop_requested",
            reason="user_requested",
        )
        client = TestClient(_create_chat_app(mock_service))

        response = client.post(
            "/api/v1/chat/stop",
            json={"generation_id": "gen-1"},
            headers={"x-user-id": "u1", "x-workspace-id": "main_workspace"},
        )

        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        scope = mock_service.cancel_generation_scoped.call_args.kwargs["identity_scope"]
        assert scope.actor_identity.user_id == "u1"
        assert scope.actor_identity.agent_id == SYSTEM_AGENT_ID


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


class TestScopeResolvedOncePerRequest:
    """同一请求内 scope 只构造一次，并在服务链路中保持同一实例。"""

    @pytest.mark.asyncio
    async def test_dependency_cache_yields_single_scope_instance(self):
        captured: list[object] = []

        app = FastAPI()
        app.include_router(topics_router, prefix="/api/v1")

        class ScopeCapturingStub:
            async def list_active_topics(self, *, identity_scope):
                captured.append(identity_scope)
                return []

        app.dependency_overrides[deps.get_topic_service] = lambda: ScopeCapturingStub()

        client = TestClient(app)
        client.get("/api/v1/topics", headers={"x-user-id": "u1"})

        assert len(captured) == 1
        scope = captured[0]
        assert scope.actor_identity.user_id == "u1"
        assert scope.workspace_identity.workspace_id == "main_workspace"
