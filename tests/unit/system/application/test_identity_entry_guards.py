"""身份入口收敛守卫测试（v0.6.2 B1）。

守卫目标：
1. ``system/application`` 公共服务方法签名不再出现 ``user_id: str`` 裸参数；
2. Chat 拒绝保留 ``system`` actor；
3. 取消路径复用创建时冻结 scope：跨 user/workspace 的取消不可见；
4. 非 Agent action 的 scope 由 server 注入 ``SYSTEM_AGENT_ID``。
"""

import inspect

import pytest

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import WorkspaceDomainError
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import MemoryApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.topic_service import TopicApplicationService
from tests.helpers.workspace import make_identity_scope, make_management_identity_scope

_APPLICATION_SERVICES = (
    AgentApplicationService,
    ChatApplicationService,
    MemoryApplicationService,
    PassiveIngressService,
    TopicApplicationService,
)


def _public_methods(cls) -> list[str]:
    return [
        name
        for name, fn in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    ]


class TestServiceSignatureGuard:
    """应用服务入口不得再以裸 user_id 字符串作为公共签名。"""

    @pytest.mark.parametrize("service_cls", _APPLICATION_SERVICES)
    def test_public_methods_do_not_accept_bare_user_id(self, service_cls):
        for name in _public_methods(service_cls):
            signature = inspect.signature(getattr(service_cls, name))
            assert "user_id" not in signature.parameters, (
                f"{service_cls.__name__}.{name} 仍接受裸 user_id 参数，"
                "应改为 identity_scope: IdentityScope 唯一入口"
            )

    @pytest.mark.parametrize("service_cls", _APPLICATION_SERVICES)
    def test_identity_bearing_methods_use_identity_scope(self, service_cls):
        """凡携带身份语义的公共方法（含 workspace/agent 关键字）必须走 identity_scope。"""
        identity_keywords = ("workspace_id", "agent_id", "owner_user_id")
        for name in _public_methods(service_cls):
            signature = inspect.signature(getattr(service_cls, name))
            params = signature.parameters
            if any(keyword in params for keyword in identity_keywords):
                assert "identity_scope" in params, (
                    f"{service_cls.__name__}.{name} 携带身份语义却未使用 identity_scope"
                )


class TestChatScopedIdentityGuard:
    """Chat 必须由具体 Agent 执行，不能静默回退到 system。"""

    @pytest.mark.asyncio
    async def test_chat_scoped_rejects_system_actor(self):
        from unittest.mock import AsyncMock

        service = ChatApplicationService(global_bus=AsyncMock())

        with pytest.raises(WorkspaceDomainError):
            await service.chat_scoped(
                user_message="hello",
                identity_scope=make_management_identity_scope(user_id="u1"),
                interaction_id="interaction-system-1",
            )

    @pytest.mark.asyncio
    async def test_chat_stream_scoped_rejects_system_actor(self):
        from unittest.mock import AsyncMock

        service = ChatApplicationService(global_bus=AsyncMock())

        with pytest.raises(WorkspaceDomainError):
            async for _ in service.chat_stream_scoped(
                user_message="hello",
                identity_scope=make_management_identity_scope(user_id="u1"),
                interaction_id="interaction-system-2",
            ):
                pass


class TestCancelUsesFrozenScope:
    """取消路径：请求方 scope 只用于 owner/workspace 校验。"""

    @staticmethod
    def _make_service():
        from unittest.mock import AsyncMock

        return ChatApplicationService(global_bus=AsyncMock())

    @pytest.mark.asyncio
    async def test_cancel_across_users_returns_not_found(self):
        service = self._make_service()
        run_scope = make_identity_scope(user_id="owner", agent_id="omni_doll")
        stream = service.chat_stream_scoped(
            user_message="hello",
            identity_scope=run_scope,
            interaction_id="interaction-owner-1",
        )
        # 消费 generation_id 事件后保持 run 存活
        first = await stream.__anext__()
        assert first["event"] == "generation_id"

        result = service.cancel_generation_scoped(
            "interaction-owner-1",
            identity_scope=make_identity_scope(user_id="other", agent_id="omni_doll"),
        )
        assert result.cancelled is False
        assert result.status == "not_found"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_across_workspaces_returns_not_found(self):
        from hivememory.core.models import build_internal_identity_scope

        service = self._make_service()
        run_scope = make_identity_scope(user_id="owner", agent_id="omni_doll")
        stream = service.chat_stream_scoped(
            user_message="hello",
            identity_scope=run_scope,
            interaction_id="interaction-owner-2",
        )
        await stream.__anext__()

        other_workspace_scope = build_internal_identity_scope(
            run_scope.actor_identity,
            "isolation_workspace",
        )
        result = service.cancel_generation_scoped(
            "interaction-owner-2",
            identity_scope=other_workspace_scope,
        )
        assert result.cancelled is False
        assert result.status == "not_found"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_without_agent_selection_reuses_frozen_scope(self):
        """取消不携带 agent 选择：与 run 同 user/workspace 即可取消成功。"""
        service = self._make_service()
        run_scope = make_identity_scope(user_id="owner", agent_id="omni_doll")
        stream = service.chat_stream_scoped(
            user_message="hello",
            identity_scope=run_scope,
            interaction_id="interaction-owner-3",
        )
        await stream.__anext__()

        # 请求方 scope 为管理语义（system actor），agent 维度与 run 不同也不影响校验
        result = service.cancel_generation_scoped(
            "interaction-owner-3",
            identity_scope=make_management_identity_scope(user_id="owner"),
        )
        assert result.cancelled is True

        await stream.aclose()


class TestManagementScopeUsesSystemActor:
    """非 Agent action 的测试 scope 构造器必须注入保留 system actor。"""

    def test_management_scope_carries_system_agent(self):
        scope = make_management_identity_scope(user_id="u1")
        assert scope.actor_identity.agent_id == SYSTEM_AGENT_ID
        assert scope.workspace_identity.owner_user_id == "u1"

    def test_management_scope_differs_from_agent_scope(self):
        assert (
            make_identity_scope(user_id="u1", agent_id="omni_doll").actor_identity.agent_id
            != SYSTEM_AGENT_ID
        )
