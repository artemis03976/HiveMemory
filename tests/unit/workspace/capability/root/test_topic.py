"""root Topic 列表、结算与驱逐的公共路由和业务结果测试。"""

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.workspace.capability.root.topic import RootTopicService
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)


@pytest.fixture
def workspace():
    """管理入口的目标 workspace（等于 context 的驻留 workspace）。"""
    return make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


class TestRootTopicService:
    """root 管理的领域输入与公开路由契约。"""

    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def composition(self, workspace) -> AccessTestComposition:
        """能力层 root 授权经共享授权者完成：为管理入口登记 system 记录。"""
        return make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1", agent_id="system", allowed_operations=frozenset()
                )
            ],
            default_workspace=workspace,
        )

    @pytest.fixture
    def service(self, bus, composition):
        return RootTopicService(global_bus=bus, operation_authorizer=composition.authorizer)

    @pytest.mark.asyncio
    async def test_list_active_topics_uses_public_route(self, service, bus, composition, workspace):
        handler = AsyncMock(return_value=["snapshot"])
        bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, handler)
        access = await composition.authenticate(agent_id="system", user_id="u1")

        await service.list_active_topics(target_workspace=workspace, access=access)

        # 传给 Patchouli 的 scope 来自 guard：已认证 actor + 目标 workspace；
        # context 不向下游传递，路由不接收 access。
        handler.assert_awaited_once()
        scope = handler.await_args.kwargs["identity_scope"]
        assert scope.actor_identity.user_id == "u1"
        assert scope.actor_identity.agent_id == "system"
        assert scope.workspace_identity == workspace
        assert "access" not in handler.await_args.kwargs

    @pytest.mark.asyncio
    async def test_settle_topic_uses_public_route(self, service, bus, composition, workspace):
        from hivememory.patchouli.contracts.topic_management import TopicSettleResult

        handler = AsyncMock(
            return_value=TopicSettleResult(
                topic_id="t1",
                generation_task_id="memtask_1",
            )
        )
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)
        access = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.settle_topic(
            target_workspace=workspace, topic_id="t1", access=access
        )

        assert result.topic_id == "t1"
        assert result.generation_task_id == "memtask_1"
        assert result.generation_submitted is True
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["identity_scope"].workspace_identity == workspace
        assert "access" not in handler.await_args.kwargs

    @pytest.mark.asyncio
    async def test_settle_topic_without_generation_still_reports_success(
        self, service, bus, composition, workspace
    ):
        """无任务时的 settle（空话题/材料被过滤）不被误报为生命周期失败。"""
        from hivememory.patchouli.contracts.topic_management import TopicSettleResult

        handler = AsyncMock(
            return_value=TopicSettleResult(
                topic_id="t1",
            )
        )
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)
        access = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.settle_topic(
            target_workspace=workspace, topic_id="t1", access=access
        )

        assert result.topic_id == "t1"
        assert result.generation_task_id is None
        assert result.generation_submitted is False

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus, composition, workspace):
        from hivememory.patchouli.contracts.topic_management import TopicEvictionResult

        handler = AsyncMock(return_value=TopicEvictionResult(topic_id="t1", removed=True))
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)
        access = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.evict_topic(target_workspace=workspace, topic_id="t1", access=access)

        # evict_topic 是纯透传；约束力来自路由与参数
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["identity_scope"].workspace_identity == workspace
        assert result.removed is True
