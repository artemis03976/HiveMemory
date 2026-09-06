"""TopicApplicationService 委托测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.workspace import make_management_identity_scope


@pytest.fixture
def passive_config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler
    return config


class TestTopicApplicationService:
    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def service(self, bus, passive_config):
        return TopicApplicationService(
            global_bus=bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_list_active_topics_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value=["snapshot"])
        bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, handler)

        identity_scope = make_management_identity_scope(user_id="u1")
        await service.list_active_topics(identity_scope=identity_scope)

        # 服务只透传 server 冻结的同一 scope 实例，不再自行解析身份。
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["identity_scope"] is identity_scope
        assert identity_scope.actor_identity.user_id == "u1"
        assert identity_scope.workspace_identity.workspace_id == "main_workspace"

    @pytest.mark.asyncio
    async def test_settle_topic_uses_public_route(self, service, bus):
        from hivememory.patchouli.contracts.topic_management import TopicSettleResult
        handler = AsyncMock(
            return_value=TopicSettleResult(
                topic_id="t1",
                generation_task_id="memtask_1",
            )
        )
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)

        identity_scope = make_management_identity_scope(user_id="u1")
        result = await service.settle_topic(identity_scope=identity_scope, topic_id="t1")

        assert result.topic_id == "t1"
        assert result.generation_task_id == "memtask_1"
        assert result.generation_submitted is True
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert (
            handler.await_args.kwargs["identity_scope"].workspace_identity.owner_user_id
            == "u1"
        )

    @pytest.mark.asyncio
    async def test_settle_topic_without_generation_still_reports_success(self, service, bus):
        """无任务时的 settle（空话题/材料被过滤）不被误报为生命周期失败。"""
        from hivememory.patchouli.contracts.topic_management import TopicSettleResult
        handler = AsyncMock(
            return_value=TopicSettleResult(
                topic_id="t1",
            )
        )
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)

        result = await service.settle_topic(
            identity_scope=make_management_identity_scope(user_id="u1"),
            topic_id="t1",
        )

        assert result.topic_id == "t1"
        assert result.generation_task_id is None
        assert result.generation_submitted is False

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus):
        from hivememory.patchouli.contracts.topic_management import TopicEvictionResult
        handler = AsyncMock(return_value=TopicEvictionResult(topic_id="t1", removed=True))
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)

        identity_scope = make_management_identity_scope(user_id="u1")
        result = await service.evict_topic(identity_scope=identity_scope, topic_id="t1")

        # evict_topic 是纯透传；约束力来自路由与参数
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert (
            handler.await_args.kwargs["identity_scope"].workspace_identity.owner_user_id
            == "u1"
        )
        assert result.removed is True
