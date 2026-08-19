"""TopicApplicationService 委托测试。"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


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

        await service.list_active_topics(user_id="u1")

        # 公共入口只在此处解析 main Workspace，Patchouli 不再接收裸 identity。
        handler.assert_awaited_once()
        access_context = handler.await_args.kwargs["access_context"]
        assert access_context.actor_identity.user_id == "u1"
        assert access_context.workspace_identity.workspace_id == "main_workspace"

    @pytest.mark.asyncio
    async def test_settle_topic_uses_public_route(self, service, bus):
        task = MagicMock(task_id="memtask_1", topic_id="t1")
        handler = AsyncMock(return_value=task)
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)

        result = await service.settle_topic(user_id="u1", topic_id="t1")

        assert result == {"success": True, "task_id": "memtask_1", "topic_id": "t1"}
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["access_context"].workspace_identity.owner_user_id == "u1"

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value={"success": True, "message": "话题 t1 已删除"})
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)

        await service.evict_topic(user_id="u1", topic_id="t1")

        # evict_topic 是纯透传；约束力来自路由与参数
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["access_context"].workspace_identity.owner_user_id == "u1"
