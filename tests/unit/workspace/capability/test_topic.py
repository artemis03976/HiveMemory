"""Topic 能力（``workspace.capability.topic``）管理用例测试。

能力层是授权点（A1 访问边界返工第 4.5 节）：方法只接收访问 context 与目标
workspace，``IdentityScope`` 由 guard 组装并传给 Patchouli；读取（list 绑定
``resource.read``）与生命周期变更（settle/evict 绑定 ``management.topic``）
的授权在本层、路由调用前执行，白名单缺少对应 operation 时以
``OperationDeniedError`` 拒绝，不触达 Patchouli 路由。
"""

from unittest.mock import AsyncMock

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import OperationDeniedError
from hivememory.workspace.capability.topic import TopicApplicationService
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


class TestTopicApplicationService:
    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def composition(self, workspace) -> AccessTestComposition:
        """能力层 operation 授权经共享 guard 完成：为管理入口登记 system 记录。"""
        return make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="system")],
            default_workspace=workspace,
        )

    @pytest.fixture
    def service(self, bus, composition):
        return TopicApplicationService(global_bus=bus, access_guard=composition.guard)

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

    @pytest.mark.asyncio
    async def test_list_active_topics_without_resource_read_denied_before_route(
        self, service, bus, composition, workspace
    ):
        """白名单缺少 ``resource.read`` 时列表在本层拒绝，不触达 Patchouli 路由。"""
        handler = AsyncMock(return_value=["snapshot"])
        bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, handler)
        observing_only = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id="system",
                    allowed_operations=frozenset({WorkspaceOperation.MANAGEMENT_TOPIC}),
                )
            ],
            default_workspace=workspace,
        )
        access = await observing_only.authenticate(agent_id="system", user_id="u1")
        denied_service = TopicApplicationService(global_bus=bus, access_guard=observing_only.guard)

        with pytest.raises(OperationDeniedError) as exc_info:
            await denied_service.list_active_topics(target_workspace=workspace, access=access)

        assert exc_info.value.details["reason"] == "operation_not_allowed"
        assert exc_info.value.details["operation"] == WorkspaceOperation.RESOURCE_READ.value
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(
                lambda service, workspace, access: service.settle_topic(
                    target_workspace=workspace, topic_id="t1", access=access
                ),
                id="settle_topic",
            ),
            pytest.param(
                lambda service, workspace, access: service.evict_topic(
                    target_workspace=workspace, topic_id="t1", access=access
                ),
                id="evict_topic",
            ),
        ],
    )
    async def test_lifecycle_change_without_management_topic_denied_before_route(
        self, bus, workspace, invoke
    ):
        """白名单缺少 ``management.topic`` 时生命周期变更在本层拒绝，不触达路由。"""
        handler = AsyncMock(return_value=None)
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)
        read_only = make_access_composition(
            [
                make_actor_access_record(
                    owner_user_id="u1",
                    agent_id="system",
                    allowed_operations=frozenset({WorkspaceOperation.RESOURCE_READ}),
                )
            ],
            default_workspace=workspace,
        )
        access = await read_only.authenticate(agent_id="system", user_id="u1")
        service = TopicApplicationService(global_bus=bus, access_guard=read_only.guard)

        with pytest.raises(OperationDeniedError) as exc_info:
            await invoke(service, workspace, access)

        assert exc_info.value.details["reason"] == "operation_not_allowed"
        assert exc_info.value.details["operation"] == WorkspaceOperation.MANAGEMENT_TOPIC.value
        handler.assert_not_awaited()
