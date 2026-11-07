"""root Topic 列表、结算与驱逐的公共路由和业务结果，以及为一轮确定话题的测试。"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkingActor
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.core.protocol.gateway import GatewayDecisionOutcome, GatewayIngressMode
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.workspace.capability.root.passive_routing import PassiveTurnRouting
from hivememory.workspace.capability.root.topic import DeterminedTopic, RootTopicService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import (
    AccessTestComposition,
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)


@pytest.fixture
def workspace():
    """管理入口的目标 workspace（等于工作 actor 的驻留 workspace）。"""
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
        bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, handler)
        actor = await composition.authenticate(agent_id="system", user_id="u1")

        await service.list_active_topics(target_workspace=workspace, actor=actor)

        # 传给 Patchouli 的 scope 来自 guard：已认证 actor + 目标 workspace；
        # 工作 actor 不向下游传递，路由不接收它。
        handler.assert_awaited_once()
        scope = handler.await_args.kwargs["identity_scope"]
        assert scope.actor_identity.user_id == "u1"
        assert scope.actor_identity.agent_id == "system"
        assert scope.workspace_identity == workspace
        assert "actor" not in handler.await_args.kwargs

    @pytest.mark.asyncio
    async def test_settle_topic_uses_public_route(self, service, bus, composition, workspace):
        from hivememory.patchouli.contracts.topic_management import TopicSettleResult

        handler = AsyncMock(
            return_value=TopicSettleResult(
                topic_id="t1",
                generation_task_id="memtask_1",
            )
        )
        bus.register(PatchouliRoutes.MANUAL_SETTLE_TOPIC, handler)
        actor = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.settle_topic(target_workspace=workspace, topic_id="t1", actor=actor)

        assert result.topic_id == "t1"
        assert result.generation_task_id == "memtask_1"
        assert result.generation_submitted is True
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["identity_scope"].workspace_identity == workspace
        assert "actor" not in handler.await_args.kwargs

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
        bus.register(PatchouliRoutes.MANUAL_SETTLE_TOPIC, handler)
        actor = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.settle_topic(target_workspace=workspace, topic_id="t1", actor=actor)

        assert result.topic_id == "t1"
        assert result.generation_task_id is None
        assert result.generation_submitted is False

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus, composition, workspace):
        from hivememory.patchouli.contracts.topic_management import TopicEvictionResult

        handler = AsyncMock(return_value=TopicEvictionResult(topic_id="t1", removed=True))
        bus.register(PatchouliRoutes.EVICT_TOPIC, handler)
        actor = await composition.authenticate(agent_id="system", user_id="u1")

        result = await service.evict_topic(target_workspace=workspace, topic_id="t1", actor=actor)

        # evict_topic 是纯透传；约束力来自路由与参数
        handler.assert_awaited_once()
        assert handler.await_args.kwargs["topic_id"] == "t1"
        assert handler.await_args.kwargs["identity_scope"].workspace_identity == workspace
        assert result.removed is True


_NEW_TOPIC_DECISION = make_gateway_decision(
    target_topic_id="NEW_TOPIC", new_topic_title="部署", new_topic_summary="讨论部署"
)
_SYSTEM_SCOPE = IdentityScope(
    actor_identity=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
    workspace_identity=make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace"),
)
_ROUTING = PassiveTurnRouting(message="测试挂了")


@dataclass
class _DetermineHarness:
    """被测 root 话题 API、提供这一轮进度的提交服务，以及 Gateway 与话题准备替身收到的请求。

    应用函数把交互应用到请求的话题；``apply_failures`` 中的异常依次让应用失败，
    ``apply_gate`` 清除时应用等它放行。话题准备把 ``NEW_TOPIC`` 建立为
    ``topic-new``，已有话题原样返回。
    """

    service: RootTopicService
    submissions: InteractionSubmissionService
    actor: WorkingActor
    gateway_calls: list[dict] = field(default_factory=list)
    prepared: list[dict] = field(default_factory=list)
    apply_failures: list[Exception] = field(default_factory=list)
    apply_gate: asyncio.Event = field(default_factory=asyncio.Event)

    async def determine(self, topic_source):
        return await self.service.determine_topic(
            target_workspace=_SYSTEM_SCOPE.workspace_identity,
            interaction_id="import_1",
            topic_source=topic_source,
            actor=self.actor,
        )

    async def submit(self, requested_topic_id, *, timeout_seconds=2.0):
        """这一轮已由导入提交：经 Patchouli 的公开提交服务提交并等待应用。"""
        return await self.submissions.submit_interaction(
            identity_scope=_SYSTEM_SCOPE,
            participant=ActorIdentity(user_id="u1", agent_id="claude_code"),
            payload=make_interaction_payload("导入的一轮", "回复"),
            requested_topic_id=requested_topic_id,
            interaction_id="import_1",
            origin="import_bus",
            timeout_seconds=timeout_seconds,
        )


@pytest_asyncio.fixture
async def determine_harness() -> AsyncIterator[_DetermineHarness]:
    """真实 root 话题 API、操作授权者与内存提交队列；Gateway 与话题准备在边界之外。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)],
        default_workspace=_SYSTEM_SCOPE.workspace_identity,
    )
    bus = GlobalSystemBus()
    ref: list[_DetermineHarness] = []

    async def apply(payload, **kwargs):
        await ref[0].apply_gate.wait()
        if ref[0].apply_failures:
            raise ref[0].apply_failures.pop(0)
        return kwargs["target_topic_id"]

    queue = InteractionSubmissionQueue(apply)
    submissions = InteractionSubmissionService(interaction_queue=queue)
    bus.register(PatchouliRoutes.INTERACTION_STATUS, submissions.interaction_status)

    async def gateway(message, **kwargs):
        ref[0].gateway_calls.append({"message": message, **kwargs})
        return GatewayDecisionOutcome(decision=_NEW_TOPIC_DECISION)

    async def prepare_topic(**kwargs):
        ref[0].prepared.append(kwargs)
        target = kwargs["target_topic_id"]
        return "topic-new" if target == "NEW_TOPIC" else target

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.PREPARE_TOPIC, prepare_topic)
    ref.append(
        _DetermineHarness(
            service=RootTopicService(
                bus, operation_authorizer=composition.authorizer, gateway_request_timeout_ms=1234
            ),
            submissions=submissions,
            actor=await composition.authenticate_system(),
        )
    )
    ref[0].apply_gate.set()
    await queue.start()
    try:
        yield ref[0]
    finally:
        ref[0].apply_gate.set()
        await queue.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_prepare", "expected"),
    [
        (
            make_gateway_decision(target_topic_id="topic-1", new_topic_title="不应使用"),
            ("topic-1", None, None),
            DeterminedTopic(topic_id="topic-1", is_new=False),
        ),
        (
            _NEW_TOPIC_DECISION,
            ("NEW_TOPIC", "部署", "讨论部署"),
            DeterminedTopic(topic_id="topic-new", is_new=True),
        ),
    ],
    ids=["existing-topic", "new-topic"],
)
async def test_gateway_decision_confirms_or_creates_its_topic_without_routing(
    determine_harness, decision, expected_prepare, expected
):
    """任务进程给出的决定直接交给话题准备：已有话题确认，新话题以标题与摘要建立；不路由。"""
    determined = await determine_harness.determine(decision)

    assert determined == expected
    [prepared] = determine_harness.prepared
    assert prepared["identity_scope"] == _SYSTEM_SCOPE
    assert (
        prepared["target_topic_id"],
        prepared["new_topic_title"],
        prepared["new_topic_summary"],
    ) == expected_prepare
    assert determine_harness.gateway_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("apply_failed", [False, True], ids=["not-submitted", "apply-failed"])
async def test_unsubmitted_or_failed_turn_is_routed_then_prepared(determine_harness, apply_failed):
    """未提交或应用失败的一轮以 PASSIVE_MEMORY 路由，再按路由结果建立话题。"""
    if apply_failed:
        determine_harness.apply_failures.append(RuntimeError("topic evicted"))
        assert (await determine_harness.submit("topic-old")).state == "failed"

    determined = await determine_harness.determine(_ROUTING)

    assert determined == DeterminedTopic(topic_id="topic-new", is_new=True)
    [call] = determine_harness.gateway_calls
    assert call["ingress_mode"] is GatewayIngressMode.PASSIVE_MEMORY
    assert call["identity_scope"] == _SYSTEM_SCOPE
    [prepared] = determine_harness.prepared
    assert (prepared["target_topic_id"], prepared["new_topic_title"]) == ("NEW_TOPIC", "部署")


@pytest.mark.asyncio
async def test_applied_turn_returns_its_actual_topic_without_routing(determine_harness):
    """这一轮已经应用：返回它实际所在的话题，不是新建；不路由、不准备话题。"""
    await determine_harness.submit("topic-applied")

    determined = await determine_harness.determine(_ROUTING)

    assert determined == DeterminedTopic(topic_id="topic-applied", is_new=False)
    assert determine_harness.gateway_calls == []
    assert determine_harness.prepared == []


@pytest.mark.asyncio
async def test_turn_still_applying_has_no_topic_yet(determine_harness):
    """这一轮已经提交、仍在应用：所在的话题尚未确定，不返回话题，也不路由或准备话题。"""
    determine_harness.apply_gate.clear()
    pending = await determine_harness.submit("topic-applied", timeout_seconds=0.05)

    determined = await determine_harness.determine(_ROUTING)

    assert pending.state == "pending"
    assert determined is None
    assert determine_harness.gateway_calls == []
    assert determine_harness.prepared == []
