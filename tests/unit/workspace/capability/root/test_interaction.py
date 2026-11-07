"""root 交互提交：root 授权提交并等待应用；提交前按交互标识查重，待路由的一轮以 PASSIVE_MEMORY 路由。

真实协作：root 交互 API、操作授权者、Patchouli 的公开提交服务与内存提交队列；
Gateway 在本单元边界之外，以记录调用的确定性替身给出决定或失败。非 system 的
拒绝与其他 root 方法一起在 root API 授权矩阵中验收。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.access import WorkingActor
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import ActorIdentity, IdentityScope
from hivememory.core.protocol.gateway import (
    CommandParseResult,
    CommandParseStatus,
    GatewayCommandOutcome,
    GatewayDecisionOutcome,
    GatewayIngressMode,
    GatewaySessionTurn,
    GatewayTimeoutError,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.passive_routing import PassiveTurnRouting
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.sessions import make_interaction_payload
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

PARTICIPANT = ActorIdentity(user_id="u1", agent_id="claude_code")
WORKSPACE = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")
GATEWAY_TIMEOUT_MS = 1234
ROUTING = PassiveTurnRouting(message="测试挂了")


def _new_topic_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=make_gateway_decision(
            target_topic_id="NEW_TOPIC", new_topic_title="部署", new_topic_summary="讨论部署"
        )
    )


@dataclass
class _Harness:
    """被测 root 交互 API 与它所在的 Patchouli 提交队列，以及 Gateway 替身的输入与回答。

    应用函数把交互应用到请求的话题（``NEW_TOPIC`` 建立为 ``topic-new``）；
    ``apply_failures`` 中的异常依次让应用失败，``apply_gate`` 不为空时应用
    等它放行。``waiting`` 在 root API 开始等待已接纳的交互时置位。
    """

    service: RootInteractionService
    queue: InteractionSubmissionQueue
    actor: WorkingActor
    gateway_outcome: object = field(default_factory=_new_topic_outcome)
    gateway_calls: list[dict] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)
    apply_failures: list[Exception] = field(default_factory=list)
    apply_gate: asyncio.Event | None = None
    waiting: asyncio.Event = field(default_factory=asyncio.Event)

    async def submit(self, topic, *, interaction_id="import_1", timeout_seconds=2.0):
        return await self.service.submit_interaction(
            target_workspace=WORKSPACE,
            participant=PARTICIPANT,
            payload=make_interaction_payload("导入的一轮", "回复"),
            interaction_id=interaction_id,
            origin="import_bus",
            topic=topic,
            timeout_seconds=timeout_seconds,
            actor=self.actor,
        )


@pytest_asyncio.fixture
async def harness() -> AsyncIterator[_Harness]:
    """system 记录只提供准入，不配置 actor 操作；提交队列在测试期间运行。"""
    composition = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)],
        default_workspace=WORKSPACE,
    )
    bus = GlobalSystemBus()
    ref: list[_Harness] = []

    async def apply(payload, **kwargs):
        ref[0].applied.append(kwargs)
        if ref[0].apply_gate is not None:
            await ref[0].apply_gate.wait()
        if ref[0].apply_failures:
            raise ref[0].apply_failures.pop(0)
        target = kwargs["target_topic_id"]
        return "topic-new" if target == "NEW_TOPIC" else target

    queue = InteractionSubmissionQueue(apply)
    submissions = InteractionSubmissionService(interaction_queue=queue)
    bus.register(PatchouliRoutes.INTERACTION_SUBMIT, submissions.submit_interaction)
    bus.register(PatchouliRoutes.INTERACTION_STATUS, submissions.interaction_status)

    async def wait(**kwargs):
        ref[0].waiting.set()
        return await submissions.wait_interaction(**kwargs)

    bus.register(PatchouliRoutes.INTERACTION_WAIT, wait)

    async def gateway(message, **kwargs):
        ref[0].gateway_calls.append({"message": message, **kwargs})
        outcome = ref[0].gateway_outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    ref.append(
        _Harness(
            service=RootInteractionService(
                bus,
                operation_authorizer=composition.authorizer,
                gateway_request_timeout_ms=GATEWAY_TIMEOUT_MS,
            ),
            queue=queue,
            actor=await composition.authenticate_system(),
        )
    )
    await queue.start()
    try:
        yield ref[0]
    finally:
        if ref[0].apply_gate is not None:
            ref[0].apply_gate.set()
        await queue.stop()


@pytest.mark.asyncio
async def test_system_submits_with_participant_as_initiator(harness):
    """system 以 root 授权提交并等到应用：归属是目标 Workspace，发起者是参与的 Agent，不是 system。"""
    result = await harness.submit("topic-1")

    assert (result.state, result.topic_id, result.previously_accepted) == (
        "succeeded",
        "topic-1",
        False,
    )
    [submission] = harness.applied
    assert submission["belong_to"] == WORKSPACE
    assert submission["from_actor"] == PARTICIPANT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        ("topic-1", ("topic-1", None, None)),
        (
            make_gateway_decision(
                target_topic_id="NEW_TOPIC", new_topic_title="部署", new_topic_summary="讨论部署"
            ),
            ("NEW_TOPIC", "部署", "讨论部署"),
        ),
        (
            make_gateway_decision(
                target_topic_id="topic-1", new_topic_title="不应使用", new_topic_summary="不应使用"
            ),
            ("topic-1", None, None),
        ),
    ],
    ids=["determined-topic", "new-topic-decision", "existing-topic-decision"],
)
async def test_topic_source_decides_requested_topic_and_new_topic_fields(harness, topic, expected):
    """已确定的话题与 Gateway 的决定不再路由；只有新话题携带标题与摘要，已有话题的元数据不被改写。"""
    await harness.submit(topic)

    [submission] = harness.applied
    assert (
        submission["target_topic_id"],
        submission["new_topic_title"],
        submission["new_topic_summary"],
    ) == expected
    assert harness.gateway_calls == []


@pytest.mark.asyncio
async def test_passive_turn_is_routed_in_passive_mode_under_system_scope(harness):
    """待路由的一轮以 PASSIVE_MEMORY 模式、system 的 scope 与配置的时限路由，按决定提交。"""
    recent = (GatewaySessionTurn(user_message="上一轮", assistant_text="好的"),)

    await harness.submit(PassiveTurnRouting(message="测试挂了", session_turns=recent))

    [call] = harness.gateway_calls
    assert call["message"] == "测试挂了"
    assert call["ingress_mode"] is GatewayIngressMode.PASSIVE_MEMORY
    assert call["request_timeout_ms"] == GATEWAY_TIMEOUT_MS
    assert call["identity_scope"] == IdentityScope(
        actor_identity=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
        workspace_identity=WORKSPACE,
    )
    assert call["session_turns"] == recent
    [submission] = harness.applied
    assert (submission["target_topic_id"], submission["new_topic_title"]) == ("NEW_TOPIC", "部署")


@pytest.mark.asyncio
async def test_applied_turn_is_previously_accepted_without_routing_or_resubmission(harness):
    """已应用的一轮再次提交时返回此前已接纳的结局与实际所在的话题，不调用 Gateway、不再应用。"""
    first = await harness.submit(ROUTING)
    again = await harness.submit(ROUTING)

    assert (again.state, again.topic_id, again.previously_accepted) == (
        "succeeded",
        first.topic_id,
        True,
    )
    assert len(harness.gateway_calls) == 1
    assert len(harness.applied) == 1


@pytest.mark.asyncio
async def test_turn_still_applying_is_waited_for_without_routing_or_resubmission(harness):
    """仍在应用的一轮再次提交时等待它应用完成，交回等待的结局，不调用 Gateway、不再应用。"""
    harness.apply_gate = asyncio.Event()
    first = await harness.submit(ROUTING, timeout_seconds=0.05)
    resend = asyncio.create_task(harness.submit(ROUTING, timeout_seconds=None))
    await asyncio.wait_for(harness.waiting.wait(), timeout=2)
    harness.apply_gate.set()
    again = await asyncio.wait_for(resend, timeout=2)

    assert first.state == "pending"
    assert (again.state, again.topic_id, again.previously_accepted) == (
        "succeeded",
        "topic-new",
        True,
    )
    assert len(harness.gateway_calls) == 1
    assert len(harness.applied) == 1


@pytest.mark.asyncio
async def test_failed_turn_is_rerouted_and_resubmitted(harness):
    """应用失败的一轮再次提交时重新路由，可以换到另一个话题，以同一标识再次提交。"""
    harness.apply_failures.append(RuntimeError("topic evicted"))
    failed = await harness.submit(ROUTING)
    harness.gateway_outcome = GatewayDecisionOutcome(
        decision=make_gateway_decision(target_topic_id="topic-2")
    )
    retried = await harness.submit(ROUTING)

    assert failed.state == "failed"
    assert (retried.state, retried.topic_id, retried.previously_accepted) == (
        "succeeded",
        "topic-2",
        False,
    )
    assert len(harness.gateway_calls) == 2
    assert [submission["target_topic_id"] for submission in harness.applied] == [
        "NEW_TOPIC",
        "topic-2",
    ]


@pytest.mark.asyncio
async def test_previous_interaction_topic_continues_last_session_turn(harness):
    """给出上一轮的交互标识时，Gateway 收到的最后一轮带上它应用后实际所在的话题，供路由失败时续接。"""
    await harness.submit("topic-prev", interaction_id="import_0")
    await harness.submit(
        PassiveTurnRouting(
            message="这一轮",
            session_turns=(
                GatewaySessionTurn(user_message="更早", assistant_text=""),
                GatewaySessionTurn(user_message="上一轮", assistant_text=""),
            ),
            previous_interaction_id="import_0",
        )
    )

    [call] = harness.gateway_calls
    assert [turn.topic_id for turn in call["session_turns"]] == [None, "topic-prev"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            GatewayCommandOutcome(
                command_parse_result=CommandParseResult(
                    raw_input="/help", parse_status=CommandParseStatus.MATCHED
                )
            ),
            RuntimeError,
        ),
        (GatewayTimeoutError("deadline"), GatewayTimeoutError),
    ],
    ids=["command-result", "gateway-error"],
)
async def test_routing_failure_submits_nothing(harness, outcome, expected):
    """PASSIVE_MEMORY 返回命令结果是契约错误，Gateway 的异常原样传出；两者都不提交交互。"""
    harness.gateway_outcome = outcome

    with pytest.raises(expected):
        await harness.submit(ROUTING)

    assert await harness.queue.lookup("import_1") is None
