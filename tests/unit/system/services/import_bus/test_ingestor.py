"""Import Bus 编排器的单元测试。

被测对象：``ImportBusIngestor`` 在串行门内的进度查询、跳过、Gateway 落位、
封口、提交与等待应用，写入时为在途轮次落位与回流沿用落点，以及关闭时等待
在途导入。Patchouli 的提交、进度查询与等待使用真实的
``InteractionSubmissionService`` 与内存队列（只在进程内）；Gateway 与
Patchouli 的 prepare 是边界之外的协作者，以确定性的替身给出决定、话题或失败。

保护的契约：
- 只有应用完成后才确认；应用失败与等待超时不确认，重发时失败的重新路由并
  再次提交、仍在应用的继续等待；
- 已应用的轮次返回 duplicate、不再调用 Gateway；没有输出的轮次返回 skipped；
- Gateway 整体的可恢复异常与队列拥塞是可重试错误，契约错误、程序错误与取消传播；
- 最近轮次按上限截断后交给 Gateway；新话题携带 Gateway 的标题与摘要；
- 同一外部会话串行、不同会话并发；
- 事件不携带正文、工具参数与工具结果；
- 关闭后拒绝新导入与门外等待的导入，并等待在途导入结束；
- 在途轮次第一次落位只路由一次并准备话题，再次落位返回同一落点；已落位的
  轮次回流时不再路由、以落点话题提交；落点在提交被接纳前保留，落点话题已
  不存在时应用失败，重发时重新路由；尚未回流的上一轮以落点续接；落位时
  Gateway 的可恢复异常交给调用方。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.work_queue import QueuePolicy
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.protocol.gateway import (
    CommandParseResult,
    CommandParseStatus,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    GatewaySessionTurn,
    GatewayTimeoutError,
    IntentType,
    RetrievalPlan,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.application import InteractionSubmissionService
from hivememory.patchouli.contracts import PatchouliRoutes, PreparedAgentRun
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
from hivememory.system.services.import_bus import (
    ImportBusApplyFailedError,
    ImportBusContractError,
    ImportBusIngestor,
    ImportBusUnavailableError,
    ImportedTurn,
    InFlightTurn,
)
from tests.helpers.workspace import make_identity_scope

SCOPE = make_identity_scope(user_id="default", agent_id="claude_code")
PRINCIPAL = "hivememory:http-server"
SECRET = "sk-secret-token"


def _turn(**overrides) -> ImportedTurn:
    fields = {
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
        "outcome": "completed",
        "user_message": f"部署时用 {SECRET}",
        "events": [
            {"kind": "thought", "sequence": 1, "content": f"想想 {SECRET}"},
            {
                "kind": "tool_call",
                "sequence": 2,
                "content": "",
                "action_id": "toolu-1",
                "tool_name": "Bash",
                "tool_args": {"command": f"deploy --token {SECRET}"},
                "status": "success",
                "result": f"deployed with {SECRET}",
            },
            {"kind": "assistant_message", "sequence": 3, "content": "部署完成"},
        ],
        "model_used": "claude-opus",
    }
    fields.update(overrides)
    return ImportedTurn.model_validate(fields)


def _in_flight(**overrides) -> InFlightTurn:
    """``_turn`` 那一轮仍在进行时的形态：外部标识与用户消息相同，没有事件与结局。"""
    turn = _turn(**overrides)
    return InFlightTurn.model_validate(turn.model_dump(include=set(InFlightTurn.model_fields)))


def _decision(**overrides) -> GatewayDecision:
    fields = {
        "target_topic_id": "NEW_TOPIC",
        "new_topic_title": "部署",
        "new_topic_summary": "讨论部署步骤",
        "rewritten_query": "如何部署",
        "retrieval_plan": RetrievalPlan(),
        "intent_type": IntentType.RAG,
    }
    fields.update(overrides)
    return GatewayDecision(**fields)


@dataclass
class _Harness:
    ingestor: ImportBusIngestor
    queue: InteractionSubmissionQueue
    events: RecordingRuntimeEventSink
    gateway_calls: list[dict] = field(default_factory=list)
    prepare_calls: list[dict] = field(default_factory=list)
    applied: list[tuple[InteractionPayload, dict]] = field(default_factory=list)

    async def ingest(self, turn: ImportedTurn):
        return await self.ingestor.ingest(turn, identity_scope=SCOPE, principal_id=PRINCIPAL)

    async def place(self, turn: InFlightTurn) -> str:
        return await self.ingestor.place(turn, identity_scope=SCOPE, principal_id=PRINCIPAL)


@asynccontextmanager
async def _harness(
    gateway=None,
    *,
    session_turn_limit: int = 3,
    start_queue: bool = True,
    queue_policy: QueuePolicy | None = None,
    apply=None,
    apply_wait_seconds: float = 2.0,
) -> AsyncIterator[_Harness]:
    """装配编排器、真实提交服务与内存队列；``gateway`` 缺省返回新话题决定。

    prepare 的替身把新话题预建为 ``topic-placed``，已有话题原样返回。
    """
    bus = GlobalSystemBus()
    events = RecordingRuntimeEventSink()
    harness_ref: list[_Harness] = []

    async def default_apply(payload, **kwargs):
        return "topic-applied"

    apply_handler = apply or default_apply

    async def recording_apply(payload, **kwargs):
        harness_ref[0].applied.append((payload, kwargs))
        return await apply_handler(payload, **kwargs)

    queue = InteractionSubmissionQueue(recording_apply, policy=queue_policy)
    submissions = InteractionSubmissionService(interaction_queue=queue)
    bus.register(PatchouliRoutes.INTERACTION_SUBMIT, submissions.submit_interaction)
    bus.register(PatchouliRoutes.INTERACTION_STATUS, submissions.interaction_status)
    bus.register(PatchouliRoutes.INTERACTION_WAIT, submissions.wait_interaction)

    async def default_gateway(**kwargs):
        return GatewayDecisionOutcome(decision=_decision())

    gateway_handler = gateway or default_gateway

    async def recording_gateway(message, **kwargs):
        harness_ref[0].gateway_calls.append({"message": message, **kwargs})
        return await gateway_handler(message=message, **kwargs)

    bus.register(GatewayPublicRoutes.PROCESS, recording_gateway)

    async def prepare(**kwargs):
        harness_ref[0].prepare_calls.append(kwargs)
        decision = kwargs["gateway_decision"]
        is_new_topic = decision.target_topic_id == "NEW_TOPIC"
        return PreparedAgentRun(
            belong_to=kwargs["identity_scope"].workspace_identity,
            interaction_id=kwargs["interaction_id"],
            topic_id="topic-placed" if is_new_topic else decision.target_topic_id,
            is_new_topic=is_new_topic,
        )

    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    harness = _Harness(
        ingestor=ImportBusIngestor(
            bus,
            gateway_request_timeout_ms=1234,
            gateway_session_turn_limit=session_turn_limit,
            apply_wait_seconds=apply_wait_seconds,
            runtime_events=events,
        ),
        queue=queue,
        events=events,
    )
    harness_ref.append(harness)
    if start_queue:
        await queue.start()
    try:
        yield harness
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_accepted_turn_is_sealed_and_submitted_with_new_topic_title() -> None:
    """接纳的一轮以共用封口得到载荷，新话题带 Gateway 的标题与摘要提交。"""
    async with _harness() as harness:
        result = await harness.ingest(_turn())
        await harness.queue.drain_all(timeout=2)

    assert result.status == "accepted"
    assert result.interaction_id.startswith("import_")
    [(payload, kwargs)] = harness.applied
    assert kwargs["interaction_id"] == result.interaction_id
    assert kwargs["belong_to"] == SCOPE.workspace_identity
    assert kwargs["from_actor"] == SCOPE.actor_identity
    assert kwargs["target_topic_id"] == "NEW_TOPIC"
    assert (kwargs["new_topic_title"], kwargs["new_topic_summary"]) == ("部署", "讨论部署步骤")
    assert payload.outcome == "completed"
    assert payload.model_used == "claude-opus"
    assert payload.used_attachments == []
    # 回复正文由事件中的助手正文得到，不随请求上传；用户消息是服务端写出的序号 0 的事件。
    assert (payload.user_message, payload.reply_text) == (f"部署时用 {SECRET}", "部署完成")
    assert [(event.kind, event.sequence) for event in payload.turn_events] == [
        ("user_message", 0),
        ("thought", 1),
        ("tool_call", 2),
        ("assistant_message", 3),
    ]


@pytest.mark.asyncio
async def test_turn_is_confirmed_only_after_it_is_applied() -> None:
    """提交被接纳后仍等待感知应用；应用完成之前不返回确认。"""
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_apply(payload, **kwargs):
        started.set()
        await release.wait()
        return "topic-applied"

    async with _harness(apply=slow_apply) as harness:
        task = asyncio.create_task(harness.ingest(_turn()))
        await started.wait()
        confirmed_before_apply = task.done()
        release.set()
        result = await task

    assert confirmed_before_apply is False
    assert result.status == "accepted"


@pytest.mark.asyncio
async def test_apply_failure_is_not_confirmed_and_resend_reroutes_and_resubmits() -> None:
    """应用失败不确认；重发时重新路由并以同一标识再次提交，应用成功后确认。"""
    failures = [RuntimeError("topic evicted")]

    async def flaky_apply(payload, **kwargs):
        if failures:
            raise failures.pop()
        return "topic-applied"

    async with _harness(apply=flaky_apply) as harness:
        with pytest.raises(ImportBusApplyFailedError):
            await harness.ingest(_turn())
        resent = await harness.ingest(_turn())

    assert resent.status == "accepted"
    assert len(harness.gateway_calls) == 2
    assert [kwargs["interaction_id"] for _, kwargs in harness.applied] == [
        resent.interaction_id,
        resent.interaction_id,
    ]
    deferred = harness.events.events[0]
    assert (deferred.event_type, deferred.reason) == (
        RuntimeEventType.IMPORT_TURN_DEFERRED.value,
        "apply_failed",
    )


@pytest.mark.asyncio
async def test_resend_while_still_applying_waits_without_rerouting() -> None:
    """等待应用超时以 apply_pending 交回；重发时只继续等待，不再路由与提交。"""
    release = asyncio.Event()

    async def slow_apply(payload, **kwargs):
        await release.wait()
        return "topic-applied"

    async with _harness(apply=slow_apply, apply_wait_seconds=0.05) as harness:
        with pytest.raises(ImportBusUnavailableError) as pending:
            await harness.ingest(_turn())
        release.set()
        resent = await harness.ingest(_turn())

    assert pending.value.details == {"reason": "apply_pending"}
    assert resent.status == "duplicate"
    assert len(harness.gateway_calls) == 1
    assert len(harness.applied) == 1


@pytest.mark.asyncio
async def test_existing_topic_submission_does_not_carry_new_topic_fields() -> None:
    """落到已有话题时不携带新话题的标题与摘要，已有话题的元数据不被改写。"""

    async def gateway(**kwargs):
        return GatewayDecisionOutcome(
            decision=_decision(target_topic_id="topic-1", new_topic_title="不应使用")
        )

    async with _harness(gateway) as harness:
        await harness.ingest(_turn())
        await harness.queue.drain_all(timeout=2)

    [(_, kwargs)] = harness.applied
    assert kwargs["target_topic_id"] == "topic-1"
    assert (kwargs["new_topic_title"], kwargs["new_topic_summary"]) == (None, None)


@pytest.mark.asyncio
async def test_resent_accepted_turn_is_duplicate_without_gateway_call() -> None:
    """已接纳的轮次重发时返回 duplicate，不再路由，交互只应用一次。"""
    async with _harness() as harness:
        first = await harness.ingest(_turn())
        resent = await harness.ingest(_turn())
        await harness.queue.drain_all(timeout=2)

    assert (first.status, resent.status) == ("accepted", "duplicate")
    assert resent.interaction_id == first.interaction_id
    assert len(harness.gateway_calls) == 1
    assert len(harness.applied) == 1


@pytest.mark.asyncio
async def test_turn_without_output_is_skipped_without_gateway_call() -> None:
    """用户消息之外没有输出的轮次不封口、不路由。"""
    async with _harness() as harness:
        result = await harness.ingest(_turn(events=[]))

    assert result.status == "skipped"
    assert harness.gateway_calls == []
    assert harness.applied == []
    assert [event.event_type for event in harness.events.events] == [
        RuntimeEventType.IMPORT_TURN_SKIPPED.value
    ]


@pytest.mark.asyncio
async def test_gateway_receives_passive_mode_and_truncated_recent_turns() -> None:
    """Gateway 以 PASSIVE_MEMORY 模式收到用户消息与按上限截断的最近轮次。"""
    recent = [
        {"user_message": f"问题 {index}", "assistant_text": f"回答 {index}"} for index in range(3)
    ]

    async with _harness(session_turn_limit=2) as harness:
        await harness.ingest(_turn(recent_turns=recent))

    [call] = harness.gateway_calls
    assert call["message"] == f"部署时用 {SECRET}"
    assert call["ingress_mode"].value == "passive_memory"
    assert call["request_timeout_ms"] == 1234
    assert call["identity_scope"] == SCOPE
    assert call["session_turns"] == (
        GatewaySessionTurn(user_message="问题 1", assistant_text="回答 1"),
        GatewaySessionTurn(user_message="问题 2", assistant_text="回答 2"),
    )


@pytest.mark.asyncio
async def test_last_recent_turn_carries_its_applied_topic_for_continuation() -> None:
    """最近一轮带外部标识时，Gateway 收到它实际落位的话题，供路由失败时续接。"""
    async with _harness() as harness:
        first = await harness.ingest(_turn(external_turn_id="prompt-1"))
        await harness.queue.wait(first.interaction_id, timeout=2)
        await harness.ingest(
            _turn(
                external_turn_id="prompt-2",
                recent_turns=[
                    {"user_message": "更早", "assistant_text": "", "external_turn_id": "prompt-0"},
                    {
                        "user_message": "上一轮",
                        "assistant_text": "",
                        "external_turn_id": "prompt-1",
                    },
                ],
            )
        )

    second_call = harness.gateway_calls[1]
    assert [turn.topic_id for turn in second_call["session_turns"]] == [None, "topic-applied"]


@pytest.mark.asyncio
async def test_last_recent_turn_carries_its_placement_before_it_is_imported() -> None:
    """上一轮已在写入时落位、回流尚未到达时，下一轮落位以它的落点续接。"""
    async with _harness() as harness:
        placed = await harness.place(_in_flight(external_turn_id="prompt-1"))
        await harness.place(
            _in_flight(
                external_turn_id="prompt-2",
                recent_turns=[
                    {"user_message": "上一轮", "assistant_text": "", "external_turn_id": "prompt-1"}
                ],
            )
        )

    second_call = harness.gateway_calls[1]
    assert [turn.topic_id for turn in second_call["session_turns"]] == [placed]


@pytest.mark.asyncio
async def test_recoverable_gateway_failure_is_retryable_and_resend_is_accepted() -> None:
    """Gateway 整体抛出可恢复异常（例如时限耗尽而入口步骤没有兜底）时不提交，
    交回可重试错误；重发后正常接纳。步骤内的失败由 Gateway 自身兜底。
    """
    failures = [GatewayTimeoutError("deadline")]

    async def gateway(**kwargs):
        if failures:
            raise failures.pop()
        return GatewayDecisionOutcome(decision=_decision())

    async with _harness(gateway) as harness:
        with pytest.raises(ImportBusUnavailableError) as error:
            await harness.ingest(_turn())
        retried = await harness.ingest(_turn())

    assert error.value.details == {"reason": "gateway_unavailable"}
    assert retried.status == "accepted"
    deferred = harness.events.events[0]
    assert deferred.event_type == RuntimeEventType.IMPORT_TURN_DEFERRED.value
    assert deferred.reason == "gateway_unavailable"
    assert deferred.data["error_class"] == "GatewayTimeoutError"


@pytest.mark.asyncio
async def test_queue_capacity_is_retryable_queue_unavailable() -> None:
    """提交队列已满时没有接管提交，以 queue_unavailable 交回可重试错误。

    队列不运行时第一轮被接纳但迟迟没有应用，等待超时以 apply_pending 交回；
    它占满容量后，第二轮在提交时被拒绝。
    """
    async with _harness(
        start_queue=False,
        queue_policy=QueuePolicy(capacity=1, max_concurrency=1),
        apply_wait_seconds=0.05,
    ) as harness:
        with pytest.raises(ImportBusUnavailableError) as pending:
            await harness.ingest(_turn(external_turn_id="prompt-1"))
        with pytest.raises(ImportBusUnavailableError) as error:
            await harness.ingest(_turn(external_turn_id="prompt-2"))

    assert pending.value.details == {"reason": "apply_pending"}
    assert error.value.details == {"reason": "queue_unavailable"}
    assert harness.events.events[-1].event_type == RuntimeEventType.IMPORT_TURN_DEFERRED.value
    assert harness.events.events[-1].reason == "queue_unavailable"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            GatewayCommandOutcome(
                command_parse_result=CommandParseResult(
                    raw_input="/help", parse_status=CommandParseStatus.MATCHED
                )
            ),
            ImportBusContractError,
        ),
        (RuntimeError("bug"), RuntimeError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
    ids=["command-result", "program-error", "cancelled"],
)
@pytest.mark.asyncio
async def test_contract_errors_program_errors_and_cancellation_propagate(outcome, expected):
    """命令结果是契约错误，程序错误与取消原样传播，都不转为可重试错误。"""

    async def gateway(**kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async with _harness(gateway) as harness:
        with pytest.raises(expected):
            await harness.ingest(_turn())

    assert harness.applied == []
    assert all(
        event.event_type != RuntimeEventType.IMPORT_TURN_DEFERRED.value
        for event in harness.events.events
    )


@pytest.mark.asyncio
async def test_same_conversation_is_serialized_while_other_conversations_proceed() -> None:
    """同一外部会话的并发重发等待前一请求，后到者看到已接纳；其他会话不被阻塞。"""
    release = asyncio.Event()
    entered: dict[str, asyncio.Event] = {"first": asyncio.Event(), "other": asyncio.Event()}

    async def gateway(*, message, **kwargs):
        entered[message].set()
        await release.wait()
        return GatewayDecisionOutcome(decision=_decision())

    async with _harness(gateway) as harness:
        first = asyncio.create_task(harness.ingest(_turn(user_message="first")))
        await entered["first"].wait()
        resend = asyncio.create_task(harness.ingest(_turn(user_message="first")))
        other = asyncio.create_task(
            harness.ingest(_turn(user_message="other", external_conversation_id="session-2"))
        )
        # 另一会话在第一个请求仍持门时到达 Gateway；先创建的重发此时已在门外等待。
        await entered["other"].wait()
        reached_gateway = [call["message"] for call in harness.gateway_calls]
        release.set()
        results = await asyncio.gather(first, resend, other)

    assert reached_gateway == ["first", "other"]
    assert [result.status for result in results] == ["accepted", "duplicate", "accepted"]


@pytest.mark.asyncio
async def test_runtime_events_carry_identifiers_without_content() -> None:
    """事件只带来源、外部标识、交互标识与计数，不带正文、工具参数或结果。"""
    async with _harness() as harness:
        result = await harness.ingest(_turn())

    [accepted] = harness.events.events
    assert accepted.event_type == RuntimeEventType.IMPORT_TURN_ACCEPTED.value
    assert accepted.agent_id == "claude_code"
    # 应用完成后才确认：新建话题的真实落点此时已经确定。
    assert accepted.topic_id == "topic-applied"
    assert accepted.data == {
        "source": "claude-code",
        "external_conversation_id": "session-1",
        "external_turn_id": "prompt-1",
        "interaction_id": result.interaction_id,
        "event_count": 3,
        "outcome": "completed",
        "is_new_topic": True,
    }
    assert SECRET not in accepted.model_dump_json()


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_import_and_rejects_new_ones() -> None:
    """关闭后新导入与门外等待的导入以 shutting_down 拒绝；在途导入完成后关闭才返回。"""
    release = asyncio.Event()
    entered = asyncio.Event()

    async def gateway(**kwargs):
        entered.set()
        await release.wait()
        return GatewayDecisionOutcome(decision=_decision())

    async with _harness(gateway) as harness:
        inflight = asyncio.create_task(harness.ingest(_turn()))
        await entered.wait()
        # 任务按创建顺序开始执行：同会话的后续轮次先在门外等待，关闭再开始，
        # 迟到的导入最后进入编排器。
        queued = asyncio.create_task(harness.ingest(_turn(external_turn_id="prompt-queued")))
        shutdown = asyncio.create_task(harness.ingestor.shutdown())
        late = asyncio.create_task(harness.ingest(_turn(external_turn_id="prompt-late")))
        with pytest.raises(ImportBusUnavailableError) as late_rejected:
            await late
        shutdown_done_before_release = shutdown.done()
        release.set()
        result = await inflight
        with pytest.raises(ImportBusUnavailableError) as queued_rejected:
            await queued
        report = await shutdown

    assert late_rejected.value.details == {"reason": "shutting_down"}
    assert queued_rejected.value.details == {"reason": "shutting_down"}
    assert shutdown_done_before_release is False
    assert result.status == "accepted"
    assert len(harness.gateway_calls) == 1
    assert report == {"waited_requests": 2}


@pytest.mark.asyncio
async def test_first_placement_routes_and_prepares_once_and_later_placements_reuse_it() -> None:
    """第一次落位以 PASSIVE_MEMORY 路由一次，关闭检索准备话题；同一轮再次落位返回同一落点。"""
    async with _harness() as harness:
        first = await harness.place(_in_flight())
        again = await harness.place(_in_flight())

    assert (first, again) == ("topic-placed", "topic-placed")
    [gateway_call] = harness.gateway_calls
    assert gateway_call["ingress_mode"].value == "passive_memory"
    [prepare] = harness.prepare_calls
    assert prepare["identity_scope"] == SCOPE
    assert prepare["gateway_decision"] == _decision()
    assert prepare["enable_memory_retrieval"] is False


@pytest.mark.asyncio
async def test_placed_turn_is_submitted_to_its_placement_without_rerouting() -> None:
    """已落位的轮次回流时不调用 Gateway，以落点话题提交，不再携带新话题的标题与摘要。"""
    async with _harness() as harness:
        topic_id = await harness.place(_in_flight())
        result = await harness.ingest(_turn())

    assert result.status == "accepted"
    assert len(harness.gateway_calls) == 1
    # 落位与导入对同一轮推导出同一个交互标识，回流因此找得到落点。
    assert harness.prepare_calls[0]["interaction_id"] == result.interaction_id
    [(_, kwargs)] = harness.applied
    assert kwargs["target_topic_id"] == topic_id
    assert (kwargs["new_topic_title"], kwargs["new_topic_summary"]) == (None, None)
    [accepted] = harness.events.events
    assert accepted.data["is_new_topic"] is True


@pytest.mark.asyncio
async def test_vanished_placement_fails_apply_and_resend_reroutes() -> None:
    """落点话题在本轮进行中结束生命周期时应用失败；重发时已没有落点，按现行规则重新路由。"""

    async def apply(payload, **kwargs):
        if kwargs["target_topic_id"] == "topic-placed":
            raise RuntimeError("topic evicted")
        return "topic-applied"

    async with _harness(apply=apply) as harness:
        await harness.place(_in_flight())
        with pytest.raises(ImportBusApplyFailedError):
            await harness.ingest(_turn())
        resent = await harness.ingest(_turn())

    assert resent.status == "accepted"
    assert len(harness.gateway_calls) == 2
    assert [kwargs["target_topic_id"] for _, kwargs in harness.applied] == [
        "topic-placed",
        "NEW_TOPIC",
    ]


@pytest.mark.asyncio
async def test_placement_is_kept_until_its_submission_is_accepted() -> None:
    """提交队列拥塞时提交没有被接纳，落点保留：重发仍以落点提交，不改由 Gateway 重新路由。

    队列不运行时第一轮被接纳但迟迟没有应用，占满容量；之后落位的一轮两次
    回流都在提交时被拒绝。
    """
    async with _harness(
        start_queue=False,
        queue_policy=QueuePolicy(capacity=1, max_concurrency=1),
        apply_wait_seconds=0.05,
    ) as harness:
        with pytest.raises(ImportBusUnavailableError):
            await harness.ingest(_turn(external_turn_id="prompt-0"))
        await harness.place(_in_flight())
        rejected = []
        for _ in range(2):
            with pytest.raises(ImportBusUnavailableError) as congested:
                await harness.ingest(_turn())
            rejected.append(congested.value.details)

    assert rejected == [{"reason": "queue_unavailable"}] * 2
    # 第一轮导入与落位各路由一次，两次回流都沿用落点。
    assert len(harness.gateway_calls) == 2


@pytest.mark.asyncio
async def test_recoverable_gateway_failure_during_placement_leaves_no_placement() -> None:
    """落位时 Gateway 的可恢复异常以 gateway_unavailable 交回，不登记落点，也不记为导入推迟。"""
    failures = [GatewayTimeoutError("deadline")]

    async def gateway(**kwargs):
        if failures:
            raise failures.pop()
        return GatewayDecisionOutcome(decision=_decision())

    async with _harness(gateway) as harness:
        with pytest.raises(ImportBusUnavailableError) as error:
            await harness.place(_in_flight())
        topic_id = await harness.place(_in_flight())

    assert error.value.details == {"reason": "gateway_unavailable"}
    assert topic_id == "topic-placed"
    assert len(harness.gateway_calls) == 2
    assert len(harness.prepare_calls) == 1
    assert harness.events.events == []
