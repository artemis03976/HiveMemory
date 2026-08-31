"""Passive ingress 可观测性契约测试。

覆盖 v0.6.0 设计 §9 与验收清单 #9：
    - Passive 事件接收、duplicate 与 memory context 按语义发布
    - submission 生命周期统一由通用 WORK_* 事件观测
    - 观测信息只进 RuntimeEventSink，不进 outcome / 公共响应
    - 事件不携带外部消息全文、tool args 或完整 memory context
"""

from __future__ import annotations

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope

SOURCE = "unit_events"
CONVERSATION = "conv-events"
IDENTITY = Identity(user_id="u1", agent_id="a1")

USER_SECRET = "私密的外部用户消息全文"
TOOL_ARG_SECRET = "s3://private-bucket/secret-key"
MEMORY_SECRET = "记忆正文不应出现在观测流"


def _decision(topic: str = "topic-1") -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id=topic,
            rewritten_query="q",
            search_keywords=("k",),
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(mode=RetrievalMode.HYBRID, top_k=5),
            intent_type=IntentType.RAG,
        )
    )


def _event(role: str, content: str, **kwargs) -> PassiveIngressEvent:
    return PassiveIngressEvent(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        role=role,
        content=content,
        **kwargs,
    )


def _key() -> PassiveConversationKey:
    return PassiveConversationKey.build(
        source=SOURCE,
        external_conversation_id=CONVERSATION,
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
    )


def _memory(content: str = MEMORY_SECRET) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(user_id="u1", source_agent_id="a1"),
        index=IndexLayer(
            title="observability fixture",
            summary="passive observability test fixture atom",
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=content),
    )


class _Recorder:
    def __init__(self) -> None:
        self.fail_submit = False
        self.memories: list[MemoryAtom] = []
        self.settled_topic_id: str | None = "topic-settled"

    async def gateway(self, **kwargs):
        return _decision()

    async def retrieve(self, **kwargs):
        return RetrievalResponse(memories=list(self.memories))

    async def submit(self, submission):
        if self.fail_submit:
            raise RuntimeError(f"patchouli down: {USER_SECRET}")
        return None


def _build(
    recorder: _Recorder,
) -> tuple[PassiveMessageIngressor, RecordingRuntimeEventSink]:
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    sink = RecordingRuntimeEventSink()
    ingressor = PassiveMessageIngressor(
        bus,
        interaction_queue=recorder,
        runtime_events=sink,
    )
    return ingressor, sink


def _types(sink: RecordingRuntimeEventSink) -> list[str]:
    return [event.event_type for event in sink.events]


def _first(sink: RecordingRuntimeEventSink, event_type: RuntimeEventType):
    for event in sink.events:
        if event.event_type == event_type.value:
            return event
    raise AssertionError(f"未发布事件: {event_type}")


# ========== 事件发布 ==========


@pytest.mark.asyncio
async def test_event_accepted_is_published_with_correlation() -> None:
    recorder = _Recorder()
    ingressor, sink = _build(recorder)

    await ingressor.route_event(
        _event("user", USER_SECRET, turn_id="t-1", sequence=7),
        IDENTITY,
    )

    accepted = _first(sink, RuntimeEventType.PASSIVE_INGRESS_EVENT_ACCEPTED)
    assert accepted.data["source"] == SOURCE
    assert accepted.data["external_conversation_id"] == CONVERSATION
    assert accepted.data["turn_id"] == "t-1"
    assert accepted.data["sequence"] == 7
    assert accepted.data["role"] == "user"
    assert accepted.status == "accepted"
    assert accepted.agent_id == "a1"
    assert accepted.workspace_id == "main_workspace"


@pytest.mark.asyncio
async def test_duplicate_ignored_is_published() -> None:
    recorder = _Recorder()
    ingressor, sink = _build(recorder)

    event = _event("user", USER_SECRET)
    await ingressor.route_event(event, IDENTITY)
    sink.events.clear()

    await ingressor.route_event(event, IDENTITY)

    duplicate = _first(sink, RuntimeEventType.PASSIVE_INGRESS_DUPLICATE_IGNORED)
    assert duplicate.data["external_event_id"] == event.external_event_id
    assert duplicate.data["source"] == SOURCE
    assert duplicate.status == "duplicate"
    # 重复事件不得重复发布 accepted
    assert RuntimeEventType.PASSIVE_INGRESS_EVENT_ACCEPTED.value not in _types(sink)


@pytest.mark.asyncio
async def test_memory_context_prepared_reports_ref_count_and_duration() -> None:
    recorder = _Recorder()
    recorder.memories = [_memory(), _memory()]
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("user", USER_SECRET, turn_id="t-1"), IDENTITY)

    prepared = _first(sink, RuntimeEventType.PASSIVE_MEMORY_CONTEXT_PREPARED)
    assert prepared.data["memory_ref_count"] == 2
    assert prepared.data["degraded"] is False
    assert "duration_ms" in prepared.data
    assert prepared.data["turn_id"] == "t-1"
    assert prepared.topic_id == "topic-1"
    assert prepared.status == "prepared"


@pytest.mark.asyncio
async def test_non_user_event_does_not_publish_memory_context() -> None:
    recorder = _Recorder()
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)

    assert RuntimeEventType.PASSIVE_INGRESS_EVENT_ACCEPTED.value in _types(sink)
    assert RuntimeEventType.PASSIVE_MEMORY_CONTEXT_PREPARED.value not in _types(sink)


# ========== 脱敏与边界 ==========


@pytest.mark.asyncio
async def test_events_never_carry_external_content_or_tool_args() -> None:
    """§9：passive event 不记录外部消息全文、tool args 或完整 memory context。"""
    recorder = _Recorder()
    recorder.memories = [_memory()]
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("user", USER_SECRET), IDENTITY)
    await ingressor.route_event(
        _event(
            "tool_call",
            "读取私密对象",
            tool_name="s3_get",
            tool_args={"uri": TOOL_ARG_SECRET},
        ),
        IDENTITY,
    )
    await ingressor.flush_conversation(_key())

    assert sink.events, "应至少发布若干观测事件"
    serialized = "\n".join(event.model_dump_json() for event in sink.events)
    assert USER_SECRET not in serialized
    assert TOOL_ARG_SECRET not in serialized
    assert MEMORY_SECRET not in serialized


@pytest.mark.asyncio
async def test_outcome_carries_no_observability_trace() -> None:
    """验收 #9：观测信息只进 sink，outcome 不累积 trace。"""
    recorder = _Recorder()
    ingressor, sink = _build(recorder)

    outcome = await ingressor.route_event(_event("user", USER_SECRET), IDENTITY)

    assert not hasattr(outcome, "runtime_events")
    assert not hasattr(outcome, "trace")
    assert not hasattr(outcome, "fallback")
    # 观测事件确实经由 sink 发布
    assert RuntimeEventType.PASSIVE_INGRESS_EVENT_ACCEPTED.value in _types(sink)


@pytest.mark.asyncio
async def test_ingressor_works_without_sink() -> None:
    """未接入 sink 时业务流程不受影响（NullRuntimeEventSink 兜底）。"""
    recorder = _Recorder()
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    outcome = await ingressor.route_event(_event("user", USER_SECRET), IDENTITY)
    assert outcome.kind == "user"

    submitted = await ingressor.flush_conversation(_key())
    assert submitted == 1
