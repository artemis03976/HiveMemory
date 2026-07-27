"""Passive ingress 可恢复失败降级契约测试。

覆盖 v0.6.0 设计 §6 与 P5 条目：
    - Gateway / retrieval 可恢复失败仍接收并累计 turn
    - 降级响应无 memory context，但原始交互照常在 turn 完成后提交
    - retrieval 失败保留已获得的 decision（topic 路由不丢）
    - 契约违约与装配缺陷不降级，向上抛出
    - 降级只经 RuntimeEventSink 观测
"""

from __future__ import annotations

import asyncio

import pytest

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayCommandOutcome,
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
    PassiveIngressContractError,
    PassiveIngressEvent,
    PassiveMessageIngressor,
    is_recoverable_ingress_error,
)

SOURCE = "unit_degrade"
CONVERSATION = "conv-degrade"
IDENTITY = Identity(user_id="u1", agent_id="a1")


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
        identity=IDENTITY,
    )


class _Recorder:
    """可注入 Gateway / retrieval 失败的探针。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.submitted: list = []
        self.gateway_error: BaseException | None = None
        self.retrieval_error: BaseException | None = None
        self.gateway_outcome = None

    async def gateway(self, **kwargs):
        self.calls.append("gateway")
        if self.gateway_error is not None:
            raise self.gateway_error
        return self.gateway_outcome or _decision()

    async def retrieve(self, **kwargs):
        self.calls.append("retrieve")
        if self.retrieval_error is not None:
            raise self.retrieval_error
        return RetrievalResponse()

    async def submit(self, sealed):
        self.calls.append("submit")
        self.submitted.append(sealed)
        return "topic-settled"


def _build(
    recorder: _Recorder,
    *,
    register_gateway: bool = True,
) -> tuple[PassiveMessageIngressor, RecordingRuntimeEventSink]:
    bus = GlobalSystemBus()
    if register_gateway:
        bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    sink = RecordingRuntimeEventSink()
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
        runtime_events=sink,
    )
    return ingressor, sink


def _context_event(sink: RecordingRuntimeEventSink):
    for event in sink.events:
        if event.event_type == RuntimeEventType.PASSIVE_MEMORY_CONTEXT_PREPARED.value:
            return event
    raise AssertionError("未发布 memory context 事件")


# ========== Gateway 可恢复失败 ==========


@pytest.mark.asyncio
async def test_gateway_failure_still_buffers_user_event() -> None:
    """§6：Gateway 可恢复失败不得阻止当前 user 进入 buffer。"""
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout")
    ingressor, _ = _build(recorder)

    outcome = await ingressor.route_event(_event("user", "u1"), IDENTITY)

    assert outcome.kind == "user"
    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer is not None
    assert buffer.has_pending_round
    assert buffer.event_count == 1


@pytest.mark.asyncio
async def test_gateway_failure_returns_no_memory_context() -> None:
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout")
    ingressor, _ = _build(recorder)

    outcome = await ingressor.route_event(_event("user", "u1"), IDENTITY)

    assert outcome.gateway_decision is None
    assert outcome.retrieval_result is None
    # Gateway 失败后不应继续请求 retrieval
    assert "retrieve" not in recorder.calls


@pytest.mark.asyncio
async def test_degraded_turn_still_submits_raw_interaction() -> None:
    """§6：降级后仍继续在 turn 完成后提交原始交互。"""
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout")
    ingressor, _ = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    submitted = await ingressor.flush_conversation(_key(), IDENTITY)

    assert submitted == 1
    sealed = recorder.submitted[0]
    # 原始交互完整保留
    assert sealed.payload.user_message == "u1"
    assert sealed.payload.assistant_final_text == "a1"
    assert [e.kind for e in sealed.payload.turn_events] == [
        "user_message",
        "assistant_message",
    ]
    # 缺少 decision 派生值，但 raw artifact 未被丢弃
    assert sealed.payload.rewritten_query is None
    assert sealed.payload.worth_saving is None
    assert sealed.target_topic is None


@pytest.mark.asyncio
async def test_degraded_turn_does_not_block_next_turn() -> None:
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout")
    ingressor, _ = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)

    # Gateway 恢复后下一轮正常工作，且上一降级轮被提交
    recorder.gateway_error = None
    outcome = await ingressor.route_event(_event("user", "u2"), IDENTITY)

    assert outcome.gateway_decision is not None
    assert len(recorder.submitted) == 1
    assert recorder.submitted[0].payload.user_message == "u1"


# ========== retrieval 可恢复失败 ==========


@pytest.mark.asyncio
async def test_retrieval_failure_preserves_decision() -> None:
    """retrieval 失败保留 decision，topic 路由与写入预判不丢。"""
    recorder = _Recorder()
    recorder.retrieval_error = ConnectionError("qdrant unreachable")
    ingressor, _ = _build(recorder)

    outcome = await ingressor.route_event(_event("user", "u1"), IDENTITY)

    assert outcome.gateway_decision is not None
    assert outcome.gateway_decision.target_topic_id == "topic-1"
    assert outcome.retrieval_result is None

    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    await ingressor.flush_conversation(_key(), IDENTITY)

    sealed = recorder.submitted[0]
    assert sealed.target_topic == "topic-1"
    assert sealed.payload.rewritten_query == "q"
    assert sealed.payload.worth_saving is True


# ========== 不可恢复失败不降级 ==========


@pytest.mark.asyncio
async def test_command_outcome_raises_contract_error() -> None:
    """PASSIVE_MEMORY 返回 command outcome 是契约违约，不降级。"""
    recorder = _Recorder()
    recorder.gateway_outcome = GatewayCommandOutcome(
        command_execution_result=CommandExecutionResult(
            command_id="cmd-1",
            status=CommandExecutionStatus.COMPLETED,
            message="ok",
        )
    )
    ingressor, _ = _build(recorder)

    with pytest.raises(PassiveIngressContractError):
        await ingressor.route_event(_event("user", "u1"), IDENTITY)


@pytest.mark.asyncio
async def test_unregistered_route_raises_instead_of_degrading() -> None:
    """总线路由未注册属于装配缺陷，不得静默降级。"""
    recorder = _Recorder()
    ingressor, _ = _build(recorder, register_gateway=False)

    with pytest.raises(KeyError):
        await ingressor.route_event(_event("user", "u1"), IDENTITY)


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    """取消信号不属于可恢复失败，必须向上传播。"""
    recorder = _Recorder()
    recorder.gateway_error = asyncio.CancelledError()
    ingressor, _ = _build(recorder)

    with pytest.raises(asyncio.CancelledError):
        await ingressor.route_event(_event("user", "u1"), IDENTITY)


def test_recoverable_classification() -> None:
    assert is_recoverable_ingress_error(TimeoutError("t"))
    assert is_recoverable_ingress_error(ConnectionError("c"))
    assert is_recoverable_ingress_error(RuntimeError("downstream blew up"))
    assert not is_recoverable_ingress_error(PassiveIngressContractError("x"))
    assert not is_recoverable_ingress_error(KeyError("route"))
    assert not is_recoverable_ingress_error(asyncio.CancelledError())


# ========== 降级只经 sink 观测 ==========


@pytest.mark.asyncio
async def test_gateway_degradation_publishes_warning_event() -> None:
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout")
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("user", "u1", turn_id="t-1"), IDENTITY)

    context = _context_event(sink)
    assert context.data["degraded"] is True
    assert context.data["failed_stage"] == "gateway"
    assert context.data["error_class"] == "TimeoutError"
    assert context.data["memory_ref_count"] == 0
    assert context.severity == "warning"
    assert context.status == "degraded"
    assert context.topic_id is None


@pytest.mark.asyncio
async def test_retrieval_degradation_publishes_stage_and_topic() -> None:
    recorder = _Recorder()
    recorder.retrieval_error = ConnectionError("qdrant unreachable")
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)

    context = _context_event(sink)
    assert context.data["failed_stage"] == "retrieval"
    assert context.data["error_class"] == "ConnectionError"
    # decision 已获得，topic 关联仍可观测
    assert context.topic_id == "topic-1"


@pytest.mark.asyncio
async def test_degradation_detail_not_leaked_into_outcome() -> None:
    """§6：降级细节只进 sink，不混入业务结果对象。"""
    recorder = _Recorder()
    recorder.gateway_error = TimeoutError("gateway timeout: secret-detail")
    ingressor, sink = _build(recorder)

    outcome = await ingressor.route_event(_event("user", "u1"), IDENTITY)

    assert not hasattr(outcome, "degraded")
    assert not hasattr(outcome, "error_class")
    assert not hasattr(outcome, "failed_stage")

    serialized = "\n".join(event.model_dump_json() for event in sink.events)
    assert "secret-detail" not in serialized


@pytest.mark.asyncio
async def test_successful_path_reports_not_degraded() -> None:
    recorder = _Recorder()
    ingressor, sink = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)

    context = _context_event(sink)
    assert context.data["degraded"] is False
    assert context.data["failed_stage"] is None
    assert context.data["error_class"] is None
    assert context.severity == "info"
