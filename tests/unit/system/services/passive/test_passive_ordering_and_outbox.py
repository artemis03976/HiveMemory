"""Passive ingress 读写时序、outbox 与幂等契约测试。

覆盖 v0.6.0 设计的以下不变量：
    - user 事件先 seal 并提交上一 turn，再请求 Gateway decision 与 retrieval
    - Gateway 失败不得连带阻塞上一 turn 的提交
    - Patchouli submit 失败保留 sealed outbox item、不阻塞下一 turn
    - source + external_event_id 进程内幂等
    - source + external_conversation_id 会话隔离
    - explicit final / next user / idle timeout / manual flush / shutdown drain
      产生同构 InteractionPayload
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import Identity
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.services.passive import (
    PassiveConversationKey,
    PassiveIngressEvent,
    PassiveMessageIngressor,
)

SOURCE = "unit_test"
CONVERSATION = "conv-1"
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


def _event(
    role: str,
    content: str,
    *,
    conversation: str = CONVERSATION,
    **kwargs,
) -> PassiveIngressEvent:
    return PassiveIngressEvent(
        source=SOURCE,
        external_conversation_id=conversation,
        role=role,
        content=content,
        **kwargs,
    )


def _key(conversation: str = CONVERSATION) -> PassiveConversationKey:
    return PassiveConversationKey.build(
        source=SOURCE,
        external_conversation_id=conversation,
        identity=IDENTITY,
    )


class _Recorder:
    """记录调用顺序的探针。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.submitted: list = []
        self.fail_submit = False

    async def gateway(self, **kwargs):
        self.calls.append("gateway")
        return _decision()

    async def retrieve(self, **kwargs):
        self.calls.append("retrieve")
        return RetrievalResponse()

    async def submit(self, sealed):
        self.calls.append("submit")
        if self.fail_submit:
            raise RuntimeError("patchouli down")
        self.submitted.append(sealed)


def _build(recorder: _Recorder, **config_kwargs) -> PassiveMessageIngressor:
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    return PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
        config=PassiveIngressConfig(**config_kwargs) if config_kwargs else None,
    )


# ========== 读写时序 ==========


@pytest.mark.asyncio
async def test_same_conversation_waits_for_inflight_user_event() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def blocking_gateway(message, **kwargs):
        recorder.calls.append(f"gateway:{message}")
        gateway_started.set()
        await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    user_task = asyncio.create_task(ingressor.route_event(_event("user", "u1"), IDENTITY))
    await asyncio.wait_for(gateway_started.wait(), timeout=1)

    assistant_task = asyncio.create_task(ingressor.route_event(_event("assistant", "a1"), IDENTITY))
    await asyncio.sleep(0)

    assert not assistant_task.done()
    assert ingressor._serial_gate.active_key_count == 1

    release_gateway.set()
    user_outcome, assistant_outcome = await asyncio.wait_for(
        asyncio.gather(user_task, assistant_task),
        timeout=1,
    )

    assert user_outcome.kind == "user"
    assert assistant_outcome.kind == "buffered"
    assert ingressor._serial_gate.active_key_count == 0

    assert await ingressor.flush_conversation(_key(), IDENTITY) == 1
    payload = recorder.submitted[0].payload
    assert payload.user_message == "u1"
    assert payload.assistant_final_text == "a1"


@pytest.mark.asyncio
async def test_different_conversations_remain_concurrent() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def selectively_blocking_gateway(message, **kwargs):
        recorder.calls.append(f"gateway:{message}")
        if message == "u-a":
            gateway_started.set()
            await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, selectively_blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    await ingressor.route_event(
        _event("user", "u-b", conversation="c-b"),
        IDENTITY,
    )
    blocked_task = asyncio.create_task(
        ingressor.route_event(
            _event("user", "u-a", conversation="c-a"),
            IDENTITY,
        )
    )
    await asyncio.wait_for(gateway_started.wait(), timeout=1)

    outcome = await asyncio.wait_for(
        ingressor.route_event(
            _event("assistant", "a-b", conversation="c-b"),
            IDENTITY,
        ),
        timeout=1,
    )
    assert outcome.kind == "buffered"

    release_gateway.set()
    await asyncio.wait_for(blocked_task, timeout=1)
    assert ingressor._serial_gate.active_key_count == 0


@pytest.mark.asyncio
async def test_cancelled_event_releases_conversation_gate() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()

    async def cancellable_gateway(message, **kwargs):
        recorder.calls.append(f"gateway:{message}")
        if message == "cancel-me":
            gateway_started.set()
            await asyncio.Event().wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, cancellable_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    cancelled_task = asyncio.create_task(
        ingressor.route_event(_event("user", "cancel-me"), IDENTITY)
    )
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task

    assert ingressor._serial_gate.active_key_count == 0
    outcome = await asyncio.wait_for(
        ingressor.route_event(_event("user", "after-cancel"), IDENTITY),
        timeout=1,
    )
    assert outcome.kind == "user"
    assert ingressor.buffers.peek_buffer(_key()).event_count == 1
    assert ingressor._serial_gate.active_key_count == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_conversation_gate() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def blocking_gateway(**kwargs):
        gateway_started.set()
        await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    holder_task = asyncio.create_task(
        ingressor.route_event(_event("user", "u1"), IDENTITY)
    )
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    waiter_task = asyncio.create_task(
        ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    )
    await asyncio.sleep(0)

    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task
    assert ingressor._serial_gate.active_key_count == 1

    release_gateway.set()
    await asyncio.wait_for(holder_task, timeout=1)
    assert ingressor._serial_gate.active_key_count == 0


@pytest.mark.asyncio
async def test_manual_flush_waits_for_inflight_event() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def blocking_gateway(**kwargs):
        gateway_started.set()
        await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    user_task = asyncio.create_task(ingressor.route_event(_event("user", "u1"), IDENTITY))
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    flush_task = asyncio.create_task(ingressor.flush_conversation(_key(), IDENTITY))
    await asyncio.sleep(0)

    assert not flush_task.done()

    release_gateway.set()
    await asyncio.wait_for(user_task, timeout=1)
    assert await asyncio.wait_for(flush_task, timeout=1) == 1
    assert recorder.submitted[0].payload.user_message == "u1"
    assert ingressor._serial_gate.active_key_count == 0


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_event_before_sealing() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def blocking_gateway(**kwargs):
        gateway_started.set()
        await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(
        bus,
        submit_sealed_turn=recorder.submit,
    )

    user_task = asyncio.create_task(
        ingressor.route_event(_event("user", "u1"), IDENTITY)
    )
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    shutdown_task = asyncio.create_task(ingressor.shutdown_drain())
    await asyncio.sleep(0)

    assert not shutdown_task.done()

    release_gateway.set()
    await asyncio.wait_for(user_task, timeout=1)
    result = await asyncio.wait_for(shutdown_task, timeout=1)

    assert result == {
        "sealed_turns": 1,
        "submitted_turns": 1,
        "outbox_pending": 0,
    }
    assert recorder.submitted[0].seal_reason == "shutdown_drain"
    assert recorder.submitted[0].payload.user_message == "u1"
    assert ingressor._serial_gate.active_key_count == 0


@pytest.mark.asyncio
async def test_next_user_submits_previous_turn_before_gateway() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    recorder.calls.clear()

    await ingressor.route_event(_event("user", "u2"), IDENTITY)

    assert recorder.calls == ["submit", "gateway", "retrieve"]
    assert recorder.submitted[0].seal_reason == "next_user"
    assert recorder.submitted[0].payload.user_message == "u1"
    assert recorder.submitted[0].payload.assistant_final_text == "a1"


@pytest.mark.asyncio
async def test_gateway_failure_does_not_block_previous_turn_submit() -> None:
    recorder = _Recorder()
    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(bus, submit_sealed_turn=recorder.submit)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)

    # 让下一次 Gateway 调用失败
    bus.unregister(GlobalRoutes.GATEWAY_PROCESS)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(side_effect=RuntimeError("gateway down")),
    )

    # Gateway 可恢复失败走降级路径（详见 test_passive_degradation.py），
    # 这里只断言它不影响上一 turn 的提交。
    outcome = await ingressor.route_event(_event("user", "u2"), IDENTITY)
    assert outcome.gateway_decision is None

    # 上一 turn 已在 Gateway 调用前提交成功，不受新请求失败影响
    assert len(recorder.submitted) == 1
    assert recorder.submitted[0].payload.user_message == "u1"
    assert ingressor.outbox.pending_count(_key()) == 0


@pytest.mark.asyncio
async def test_non_user_events_do_not_call_gateway() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    recorder.calls.clear()

    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    await ingressor.route_event(
        _event("tool_call", "call", action_id="t1", tool_name="grep"),
        IDENTITY,
    )
    await ingressor.route_event(
        _event("tool_result", "out", action_id="t1", status="ok"),
        IDENTITY,
    )

    assert recorder.calls == []


# ========== outbox 失败语义 ==========


@pytest.mark.asyncio
async def test_submit_failure_retains_outbox_and_allows_next_turn() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)

    recorder.fail_submit = True
    outcome = await ingressor.route_event(_event("user", "u2"), IDENTITY)

    # 提交失败：sealed item 保留，但新 turn 已正常开始
    assert outcome.kind == "user"
    assert ingressor.outbox.pending_count(_key()) == 1
    assert recorder.submitted == []

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer is not None
    assert buffer.has_pending_round

    # 恢复后重试：先提交旧 turn，再提交当前 turn
    recorder.fail_submit = False
    await ingressor.route_event(_event("assistant", "a2"), IDENTITY)
    submitted = await ingressor.flush_conversation(_key(), IDENTITY)

    assert submitted == 2
    assert [s.payload.user_message for s in recorder.submitted] == ["u1", "u2"]
    assert recorder.submitted[0].attempts == 2
    assert ingressor.outbox.pending_count(_key()) == 0


@pytest.mark.asyncio
async def test_outbox_preserves_order_on_partial_failure() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    recorder.fail_submit = True
    for i in range(3):
        await ingressor.route_event(_event("user", f"u{i}"), IDENTITY)
        await ingressor.route_event(_event("assistant", f"a{i}"), IDENTITY)
        await ingressor.flush_conversation(_key(), IDENTITY)
    recorder.fail_submit = False

    assert ingressor.outbox.pending_count(_key()) == 3

    submitted = await ingressor.drain_outbox(_key())
    assert submitted == 3
    assert [s.payload.user_message for s in recorder.submitted] == ["u0", "u1", "u2"]


# ========== 幂等与会话隔离 ==========


@pytest.mark.asyncio
async def test_duplicate_external_event_id_is_idempotent() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    event = _event("user", "u1", external_event_id="evt-1")
    first = await ingressor.route_event(event, IDENTITY)
    second = await ingressor.route_event(event, IDENTITY)

    assert first.kind == "user"
    assert second.kind == "duplicate"
    assert recorder.calls.count("gateway") == 1

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer is not None
    assert buffer.event_count == 1


@pytest.mark.asyncio
async def test_same_event_id_from_different_source_is_not_duplicate() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(
        PassiveIngressEvent(
            source="src_a",
            external_conversation_id=CONVERSATION,
            external_event_id="evt-1",
            role="user",
            content="u1",
        ),
        IDENTITY,
    )
    outcome = await ingressor.route_event(
        PassiveIngressEvent(
            source="src_b",
            external_conversation_id=CONVERSATION,
            external_event_id="evt-1",
            role="user",
            content="u1",
        ),
        IDENTITY,
    )

    assert outcome.kind == "user"


@pytest.mark.asyncio
async def test_conversations_are_isolated_by_external_conversation_id() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "alpha", conversation="c-a"), IDENTITY)
    await ingressor.route_event(_event("user", "beta", conversation="c-b"), IDENTITY)
    await ingressor.route_event(
        _event("assistant", "reply-a", conversation="c-a"),
        IDENTITY,
    )

    submitted = await ingressor.flush_conversation(_key("c-a"), IDENTITY)
    assert submitted == 1
    assert recorder.submitted[0].payload.user_message == "alpha"
    assert recorder.submitted[0].payload.assistant_final_text == "reply-a"

    # c-b 的 accumulator 未被影响
    buffer_b = ingressor.buffers.peek_buffer(_key("c-b"))
    assert buffer_b is not None
    assert buffer_b.has_pending_round


# ========== seal 触发源同构 ==========


@pytest.mark.asyncio
async def test_explicit_final_seals_turn() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(
        _event("assistant", "a1", is_final=True),
        IDENTITY,
    )

    assert len(recorder.submitted) == 1
    assert recorder.submitted[0].seal_reason == "explicit_final"
    assert ingressor.outbox.pending_count(_key()) == 0
    assert not ingressor.buffers.peek_buffer(_key()).has_pending_round


@pytest.mark.asyncio
async def test_idle_timeout_seals_turn() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)
    ingressor.configure_idle_flush(timeout_seconds=-1.0)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)

    assert await ingressor.scan_idle_conversations_once() == 1
    assert recorder.submitted[0].seal_reason == "idle_timeout"


@pytest.mark.asyncio
async def test_shutdown_drain_seals_all_conversations() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u-a", conversation="c-a"), IDENTITY)
    await ingressor.route_event(_event("user", "u-b", conversation="c-b"), IDENTITY)

    result = await ingressor.shutdown_drain()

    assert result["sealed_turns"] == 2
    assert result["submitted_turns"] == 2
    assert result["outbox_pending"] == 0
    assert {s.seal_reason for s in recorder.submitted} == {"shutdown_drain"}


@pytest.mark.asyncio
async def test_all_seal_triggers_produce_isomorphic_payload() -> None:
    """不同触发源产出的 InteractionPayload 结构一致。"""
    payloads = {}

    for trigger in ("explicit_final", "next_user", "manual_flush", "idle_timeout"):
        recorder = _Recorder()
        ingressor = _build(recorder)
        ingressor.configure_idle_flush(timeout_seconds=-1.0)

        await ingressor.route_event(_event("user", "u1"), IDENTITY)
        await ingressor.route_event(
            _event("tool_call", "call", action_id="t1", tool_name="grep"),
            IDENTITY,
        )
        await ingressor.route_event(
            _event("tool_result", "out", action_id="t1", status="ok"),
            IDENTITY,
        )

        if trigger == "explicit_final":
            await ingressor.route_event(
                _event("assistant", "a1", is_final=True), IDENTITY
            )
        else:
            await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
            if trigger == "next_user":
                await ingressor.route_event(_event("user", "u2"), IDENTITY)
            elif trigger == "manual_flush":
                await ingressor.flush_conversation(_key(), IDENTITY)
            else:
                await ingressor.scan_idle_conversations_once()

        assert recorder.submitted, f"{trigger} 未产出 sealed turn"
        payload = recorder.submitted[0].payload
        payloads[trigger] = (
            payload.user_message,
            payload.assistant_final_text,
            [e.kind for e in payload.turn_events],
            payload.rewritten_query,
            payload.worth_saving,
        )

    reference = payloads["explicit_final"]
    assert reference[2] == [
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    for trigger, shape in payloads.items():
        assert shape == reference, f"{trigger} 产出的 payload 与其他触发源不同构"


# ========== 有界性 ==========


@pytest.mark.asyncio
async def test_turn_event_cap_is_bounded() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder, max_buffered_events_per_turn=3)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    for i in range(10):
        await ingressor.route_event(_event("assistant", f"a{i}"), IDENTITY)

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer.event_count == 3
    assert buffer.dropped_events == 8


@pytest.mark.asyncio
async def test_outbox_capacity_is_bounded() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder, max_outbox_items_per_conversation=2)
    recorder.fail_submit = True

    for i in range(4):
        await ingressor.route_event(_event("user", f"u{i}"), IDENTITY)
        await ingressor.route_event(_event("assistant", f"a{i}"), IDENTITY)
        await ingressor.drain_outbox(_key())

    assert ingressor.outbox.pending_count(_key()) == 2
