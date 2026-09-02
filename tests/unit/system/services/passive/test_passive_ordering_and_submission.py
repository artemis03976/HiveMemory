"""Passive ingress 读写时序、submission admission 与幂等契约测试。"""

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
from tests.helpers.workspace import make_identity_scope

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
        identity_scope=make_identity_scope(user_id="u1", agent_id="a1"),
    )


class _Recorder:
    """记录 Gateway、retrieval 与 queue admission 调用顺序。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.submitted: list = []

    async def gateway(self, **kwargs):
        self.calls.append("gateway")
        return _decision()

    async def retrieve(self, **kwargs):
        self.calls.append("retrieve")
        return RetrievalResponse()

    async def submit(self, submission):
        self.calls.append("submit")
        self.submitted.append(submission)


def _build(
    recorder: _Recorder,
    *,
    bus: GlobalSystemBus | None = None,
    **config_kwargs,
) -> PassiveMessageIngressor:
    bus = bus or GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, recorder.gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    return PassiveMessageIngressor(
        bus,
        interaction_queue=recorder,
        config=PassiveIngressConfig(**config_kwargs) if config_kwargs else None,
    )


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
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    user_task = asyncio.create_task(ingressor.route_event(_event("user", "u1"), IDENTITY))
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    assistant_task = asyncio.create_task(ingressor.route_event(_event("assistant", "a1"), IDENTITY))
    await asyncio.sleep(0)

    # 同一会话的 assistant 事件被串行门阻塞，等待 user 的 gateway 完成
    assert not assistant_task.done()

    release_gateway.set()
    user_outcome, assistant_outcome = await asyncio.wait_for(
        asyncio.gather(user_task, assistant_task),
        timeout=1,
    )
    assert user_outcome.kind == "user"
    assert assistant_outcome.kind == "buffered"
    assert await ingressor.flush_conversation(_key()) == 1
    assert recorder.submitted[0].payload.user_message == "u1"
    assert recorder.submitted[0].payload.assistant_final_text == "a1"


@pytest.mark.asyncio
async def test_different_conversations_remain_concurrent() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()
    release_gateway = asyncio.Event()

    async def selectively_blocking_gateway(message, **kwargs):
        if message == "u-a":
            gateway_started.set()
            await release_gateway.wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, selectively_blocking_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    await ingressor.route_event(_event("user", "u-b", conversation="c-b"), IDENTITY)
    blocked_task = asyncio.create_task(
        ingressor.route_event(_event("user", "u-a", conversation="c-a"), IDENTITY)
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
    blocked_outcome = await asyncio.wait_for(blocked_task, timeout=1)
    # 阻塞的 c-a 事件在门释放后正常完成
    assert blocked_outcome.kind == "user"


@pytest.mark.asyncio
async def test_cancelled_event_releases_conversation_gate() -> None:
    recorder = _Recorder()
    gateway_started = asyncio.Event()

    async def cancellable_gateway(message, **kwargs):
        if message == "cancel-me":
            gateway_started.set()
            await asyncio.Event().wait()
        return _decision()

    bus = GlobalSystemBus()
    bus.register(GlobalRoutes.GATEWAY_PROCESS, cancellable_gateway)
    bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, recorder.retrieve)
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    cancelled_task = asyncio.create_task(
        ingressor.route_event(_event("user", "cancel-me"), IDENTITY)
    )
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task

    # 门已释放：同一会话后续事件能正常完成（不再被串行门阻塞）
    outcome = await asyncio.wait_for(
        ingressor.route_event(_event("user", "after-cancel"), IDENTITY),
        timeout=1,
    )
    assert outcome.kind == "user"


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
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    user_task = asyncio.create_task(ingressor.route_event(_event("user", "u1"), IDENTITY))
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    flush_task = asyncio.create_task(ingressor.flush_conversation(_key()))
    await asyncio.sleep(0)
    assert not flush_task.done()

    release_gateway.set()
    await user_task
    assert await flush_task == 1


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_event_before_finalizing() -> None:
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
    ingressor = PassiveMessageIngressor(bus, interaction_queue=recorder)

    user_task = asyncio.create_task(ingressor.route_event(_event("user", "u1"), IDENTITY))
    await asyncio.wait_for(gateway_started.wait(), timeout=1)
    shutdown_task = asyncio.create_task(ingressor.shutdown_drain())
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    release_gateway.set()
    await user_task
    result = await shutdown_task
    assert result == {"finalized_turns": 1, "accepted_submissions": 1}
    assert recorder.submitted[0].correlation["seal_reason"] == "shutdown_drain"


@pytest.mark.asyncio
async def test_next_user_submits_previous_turn_before_gateway() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    recorder.calls.clear()
    await ingressor.route_event(_event("user", "u2"), IDENTITY)

    assert recorder.calls == ["submit", "gateway", "retrieve"]
    assert recorder.submitted[0].correlation["seal_reason"] == "next_user"
    assert recorder.submitted[0].payload.user_message == "u1"


@pytest.mark.asyncio
async def test_gateway_failure_does_not_block_previous_turn_submit() -> None:
    recorder = _Recorder()
    bus = GlobalSystemBus()
    ingressor = _build(recorder, bus=bus)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
    bus.unregister(GlobalRoutes.GATEWAY_PROCESS)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(side_effect=RuntimeError("gateway down")),
    )

    outcome = await ingressor.route_event(_event("user", "u2"), IDENTITY)
    assert outcome.gateway_decision is None
    assert len(recorder.submitted) == 1
    assert recorder.submitted[0].payload.user_message == "u1"


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
    assert ingressor.buffers.peek_buffer(_key()).event_count == 1


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

    assert await ingressor.flush_conversation(_key("c-a")) == 1
    assert recorder.submitted[0].payload.user_message == "alpha"
    assert ingressor.buffers.peek_buffer(_key("c-b")).has_pending_round


@pytest.mark.asyncio
async def test_all_finalize_triggers_produce_isomorphic_payload() -> None:
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
                _event("assistant", "a1", is_final=True),
                IDENTITY,
            )
        else:
            await ingressor.route_event(_event("assistant", "a1"), IDENTITY)
            if trigger == "next_user":
                await ingressor.route_event(_event("user", "u2"), IDENTITY)
            elif trigger == "manual_flush":
                await ingressor.flush_conversation(_key())
            else:
                await ingressor.scan_idle_conversations_once()

        payload = recorder.submitted[0].payload
        payloads[trigger] = (
            payload.user_message,
            payload.assistant_final_text,
            [event.kind for event in payload.turn_events],
            payload.rewritten_query,
            payload.worth_saving,
        )

    assert all(shape == payloads["explicit_final"] for shape in payloads.values())


@pytest.mark.asyncio
async def test_turn_event_cap_is_bounded() -> None:
    recorder = _Recorder()
    ingressor = _build(recorder, max_buffered_events_per_turn=3)

    await ingressor.route_event(_event("user", "u1"), IDENTITY)
    for index in range(10):
        await ingressor.route_event(_event("assistant", f"a{index}"), IDENTITY)

    buffer = ingressor.buffers.peek_buffer(_key())
    assert buffer.event_count == 3
    assert buffer.dropped_events == 8
