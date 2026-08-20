"""Phase 3F 主动聊天 Gateway 编排测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayCommandOutcome,
    GatewayDecision,
    GatewayDecisionOutcome,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunResult,
    AgentRunStatus,
)
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def _decision_outcome() -> GatewayDecisionOutcome:
    return GatewayDecisionOutcome(
        decision=GatewayDecision(
            target_topic_id="topic-1",
            rewritten_query="原问题",
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(),
            intent_type=IntentType.RAG,
        )
    )


def _command_outcome() -> GatewayCommandOutcome:
    return GatewayCommandOutcome(
        command_execution_result=CommandExecutionResult(
            command_id="system.clear",
            status=CommandExecutionStatus.COMPLETED,
            message="已清空聊天。",
            client_action={"type": "clear_chat"},
        )
    )


def _scoped_prepared_route(prepared):
    async def route(*, identity_scope, **_kwargs):
        prepared.identity_scope = identity_scope
        return prepared

    return route


@pytest.mark.asyncio
async def test_non_streaming_command_short_circuits_patchouli_and_alice() -> None:
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)

    result = await ChatApplicationService(bus).chat("/clear", "u1")

    assert result.kind == "command"
    assert result.command_execution_result.command_id == "system.clear"
    assert bus.list_routes() == [GlobalRoutes.GATEWAY_PROCESS]


@pytest.mark.asyncio
async def test_non_streaming_decision_uses_one_prepare_run_finalize_sequence() -> None:
    bus = GlobalSystemBus()
    calls: list[str] = []
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"

    async def gateway(**_kwargs):
        calls.append("gateway")
        return _decision_outcome()

    async def prepare(**kwargs):
        calls.append("prepare")
        assert kwargs["gateway_decision"] == _decision_outcome().decision
        prepared.identity_scope = kwargs["identity_scope"]
        return prepared

    async def run_agent(**_kwargs):
        calls.append("alice")
        return AgentRunResult(final_text="完成")

    async def finalize(**_kwargs):
        calls.append("finalize")
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, run_agent)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await ChatApplicationService(bus).chat("问题", "u1")

    assert result.kind == "agent"
    assert result.agent_run_result.final_text == "完成"
    assert calls == ["gateway", "prepare", "alice", "finalize"]


@pytest.mark.asyncio
async def test_streaming_command_emits_result_and_done_only() -> None:
    bus = GlobalSystemBus()
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_command_outcome()),
    )

    events = [event async for event in ChatApplicationService(bus).chat_stream("/clear", "u1")]

    assert [event["event"] for event in events] == [
        "generation_id",
        "command_result",
        "done",
    ]
    assert events[1]["data"]["client_action"] == {"type": "clear_chat"}
    assert events[2]["data"]["final_text"] == "已清空聊天。"


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service = ChatApplicationService(bus)

    task = asyncio.create_task(service.chat("问题", "u1", generation_id="gen-gateway"))
    await started.wait()
    stop_result = service.cancel_generation("gen-gateway", user_id="u1")
    result = await task

    assert stop_result.cancelled is True

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"


@pytest.mark.asyncio
async def test_non_streaming_cancel_after_prepare_cleans_prepared_run() -> None:
    bus = GlobalSystemBus()
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"
    cleanup = AsyncMock(return_value=True)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(prepared),
    )
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(status=AgentRunStatus.CANCELLED)),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        cleanup,
    )

    result = await ChatApplicationService(bus).chat("问题", "u1")

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"
    cleanup.assert_awaited_once_with(prepared_run=prepared)


@pytest.mark.asyncio
async def test_non_streaming_failed_agent_run_is_not_rewritten_as_cancelled() -> None:
    bus = GlobalSystemBus()
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"
    prepared.stream_prelude.topic_id = "topic-1"
    prepared.stream_prelude.is_new_topic = False
    prepared.stream_prelude.pool_topics = []
    prepared.stream_prelude.memory_refs = []
    finalize = AsyncMock(return_value=[])
    cleanup = AsyncMock(return_value=True)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(prepared),
    )
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(status=AgentRunStatus.FAILED)),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    result = await ChatApplicationService(bus).chat("问题", "u1")

    assert result.agent_run_result.status == AgentRunStatus.FAILED.value
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once_with(prepared_run=prepared)


@pytest.mark.asyncio
async def test_streaming_failed_agent_run_preserves_failed_done_status() -> None:
    bus = GlobalSystemBus()
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"
    prepared.stream_prelude.topic_id = "topic-1"
    prepared.stream_prelude.is_new_topic = False
    prepared.stream_prelude.pool_topics = []
    prepared.stream_prelude.memory_refs = []

    async def alice_stream(**_kwargs):
        yield {
            "event": "done",
            "data": AgentRunResult(status=AgentRunStatus.FAILED).model_dump(),
        }

    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(prepared),
    )
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, AsyncMock(return_value=alice_stream()))
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    events = [event async for event in ChatApplicationService(bus).chat_stream("问题", "u1")]

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == AgentRunStatus.FAILED.value
    assert events[-1]["data"]["stopped"] is True
    cleanup.assert_awaited_once_with(prepared_run=prepared)


@pytest.mark.asyncio
async def test_stop_during_prepare_waits_for_prepare_then_skips_alice_and_finalize() -> None:
    bus = GlobalSystemBus()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    prepare_cancelled = False
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"

    async def prepare(*, identity_scope, **_kwargs):
        nonlocal prepare_cancelled
        prepare_started.set()
        try:
            await release_prepare.wait()
        except asyncio.CancelledError:
            prepare_cancelled = True
            raise
        prepared.identity_scope = identity_scope
        return prepared

    alice = AsyncMock()
    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = ChatApplicationService(bus)

    task = asyncio.create_task(service.chat("问题", "u1", generation_id="gen-prepare"))
    await prepare_started.wait()
    stop_result = service.cancel_generation("gen-prepare", user_id="u1")
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert prepare_cancelled is False
    assert result.agent_run_result.status == AgentRunStatus.CANCELLED.value
    alice.assert_not_awaited()
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once_with(prepared_run=prepared)


@pytest.mark.asyncio
async def test_stream_stop_cancels_current_alice_pull_and_closes_stream() -> None:
    bus = GlobalSystemBus()
    pull_started = asyncio.Event()
    stream_closed = asyncio.Event()
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"
    prepared.stream_prelude.topic_id = "topic-1"
    prepared.stream_prelude.is_new_topic = False
    prepared.stream_prelude.pool_topics = []
    prepared.stream_prelude.memory_refs = []

    async def alice_stream():
        try:
            pull_started.set()
            await asyncio.Event().wait()
            yield {"event": "token", "data": {"content": "late"}}
        finally:
            stream_closed.set()

    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(prepared),
    )
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, AsyncMock(return_value=alice_stream()))
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = ChatApplicationService(bus)

    task = asyncio.create_task(
        _collect_stream(service, generation_id="gen-stream-cancel")
    )
    await pull_started.wait()
    stop_result = service.cancel_generation("gen-stream-cancel", user_id="u1")
    events = await task

    assert stop_result.cancelled is True
    assert stream_closed.is_set()
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once_with(prepared_run=prepared)


@pytest.mark.asyncio
async def test_stop_during_finalize_is_rejected_and_finalize_completes() -> None:
    bus = GlobalSystemBus()
    finalize_started = asyncio.Event()
    release_finalize = asyncio.Event()
    prepared = AsyncMock()
    prepared.agent_run_context = object()
    prepared.generation_options = None
    prepared.topic_id = "topic-1"

    async def finalize(**_kwargs):
        finalize_started.set()
        await release_finalize.wait()
        return []

    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(prepared),
    )
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(final_text="完成")),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = ChatApplicationService(bus)

    task = asyncio.create_task(service.chat("问题", "u1", generation_id="gen-finalize"))
    await finalize_started.wait()
    stop_result = service.cancel_generation("gen-finalize", user_id="u1")
    release_finalize.set()
    result = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    assert result.agent_run_result.status == AgentRunStatus.COMPLETED.value
    cleanup.assert_not_awaited()


async def _collect_stream(
    service: ChatApplicationService,
    *,
    generation_id: str,
) -> list[dict]:
    return [
        event
        async for event in service.chat_stream(
            "问题",
            "u1",
            generation_id=generation_id,
        )
    ]
