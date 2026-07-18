"""Phase 3F 主动聊天 Gateway 编排测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
    GatewayCancelledError,
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

    events = [
        event
        async for event in ChatApplicationService(bus).chat_stream("/clear", "u1")
    ]

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
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(side_effect=GatewayCancelledError("cancelled")),
    )
    service = ChatApplicationService(bus)

    result = await service.chat("问题", "u1")
    stream_events = [
        event async for event in service.chat_stream("问题", "u1")
    ]

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"
    assert [event["event"] for event in stream_events] == [
        "generation_id",
        "done",
    ]
    assert stream_events[-1]["data"]["status"] == "cancelled"


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
        AsyncMock(return_value=prepared),
    )
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(
            return_value=AgentRunResult(status=AgentRunStatus.CANCELLED)
        ),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        cleanup,
    )

    result = await ChatApplicationService(bus).chat("问题", "u1")

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"
    cleanup.assert_awaited_once_with(prepared_run=prepared)
