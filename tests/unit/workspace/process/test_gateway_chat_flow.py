"""Phase 3F 主动聊天 Gateway 编排测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import AssetNotReadyError, WorkspaceDomainError
from hivememory.core.models import (
    AttachmentSelectionRequest,
    IdentityScope,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    WorkspaceAssetRef,
)
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
    RetrievalResponse,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _u1_scope() -> IdentityScope:
    """Chat 是 Agent action：构造携带具体 Agent 的显式 scope。"""
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


async def _run_once(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str | None = None,
    **kwargs,
):
    return await service.run_process(
        stream=False,
        message=message,
        identity_scope=_u1_scope(),
        process_id=process_id or f"process_{uuid4().hex}",
        **kwargs,
    )


async def _stream_events(
    service: TaskProcessService,
    message: str,
    *,
    process_id: str | None = None,
) -> list[dict]:
    return [
        event
        async for event in service.run_process(
            message=message,
            identity_scope=_u1_scope(),
            process_id=process_id or f"process_{uuid4().hex}",
        )
    ]


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


def _memory_ref_atom() -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id="u1"),
        index=IndexLayer(
            title="引用记忆",
            summary="引用摘要",
            tags=["t"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="引用正文"),
    )


def _scoped_prepared_route(
    *,
    memories: list[MemoryAtom] | None = None,
    topic_id: str = "topic-1",
    is_new_topic: bool = False,
):
    """prepare 替身：按请求 scope 构造真实 PreparedAgentRun（进程 CPU 分配的输入）。"""

    async def route(
        *,
        identity_scope,
        user_message,
        interaction_id,
        gateway_decision,
        **_kwargs,
    ):
        return PreparedAgentRun(
            identity_scope=identity_scope,
            interaction_id=interaction_id,
            user_message=user_message,
            gateway_decision=gateway_decision,
            topic_id=topic_id,
            is_new_topic=is_new_topic,
            topic_context=None,
            pool_topics=[],
            retrieval_result=RetrievalResponse.from_memories(list(memories or [])),
            storage_available=True,
        )

    return route


def _profile_route():
    """PATCHOULI_GET_AGENT_PROFILE 替身：返回 builtin omni_doll Profile。"""
    from hivememory.core.models import OMNI_DOLL_PROFILE, ResolvedAgentProfile

    async def route(agent_id, *, identity_scope):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    return route


def _register_profile(bus: GlobalSystemBus) -> None:
    bus.register(GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE, _profile_route())


@pytest.mark.asyncio
async def test_non_streaming_command_short_circuits_patchouli_and_alice() -> None:
    bus = GlobalSystemBus()
    gateway = AsyncMock(return_value=_command_outcome())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)

    result = await _run_once(TaskProcessService(bus), "/clear")

    assert result.kind == "command"
    assert result.command_execution_result.command_id == "system.clear"
    assert bus.list_routes() == [GlobalRoutes.GATEWAY_PROCESS]


@pytest.mark.asyncio
async def test_non_streaming_decision_uses_one_prepare_run_finalize_sequence() -> None:
    bus = GlobalSystemBus()
    calls: list[str] = []

    async def gateway(**_kwargs):
        calls.append("gateway")
        return _decision_outcome()

    async def prepare(**kwargs):
        calls.append("prepare")
        assert kwargs["gateway_decision"] == _decision_outcome().decision
        return await _scoped_prepared_route()(**kwargs)

    async def run_agent(**_kwargs):
        calls.append("alice")
        return AgentRunResult(final_text="完成")

    async def finalize(**_kwargs):
        calls.append("finalize")
        return []

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, run_agent)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)

    result = await _run_once(TaskProcessService(bus), "问题")

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

    events = await _stream_events(TaskProcessService(bus), "/clear")

    assert [event["event"] for event in events] == [
        "process_id",
        "command_result",
        "done",
    ]
    assert events[1]["data"]["client_action"] == {"type": "clear_chat"}
    assert events[2]["data"]["final_text"] == "已清空聊天。"


@pytest.mark.asyncio
async def test_completed_stream_uses_one_process_id_for_events_and_downstream_routes() -> None:
    """完成的流式进程：SSE 首个事件、finalizing 状态与 done 携带同一 process_id，
    并以它调用 Patchouli prepare（interaction_id）与 Alice（清单 process_id）；
    topic_info 与 memory_refs 由进程从 prepare 结果推导。"""
    bus = GlobalSystemBus()
    prepare_calls: list[dict] = []

    async def prepare(*, identity_scope, **kwargs):
        prepare_calls.append(kwargs)
        route = _scoped_prepared_route(memories=[_memory_ref_atom()])
        return await route(identity_scope=identity_scope, **kwargs)

    async def alice_stream(**_kwargs):
        yield {"event": "token", "data": {"content": "完成"}}
        yield {"event": "done", "data": AgentRunResult(final_text="完成").model_dump()}

    alice = AsyncMock(return_value=alice_stream())
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, AsyncMock(return_value=[]))
    bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, AsyncMock(return_value=[]))

    events = await _stream_events(
        TaskProcessService(bus),
        "问题",
        process_id="process-complete",
    )

    assert events[0] == {"event": "process_id", "data": {"process_id": "process-complete"}}
    topic_info = next(event for event in events if event["event"] == "topic_info")
    assert topic_info["data"] == {
        "topic_id": "topic-1",
        "is_new": False,
        "pool_topics": [],
    }
    memory_refs = next(event for event in events if event["event"] == "memory_refs")
    assert memory_refs["data"]["memories"][0]["title"] == "引用记忆"
    assert [event["data"] for event in events if event["event"] == "run_status"] == [
        {"process_id": "process-complete", "status": "finalizing"}
    ]
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    assert events[-1]["data"]["process_id"] == "process-complete"
    assert [call["interaction_id"] for call in prepare_calls] == ["process-complete"]
    assert alice.await_args.kwargs["input_manifest"].process_id == "process-complete"


@pytest.mark.asyncio
async def test_gateway_cancellation_maps_to_cancelled_agent_outcomes() -> None:
    bus = GlobalSystemBus()
    started = asyncio.Event()

    async def gateway(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway)
    service = TaskProcessService(bus)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-gateway"))
    await started.wait()
    stop_result = service.cancel_process("process-gateway", identity_scope=_u1_scope())
    result = await task

    assert stop_result.cancelled is True

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"


@pytest.mark.asyncio
async def test_non_streaming_cancel_after_prepare_cleans_prepared_run() -> None:
    bus = GlobalSystemBus()
    prepared_holder: dict = {}

    async def prepare(**kwargs):
        prepared = await _scoped_prepared_route()(**kwargs)
        prepared_holder["prepared"] = prepared
        return prepared

    cleanup = AsyncMock(return_value=True)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        prepare,
    )
    _register_profile(bus)
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(status=AgentRunStatus.CANCELLED)),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN,
        cleanup,
    )

    result = await _run_once(TaskProcessService(bus), "问题")

    assert result.kind == "agent"
    assert result.agent_run_result.status == "cancelled"
    cleanup.assert_awaited_once_with(prepared_run=prepared_holder["prepared"])


@pytest.mark.asyncio
async def test_non_streaming_failed_agent_run_is_not_rewritten_as_cancelled() -> None:
    bus = GlobalSystemBus()
    finalize = AsyncMock(return_value=[])
    cleanup = AsyncMock(return_value=True)
    bus.register(
        GlobalRoutes.GATEWAY_PROCESS,
        AsyncMock(return_value=_decision_outcome()),
    )
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(status=AgentRunStatus.FAILED)),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    result = await _run_once(TaskProcessService(bus), "问题")

    assert result.agent_run_result.status == AgentRunStatus.FAILED.value
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_failed_agent_run_preserves_failed_done_status() -> None:
    bus = GlobalSystemBus()

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
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, AsyncMock(return_value=alice_stream()))
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    events = await _stream_events(TaskProcessService(bus), "问题")

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == AgentRunStatus.FAILED.value
    assert events[-1]["data"]["stopped"] is True
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_during_prepare_waits_for_prepare_then_skips_alice_and_finalize() -> None:
    bus = GlobalSystemBus()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    prepare_cancelled = False

    async def prepare(*, identity_scope, **kwargs):
        nonlocal prepare_cancelled
        prepare_started.set()
        try:
            await release_prepare.wait()
        except asyncio.CancelledError:
            prepare_cancelled = True
            raise
        route = _scoped_prepared_route()
        return await route(identity_scope=identity_scope, **kwargs)

    alice = AsyncMock()
    finalize = AsyncMock()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    _register_profile(bus)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = TaskProcessService(bus)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-prepare"))
    await prepare_started.wait()
    stop_result = service.cancel_process("process-prepare", identity_scope=_u1_scope())
    release_prepare.set()
    result = await task

    assert stop_result.cancelled is True
    assert prepare_cancelled is False
    assert result.agent_run_result.status == AgentRunStatus.CANCELLED.value
    alice.assert_not_awaited()
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_stop_cancels_current_alice_pull_and_closes_stream() -> None:
    bus = GlobalSystemBus()
    pull_started = asyncio.Event()
    stream_closed = asyncio.Event()

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
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(GlobalRoutes.ALICE_RUN_AGENT_STREAM, AsyncMock(return_value=alice_stream()))
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = TaskProcessService(bus)

    task = asyncio.create_task(_collect_stream(service, process_id="process-stream-cancel"))
    await pull_started.wait()
    stop_result = service.cancel_process("process-stream-cancel", identity_scope=_u1_scope())
    events = await task

    assert stop_result.cancelled is True
    assert stream_closed.is_set()
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "cancelled"
    finalize.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_during_finalize_is_rejected_and_finalize_completes() -> None:
    bus = GlobalSystemBus()
    finalize_started = asyncio.Event()
    release_finalize = asyncio.Event()

    async def finalize(**_kwargs):
        finalize_started.set()
        await release_finalize.wait()
        return []

    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(
        GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN,
        _scoped_prepared_route(),
    )
    _register_profile(bus)
    bus.register(
        GlobalRoutes.ALICE_RUN_AGENT,
        AsyncMock(return_value=AgentRunResult(final_text="完成")),
    )
    bus.register(GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN, finalize)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)
    service = TaskProcessService(bus)

    task = asyncio.create_task(_run_once(service, "问题", process_id="process-finalize"))
    await finalize_started.wait()
    stop_result = service.cancel_process("process-finalize", identity_scope=_u1_scope())
    release_finalize.set()
    result = await task

    assert stop_result.cancelled is False
    assert stop_result.reason == "already_finalizing"
    assert result.agent_run_result.status == AgentRunStatus.COMPLETED.value
    cleanup.assert_not_awaited()


async def _collect_stream(
    service: TaskProcessService,
    *,
    process_id: str,
) -> list[dict]:
    return await _stream_events(service, "问题", process_id=process_id)


@pytest.mark.asyncio
async def test_attachment_selection_without_reader_fails_allocation() -> None:
    """捕获装配遗漏：有附件选择但 Store 未注入时，分配显式失败且不进入 Alice。"""
    bus = GlobalSystemBus()
    cleanup = AsyncMock(return_value=True)
    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, _scoped_prepared_route())
    _register_profile(bus)
    alice = AsyncMock(return_value=AgentRunResult(final_text="完成"))
    bus.register(GlobalRoutes.ALICE_RUN_AGENT, alice)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, cleanup)

    with pytest.raises(WorkspaceDomainError, match="附件读取能力"):
        await _run_once(
            TaskProcessService(bus),
            "问题",
            process_id="process-no-reader",
            attachments=[
                AttachmentSelectionRequest(
                    asset_ref=WorkspaceAssetRef(token="ref-a", asset_id="asset-a"),
                ),
            ],
        )

    alice.assert_not_awaited()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_workspace_domain_error_yields_safe_code() -> None:
    """捕获附件拒绝等 Workspace 领域错误丢失安全 code 或被包装成系统错误。"""
    bus = GlobalSystemBus()

    async def prepare(**_kwargs):
        raise AssetNotReadyError("附件尚未完成解析，不能在本轮使用")

    bus.register(GlobalRoutes.GATEWAY_PROCESS, AsyncMock(return_value=_decision_outcome()))
    _register_profile(bus)
    bus.register(GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN, prepare)
    bus.register(GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN, AsyncMock(return_value=False))

    events = [
        event
        async for event in TaskProcessService(bus).run_process(
            "问题",
            identity_scope=_u1_scope(),
            process_id="process-domain-error",
        )
    ]

    errors = [event for event in events if event["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == "workspace.asset.not_ready"
    assert "附件" in errors[0]["data"]["message"]
