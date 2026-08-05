from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.execution import AgentLoopExecutor
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.output import NullFrameOutputSink
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Identity,
    PendingAtomStatus,
    RuntimeScope,
    TurnEvent,
)
from hivememory.system.config import HiveMemoryConfig
from hivememory.system.model_registry import ModelNotFoundError


def _runtime_with_pending(pending_runtime: PendingAtomRuntime) -> AgentRuntime:
    return AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=MagicMock(),
        pending_runtime=pending_runtime,
    )


def _frame(*, run_id: str = "run-1", frame_id: str = "frame-1") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(run_id=run_id, frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
        identity=Identity(user_id="user-1", agent_id="omni_doll"),
    )


def test_agent_runtime_builds_engine_facade():
    """AgentRuntime 作为单 Agent 运行时门面，内部装配 loop_executor 引擎。"""
    config = HiveMemoryConfig()
    mtp_executor = MagicMock()

    runtime = AgentRuntime(
        mtp_executor=mtp_executor,
        runtime_config=config.alice.runtime,
    )

    assert isinstance(runtime._loop_executor, AgentLoopExecutor)
    assert runtime._loop_executor._mtp_executor is mtp_executor
    assert runtime._max_iterations == config.alice.runtime.max_loop_iterations


def test_agent_runtime_accepts_injected_loop_executor():
    """门面支持注入预构建的 loop_executor（测试/高级装配 seam）。"""
    injected = MagicMock()

    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=injected,
    )

    assert runtime._loop_executor is injected


@pytest.mark.asyncio
async def test_run_frame_maps_missing_model_to_failed_outcome():
    loop_executor = SimpleNamespace(
        config=SimpleNamespace(max_loop_iterations=8),
        execute_frame=AsyncMock(),
    )
    model_registry = MagicMock()
    error = ModelNotFoundError("missing")
    model_registry.resolve.side_effect = error
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=loop_executor,
        model_registry=model_registry,
    )

    result = await runtime.run_frame(
        _frame(),
        output_sink=NullFrameOutputSink(),
    )

    assert result.status == FrameExecutionStatus.FAILED
    assert result.error is error
    loop_executor.execute_frame.assert_not_awaited()


def test_finalize_completed_frame_projects_pending_and_update_aliases():
    pending_runtime = PendingAtomRuntime()
    runtime = _runtime_with_pending(pending_runtime)
    frame = _frame()
    atom = pending_runtime.register_write(
        content="draft",
        title="Draft",
        reason=None,
        identity=frame.identity,
        runtime_scope=frame.runtime_scope,
    )
    frame.add_harvested_alias("draft_existing")
    frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="update",
            tool_kind="UPDATE",
            target="fact_existing",
        )
    )

    products = runtime.finalize_frame(
        frame,
        FrameExecutionResult(status=FrameExecutionStatus.COMPLETED),
    )

    assert products.artifact_aliases == (
        "draft_existing",
        atom.pending_alias,
        "fact_existing",
    )
    assert frame.harvested_aliases == list(products.artifact_aliases)
    assert atom.status == PendingAtomStatus.PENDING


def test_finalize_unsuccessful_frame_cancels_atoms_without_products():
    pending_runtime = PendingAtomRuntime()
    runtime = _runtime_with_pending(pending_runtime)
    frame = _frame()
    atom = pending_runtime.register_write(
        content="draft",
        title="Draft",
        reason=None,
        identity=frame.identity,
        runtime_scope=frame.runtime_scope,
    )

    products = runtime.finalize_frame(
        frame,
        FrameExecutionResult(status=FrameExecutionStatus.FAILED),
    )

    assert products.artifact_aliases == ()
    assert atom.status == PendingAtomStatus.CANCELLED


def test_finalize_completed_run_claims_tasks_and_advances_retention_epoch():
    pending_runtime = PendingAtomRuntime()
    runtime = _runtime_with_pending(pending_runtime)
    identity = Identity(user_id="user-1", agent_id="omni_doll")
    old_atom = pending_runtime.register_write(
        content="old",
        title="Old",
        reason=None,
        identity=identity,
        runtime_scope=RuntimeScope(run_id="run-old", frame_id="frame-old"),
    )
    pending_runtime.cancel(old_atom.pending_alias)
    current_atom = pending_runtime.register_write(
        content="current",
        title="Current",
        reason=None,
        identity=identity,
        runtime_scope=RuntimeScope(run_id="run-current", frame_id="frame-current"),
    )

    products = runtime.finalize_run(
        "run-current",
        FrameExecutionResult(status=FrameExecutionStatus.COMPLETED),
    )

    assert [task.pending_alias for task in products.materialize_tasks] == [
        current_atom.pending_alias
    ]
    assert current_atom.status == PendingAtomStatus.MATERIALIZING
    assert old_atom.status == PendingAtomStatus.EXPIRED

    runtime.finalize_run(
        "run-next",
        FrameExecutionResult(status=FrameExecutionStatus.COMPLETED),
    )
    assert pending_runtime.get(old_atom.pending_alias) is None


def test_finalize_unsuccessful_run_does_not_advance_retention_epoch():
    pending_runtime = PendingAtomRuntime()
    runtime = _runtime_with_pending(pending_runtime)
    identity = Identity(user_id="user-1", agent_id="omni_doll")
    old_atom = pending_runtime.register_write(
        content="old",
        title="Old",
        reason=None,
        identity=identity,
        runtime_scope=RuntimeScope(run_id="run-old", frame_id="frame-old"),
    )
    pending_runtime.cancel(old_atom.pending_alias)
    current_atom = pending_runtime.register_write(
        content="current",
        title="Current",
        reason=None,
        identity=identity,
        runtime_scope=RuntimeScope(run_id="run-current", frame_id="frame-current"),
    )

    products = runtime.finalize_run(
        "run-current",
        FrameExecutionResult(status=FrameExecutionStatus.CANCELLED),
    )

    assert products.materialize_tasks == ()
    assert current_atom.status == PendingAtomStatus.CANCELLED
    assert old_atom.status == PendingAtomStatus.CANCELLED
