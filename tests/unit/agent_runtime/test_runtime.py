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
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    TurnEvent,
)
from tests.helpers.workspace import make_runtime_scope


def _frame(*, run_id: str = "run-1", frame_id: str = "frame-1") -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=make_runtime_scope(run_id=run_id, frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
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
    assert runtime.max_iterations == config.alice.runtime.max_loop_iterations


@pytest.mark.asyncio
async def test_agent_runtime_uses_injected_loop_executor():
    """门面使用注入的 loop_executor 执行帧（行为级验证注入 seam）。"""
    result = FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    loop_executor = SimpleNamespace(
        config=SimpleNamespace(max_loop_iterations=8),
        execute_frame=AsyncMock(return_value=result),
    )
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=MagicMock(),
        loop_executor=loop_executor,
    )

    actual = await runtime.run_frame(
        _frame(),
        output_sink=NullFrameOutputSink(),
    )

    assert actual is result
    loop_executor.execute_frame.assert_awaited_once()


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


def test_finalize_completed_frame_projects_only_acknowledged_aliases():
    """成功子帧只返回收到 ACK 的句柄，不把 UPDATE 基础 alias 当作新产物。"""
    runtime = AgentRuntime(
        mtp_executor=MagicMock(), runtime_config=MagicMock(), loop_executor=MagicMock()
    )
    frame = _frame()
    frame.add_harvested_alias("draft_accepted_1234")
    frame.add_harvested_alias("rev_fact_base_1234")
    frame.progress.turn_events.append(
        TurnEvent(
            kind="tool_call",
            sequence=0,
            role="assistant",
            content="update",
            tool_kind="UPDATE",
            target="fact_base",
        )
    )
    products = runtime.finalize_frame(
        frame, FrameExecutionResult(status=FrameExecutionStatus.COMPLETED)
    )
    assert products.artifact_aliases == ("draft_accepted_1234", "rev_fact_base_1234")


def test_finalize_unsuccessful_frame_does_not_harvest_aliases():
    """失败子帧不把部分完成的产物回填 caller，登记状态由进程负责。"""
    runtime = AgentRuntime(
        mtp_executor=MagicMock(), runtime_config=MagicMock(), loop_executor=MagicMock()
    )
    frame = _frame()
    frame.add_harvested_alias("draft_partial_1234")
    products = runtime.finalize_frame(
        frame, FrameExecutionResult(status=FrameExecutionStatus.FAILED)
    )
    assert products.artifact_aliases == ()
