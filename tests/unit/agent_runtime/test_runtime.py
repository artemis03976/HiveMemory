from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionResult,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.loop import AgentLoopExecutor
from hivememory.agent_runtime.output import NullFrameOutputSink
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.config.alice import AgentRuntimeConfig
from hivememory.config.shared import LLMConfig
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    TurnEvent,
)
from tests.helpers.call import StubOperationSubmitter
from tests.helpers.workspace import make_runtime_scope


def _frame(*, run_id: str = "run-1", frame_id: str = "frame-1") -> ExecutionFrame:
    return ExecutionFrame(
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        submit_operation=StubOperationSubmitter(),
        runtime_scope=make_runtime_scope(run_id=run_id, frame_id=frame_id),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[],
        topic_id="topic-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("frame_limit", "expected_iterations"), [(None, 2), (1, 1)])
async def test_built_runtime_uses_config_budget_unless_frame_overrides(
    monkeypatch, frame_limit, expected_iterations
):
    """完整装配的内核执行 MTP；帧未覆盖上限时统一使用运行时预算。"""
    from hivememory.agent_runtime import GenerationResult, LLMClient, build_agent_runtime
    from hivememory.config.alice import KoakumaConfig
    from hivememory.config.memory_compiler import MemoryCompilerConfig

    command = "⟪ RUN | sys_clock | ⟫"
    monkeypatch.setattr(
        LLMClient,
        "generate_async",
        AsyncMock(
            return_value=GenerationResult(
                text=command, was_mtp_interrupted=True, mtp_fragment=command
            )
        ),
    )
    runtime = build_agent_runtime(
        runtime_config=AgentRuntimeConfig(max_loop_iterations=2),
        koakuma_config=KoakumaConfig(),
        memory_compiler_config=MemoryCompilerConfig(),
    )
    frame = _frame()
    frame.execution_policy = FrameExecutionPolicy.from_verbs({"RUN"}, max_iterations=frame_limit)

    result = await runtime.run_frame(frame, output_sink=NullFrameOutputSink())

    assert result.status == FrameExecutionStatus.BUDGET_EXHAUSTED
    assert frame.progress.iteration == expected_iterations
    assert [
        event.status for event in frame.progress.turn_events if event.kind == "tool_result"
    ] == ["success"] * expected_iterations
    assert runtime.max_iterations == 2


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
        runtime_config=AgentRuntimeConfig(),
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
        runtime_config=AgentRuntimeConfig(),
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
        mtp_executor=MagicMock(),
        runtime_config=AgentRuntimeConfig(),
        loop_executor=MagicMock(),
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
        mtp_executor=MagicMock(),
        runtime_config=AgentRuntimeConfig(),
        loop_executor=MagicMock(),
    )
    frame = _frame()
    frame.add_harvested_alias("draft_partial_1234")
    products = runtime.finalize_frame(
        frame, FrameExecutionResult(status=FrameExecutionStatus.FAILED)
    )
    assert products.artifact_aliases == ()


@pytest.mark.parametrize(
    ("window", "max_tokens", "expected_limit"),
    [(10000, 2000, 7500), (None, 2000, 17500), (1000, 1000, -500)],
)
def test_resolved_model_budget_reserves_output_and_margin(window, max_tokens, expected_limit):
    """窗口缺省时回落配置；生成预算挤满窗口时保留非正上限供输入检查拒绝。"""
    resolver = MagicMock()
    resolver.resolve.return_value = (
        LLMConfig(model="provider/model", context_window=window, max_tokens=max_tokens),
        "显示名称",
    )
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=AgentRuntimeConfig(default_context_window=20000, safety_margin=500),
        loop_executor=MagicMock(),
        model_registry=resolver,
    )
    resolved = runtime.resolve_model(OMNI_DOLL_PROFILE, None)

    assert resolved.input_token_limit == expected_limit
    assert resolved.generation_options["model"] == "provider/model"
    assert "context_window" not in resolved.generation_options
    assert "input_token_limit" not in resolved.generation_options


@pytest.mark.asyncio
async def test_frame_reentry_keeps_resolved_model_and_budget():
    """挂起恢复不重新读取注册表，以免预算与已经组装的请求发生漂移。"""
    from hivememory.agent_runtime.model.client import GenerationResult

    resolver = MagicMock()
    resolver.resolve.return_value = (LLMConfig(model="model", max_tokens=100), "模型")
    worker = MagicMock(generate_async=AsyncMock(return_value=GenerationResult(text="完成")))
    runtime = AgentRuntime(
        mtp_executor=MagicMock(),
        runtime_config=AgentRuntimeConfig(),
        loop_executor=AgentLoopExecutor(worker, MagicMock(), MagicMock(max_loop_iterations=3)),
        model_registry=resolver,
    )
    frame = _frame()
    frame.resolved_model = runtime.resolve_model(frame.agent_profile, None)
    resolver.resolve.side_effect = ModelNotFoundError("模型已被移除")

    result = await runtime.run_frame(frame, output_sink=NullFrameOutputSink())

    assert result.status == FrameExecutionStatus.COMPLETED
    assert frame.progress.text_segments == ["完成"]
    assert frame.progress.model_used == "模型"
