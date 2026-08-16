"""
LoopExecutor TurnEvent 采集单测

验证 Phase 1 新增的结构化事件采集行为:
1. 自然停止 -> 1 个 assistant_message 事件
2. 单次 MTP -> prefix + tool_call + tool_result，sequence 递增
3. CALL 路径 -> 编排侧产出 kind=tool_result tool_kind=CALL 事件
4. 无 MTP 时 frame.progress 正常，final_text 正确
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.execution.loop import AgentLoopExecutor
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionStatus,
    GenerationResult,
    MTPExecutionContext,
    RuntimeScope,
)
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.frame_factory import FrameFactory
from hivememory.alice.orchestration.run_executor import RunExecutor
from hivememory.alice.orchestration.run_session import RunSession
from hivememory.alice.orchestration.sub_agent import CallContextProvider, CallCoordinator
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Identity,
    TurnEvent,
)
from hivememory.core.mtp import MTP_RIGHT_DELIMITER, MTPCallRequest
from hivememory.core.protocol.models import MTPExecutionResult


def _natural_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        was_mtp_interrupted=False,
        prefix_text="",
        mtp_fragment="",
    )


def _mtp_result(prefix: str, mtp_text: str) -> GenerationResult:
    text = (
        mtp_text if mtp_text.endswith(MTP_RIGHT_DELIMITER) else f"{mtp_text} {MTP_RIGHT_DELIMITER}"
    )
    return GenerationResult(
        text=text,
        was_mtp_interrupted=True,
        prefix_text=prefix,
        mtp_fragment=text,
    )


def _mtp_exec_result(verb: str, status: str = "success") -> MTPExecutionResult:
    cmd = MagicMock()
    cmd.verb = MagicMock()
    cmd.verb.value = verb
    cmd.target = MagicMock()
    cmd.target.is_wildcard = False
    cmd.target.aliases = ["alias_x"]
    cmd.args = {}
    cmd.raw_text = f"<< {verb} | alias_x >>"

    return MTPExecutionResult(
        command=cmd,
        response_status=status,
        response_content="",
        formatted_response=f'<mtp_response status="{status}">{verb} result</mtp_response>',
        success=(status == "success"),
        execution_time_ms=1.0,
    )


def _call_mtp_exec_result() -> MTPExecutionResult:
    cmd = MagicMock()
    cmd.verb = MagicMock()
    cmd.verb.value = "CALL"
    cmd.target = MagicMock()
    cmd.target.is_wildcard = False
    cmd.target.aliases = ["sub_agent"]
    cmd.args = {}
    cmd.raw_text = '<< CALL | sub_agent | task="do work" >>'

    return MTPExecutionResult(
        command=cmd,
        response_status="suspend",
        response_content="",
        formatted_response="",
        success=True,
        execution_time_ms=1.0,
        call_request=MTPCallRequest(
            target_alias="sub_agent",
            task="do work",
            context_refs=[],
        ),
    )


def _make_frame() -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(
            run_id="run_test_1",
            frame_id="test_frame",
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic_1",
        identity=Identity(user_id="u1", agent_id="agent_a"),
    )


def _build_executor(generate_async_side_effect) -> tuple[AgentLoopExecutor, MagicMock]:
    kernel = MagicMock()
    kernel.config = MagicMock()
    kernel.config.agent_runtime = MagicMock(max_loop_iterations=10)
    mtp_executor = MagicMock()
    mtp_executor.intercept_and_execute = AsyncMock(return_value=None)

    worker_agent = MagicMock()
    worker_agent.generate_async = AsyncMock(side_effect=generate_async_side_effect)

    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=mtp_executor,
        config=kernel.config.agent_runtime,
    )
    return executor, kernel


@pytest.mark.asyncio
async def test_natural_stop_produces_one_assistant_message_event():
    """自然停止: 1 个 assistant_message 事件，sequence=0，role=assistant"""
    frame = _make_frame()
    executor, _kernel = _build_executor([_natural_result("Hello world")])

    engine_result = await executor.execute_frame(frame, max_iterations=5)

    assert engine_result.status == FrameExecutionStatus.COMPLETED
    assert "".join(frame.progress.text_segments) == "Hello world"
    assert len(frame.progress.turn_events) == 1

    ev: TurnEvent = frame.progress.turn_events[0]
    assert ev.kind == "assistant_message"
    assert ev.sequence == 0
    assert ev.role == "assistant"
    assert ev.content == "Hello world"
    assert ev.tool_kind is None


@pytest.mark.asyncio
async def test_iteration_limit_returns_budget_exhausted():
    frame = _make_frame()
    executor, _kernel = _build_executor([_mtp_result("", "<< READ | alias_x >>")])
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    engine_result = await executor.execute_frame(frame, max_iterations=1)

    assert engine_result.status == FrameExecutionStatus.BUDGET_EXHAUSTED
    assert frame.progress.iteration == 1


@pytest.mark.asyncio
async def test_missing_mtp_result_returns_failed():
    frame = _make_frame()
    executor, _kernel = _build_executor([_mtp_result("", "<< READ | alias_x >>")])

    engine_result = await executor.execute_frame(frame, max_iterations=2)

    assert engine_result.status == FrameExecutionStatus.FAILED
    assert isinstance(engine_result.error, RuntimeError)
    assert frame.progress.turn_events[-1].status == "failed"


@pytest.mark.asyncio
async def test_generation_failure_returns_failed_frame_outcome():
    frame = _make_frame()
    executor, _kernel = _build_executor([])
    error = RuntimeError("provider unavailable")
    executor.worker_agent.generate_async = AsyncMock(side_effect=error)

    result = await executor.execute_frame(frame, max_iterations=2)

    assert result.status == FrameExecutionStatus.FAILED
    assert result.error is error


@pytest.mark.asyncio
async def test_stream_generation_failure_returns_failed_frame_outcome():
    frame = _make_frame()
    executor, _kernel = _build_executor([])
    error = RuntimeError("stream unavailable")

    async def fail_stream(*_args, **_kwargs):
        if False:
            yield None
        raise error

    class StreamingSink:
        streams_tokens = True

        async def send(self, _output):
            return None

    executor.worker_agent.generate_stream = fail_stream

    result = await executor.execute_frame(
        frame,
        max_iterations=2,
        output_sink=StreamingSink(),
    )

    assert result.status == FrameExecutionStatus.FAILED
    assert result.error is error


@pytest.mark.asyncio
async def test_generation_cancelled_error_is_not_converted_to_failed_outcome():
    frame = _make_frame()
    executor, _kernel = _build_executor([])
    executor.worker_agent.generate_async = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_frame(frame, max_iterations=2)


@pytest.mark.asyncio
async def test_single_mtp_produces_four_events():
    """单次 MTP: prefix(assistant_message) + tool_call + tool_result + final(assistant_message)"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("查找中", "<< READ | alias_x >>"),
        _natural_result("找到了"),
    ]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5)

    assert "".join(frame.progress.text_segments) == "查找中找到了"
    events = frame.progress.turn_events
    assert len(events) == 4

    prefix_ev, cmd_ev, res_ev, final_ev = events
    assert prefix_ev.kind == "assistant_message"
    assert prefix_ev.sequence == 0
    assert prefix_ev.role == "assistant"
    assert prefix_ev.content == "查找中"

    assert cmd_ev.kind == "tool_call"
    assert cmd_ev.sequence == 1
    assert cmd_ev.role == "assistant"
    assert cmd_ev.tool_kind == "READ"
    assert cmd_ev.content.endswith(MTP_RIGHT_DELIMITER)

    assert res_ev.kind == "tool_result"
    assert res_ev.sequence == 2
    assert res_ev.role == "user"
    assert res_ev.tool_kind == "READ"
    assert res_ev.status == "success"
    assert res_ev.render_as == "system_tool_result"
    assert frame.working_history[-2]["content"] == cmd_ev.content

    assert final_ev.kind == "assistant_message"
    assert final_ev.sequence == 3
    assert final_ev.content == "找到了"


@pytest.mark.asyncio
async def test_mtp_execution_receives_frame_context():
    frame = _make_frame()
    gen_results = [_mtp_result("", "READ alias_x"), _natural_result("done")]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5)

    _, kwargs = executor._mtp_executor.intercept_and_execute.await_args
    context = kwargs["context"]
    assert isinstance(context, MTPExecutionContext)
    assert context.identity == frame.identity
    assert context.agent_profile is frame.agent_profile
    assert context.runtime_scope.run_id == frame.runtime_scope.run_id
    assert context.runtime_scope.frame_id == frame.runtime_scope.frame_id
    assert context.runtime_scope.action_id == "action_1_0"


@pytest.mark.asyncio
async def test_task_cancellation_during_generation_propagates_before_mtp():
    frame = _make_frame()
    started = asyncio.Event()

    async def slow_generate(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    executor, _kernel = _build_executor([])
    executor.worker_agent.generate_async = slow_generate

    task = asyncio.create_task(executor.execute_frame(frame, max_iterations=5))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    executor._mtp_executor.intercept_and_execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_cancellation_during_mtp_propagates():
    frame = _make_frame()
    gen_results = [_mtp_result("前缀", "<< READ | alias_x >>")]
    executor, _kernel = _build_executor(gen_results)
    started = asyncio.Event()

    async def slow_execute(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    executor._mtp_executor.intercept_and_execute = slow_execute

    task = asyncio.create_task(executor.execute_frame(frame, max_iterations=5))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert "".join(frame.progress.text_segments) == "前缀"
    assert [ev.kind for ev in frame.progress.turn_events] == ["assistant_message"]


@pytest.mark.asyncio
async def test_sequence_is_monotonically_increasing_across_iterations():
    """多次 MTP: sequence 单调递增"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("", '<< SEARCH | * | query="x" >>'),
        _mtp_result("", "<< READ | alias_y >>"),
        _natural_result("done"),
    ]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        side_effect=[
            _mtp_exec_result("SEARCH"),
            _mtp_exec_result("READ"),
        ]
    )

    await executor.execute_frame(frame, max_iterations=10)

    seqs = [ev.sequence for ev in frame.progress.turn_events]
    assert seqs == sorted(seqs), "sequence 必须单调递增"
    assert len(set(seqs)) == len(seqs), "sequence 不能重复"


@pytest.mark.asyncio
async def test_empty_prefix_text_not_recorded():
    """prefix_text 为空时，不生成 assistant_message 事件"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("", "<< READ | alias_x >>"),
        _natural_result("done"),
    ]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5)

    # prefix 为空：不产生前缀 assistant_message，只保留 tool_call/tool_result/最终回复
    kinds = [ev.kind for ev in frame.progress.turn_events]
    assert kinds == ["tool_call", "tool_result", "assistant_message"]
    assistant = [ev for ev in frame.progress.turn_events if ev.kind == "assistant_message"]
    assert len(assistant) == 1
    assert assistant[0].content == "done"


@pytest.mark.asyncio
async def test_call_path_produces_mtp_result_event_with_call_verb():
    """CALL 路径: 编排侧产出 kind=tool_result, tool_kind=CALL, role=user"""
    call_counter = {"n": 0}

    async def gen_async_side(*args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return _mtp_result("正在调用", '<< CALL | sub_agent | task="do work" >>')
        return _natural_result("完成")

    executor, _kernel = _build_executor([])
    worker_agent = MagicMock()
    worker_agent.generate_async = AsyncMock(side_effect=gen_async_side)
    executor.worker_agent = worker_agent
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_call_mtp_exec_result())

    profile_resolver = MagicMock()
    profile_resolver.resolve = AsyncMock(return_value=OMNI_DOLL_PROFILE)
    alias_resolver = MagicMock()

    agent_runtime = AgentRuntime(
        mtp_executor=MagicMock(), runtime_config=MagicMock(), loop_executor=executor
    )
    frame_factory = FrameFactory()
    prompt_assembler = MagicMock()
    prompt_assembler.build_sub_agent_messages.return_value = [
        {"role": "user", "content": "sub task"}
    ]
    coordinator = CallCoordinator(
        agent_runtime,
        CallContextProvider(profile_resolver, alias_resolver),
        frame_factory=frame_factory,
        prompt_assembler=prompt_assembler,
    )
    frame = _make_frame()
    session = RunSession(agent_run_id="run_test_1")
    session.register_root_frame(frame)
    executor = RunExecutor(
        agent_runtime,
        session=session,
        call_coordinator=coordinator,
    )

    executor_result = await executor.run(frame)

    call_events = [
        ev
        for ev in frame.progress.turn_events
        if ev.kind == "tool_result" and ev.tool_kind == "CALL"
    ]
    assert executor_result.status == FrameExecutionStatus.COMPLETED
    assert len(call_events) == 1, (
        f"应有 1 个 CALL tool_result 事件，实际: {frame.progress.turn_events}"
    )
    call_ev = call_events[0]
    assert call_ev.role == "user"
    assert call_ev.status == "success"
    assert call_ev.render_as == "system_call_response"
    assert call_ev.content.startswith("[System MTP Call Response]\n")


def test_chat_result_default_turn_events():
    """AgentRunResult 新字段有默认值"""
    from hivememory.core.protocol.models import AgentRunResult

    r = AgentRunResult(final_text="hi")
    assert r.turn_events == []


@pytest.mark.asyncio
async def test_run_command_event_carries_execution_status_for_reducer():
    """RUN 指令的 tool_call 事件应带上执行状态，避免 reducer 降级为 unknown"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("", '<< RUN | tool_x | cmd="echo hi" >>'),
        _natural_result("done"),
    ]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("RUN"))

    await executor.execute_frame(frame, max_iterations=5)

    run_commands = [
        ev for ev in frame.progress.turn_events if ev.kind == "tool_call" and ev.tool_kind == "RUN"
    ]
    assert len(run_commands) == 1
    assert run_commands[0].status == "success"
