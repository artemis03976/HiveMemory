"""
LoopExecutor TurnEvent 采集单测

验证 Phase 1 新增的结构化事件采集行为:
1. 自然停止 -> 1 个 assistant_message 事件
2. 单次 MTP -> prefix + tool_call + tool_result，sequence 递增
3. CALL 路径 -> 编排侧产出 kind=tool_result tool_kind=CALL 事件
4. 无 MTP 时 frame.progress 正常，final_text 正确
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import asyncio
import pytest

from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionStatus,
    GenerationResult,
    MTPExecutionContext,
    RuntimeScope,
)
from hivememory.agent_runtime.resolver import ResolveResult
from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    OMNI_DOLL_PROFILE,
    PayloadLayer,
    TurnEvent,
)
from hivememory.core.protocol.models import MTPExecutionResult
from hivememory.core.mtp import MTPCallRequest, MTP_RIGHT_DELIMITER


def _natural_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        was_mtp_interrupted=False,
        prefix_text="",
        mtp_fragment="",
    )


def _mtp_result(prefix: str, mtp_text: str) -> GenerationResult:
    text = mtp_text if mtp_text.endswith(MTP_RIGHT_DELIMITER) else f"{mtp_text} {MTP_RIGHT_DELIMITER}"
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


def _make_frame(depth: int = 0) -> ExecutionFrame:
    return ExecutionFrame(
        runtime_scope=RuntimeScope(
            run_id="run_test_1",
            frame_id="test_frame",
            depth=depth,
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic_1",
        identity=Identity(user_id="u1", agent_id="agent_a"),
    )


def _make_context_atom(title: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        index=IndexLayer(
            title=title,
            summary=f"{title} summary",
            memory_type=MemoryType.FACT,
            tags=["context"],
        ),
        payload=PayloadLayer(content=content),
        meta=MetaData(
            source_agent_id="test",
            user_id="u1",
            updated_at=datetime.now(),
            confidence_score=0.9,
        ),
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
async def test_natural_stop_no_prefix_no_extra_events():
    """没有 MTP 的情况下，turn_events 只有一个事件"""
    frame = _make_frame()
    executor, _kernel = _build_executor([_natural_result("Simple reply")])

    await executor.execute_frame(frame, max_iterations=5)

    assert len(frame.progress.turn_events) == 1
    assert frame.progress.turn_events[0].kind == "assistant_message"


@pytest.mark.asyncio
async def test_single_mtp_produces_four_events():
    """单次 MTP: prefix(assistant_message) + tool_call + tool_result + final(assistant_message)"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("查找中", "<< READ | alias_x >>"),
        _natural_result("找到了"),
    ]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=_mtp_exec_result("READ")
    )

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
    frame = _make_frame(depth=1)
    gen_results = [_mtp_result("", "READ alias_x"), _natural_result("done")]
    executor, _kernel = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=_mtp_exec_result("READ")
    )

    await executor.execute_frame(frame, max_iterations=5)

    _, kwargs = executor._mtp_executor.intercept_and_execute.await_args
    context = kwargs["context"]
    assert isinstance(context, MTPExecutionContext)
    assert context.identity == frame.identity
    assert context.agent_profile is frame.agent_profile
    assert context.runtime_scope.run_id == frame.runtime_scope.run_id
    assert context.runtime_scope.frame_id == frame.runtime_scope.frame_id
    assert context.runtime_scope.depth == frame.runtime_scope.depth
    assert context.runtime_scope.action_id == "action_1_0"


@pytest.mark.asyncio
async def test_cancel_before_mtp_execution_skips_executor():
    frame = _make_frame()
    cancel_event = asyncio.Event()

    async def generate_and_cancel(*args, **kwargs):
        cancel_event.set()
        return _mtp_result("前缀", "<< READ | alias_x >>")

    executor, _kernel = _build_executor([])
    executor.worker_agent.generate_async = AsyncMock(side_effect=generate_and_cancel)

    await executor.execute_frame(frame, max_iterations=5, cancel_event=cancel_event)

    executor._mtp_executor.intercept_and_execute.assert_not_awaited()
    assert "".join(frame.progress.text_segments) == "前缀"
    assert [ev.kind for ev in frame.progress.turn_events] == ["assistant_message"]


@pytest.mark.asyncio
async def test_cancel_after_mtp_execution_skips_result_processing():
    frame = _make_frame()
    gen_results = [_mtp_result("前缀", "<< READ | alias_x >>")]
    executor, _kernel = _build_executor(gen_results)
    cancel_event = asyncio.Event()

    async def execute_and_cancel(*args, **kwargs):
        cancel_event.set()
        return _mtp_exec_result("READ")

    executor._mtp_executor.intercept_and_execute = AsyncMock(side_effect=execute_and_cancel)

    await executor.execute_frame(frame, max_iterations=5, cancel_event=cancel_event)

    executor._mtp_executor.intercept_and_execute.assert_awaited_once()
    assert "".join(frame.progress.text_segments) == "前缀"
    assert [ev.kind for ev in frame.progress.turn_events] == ["assistant_message"]
    assert len(frame.working_history) == 1


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
    executor._mtp_executor.intercept_and_execute = AsyncMock(side_effect=[
        _mtp_exec_result("SEARCH"),
        _mtp_exec_result("READ"),
    ])

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
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=_mtp_exec_result("READ")
    )

    await executor.execute_frame(frame, max_iterations=5)

    kinds = [ev.kind for ev in frame.progress.turn_events]
    assert "assistant_message" not in kinds or all(
        ev.content != "" for ev in frame.progress.turn_events if ev.kind == "assistant_message"
    )
    assert "tool_call" in kinds
    assert "tool_result" in kinds


@pytest.mark.asyncio
async def test_call_path_produces_mtp_result_event_with_call_verb():
    """CALL 路径: 编排侧产出 kind=tool_result, tool_kind=CALL, role=user"""
    from hivememory.alice.runtime.orchestrator import AgentOrchestrator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime

    main_frame = _make_frame(depth=0)
    sub_frame = ExecutionFrame(
        runtime_scope=main_frame.runtime_scope.for_child("sub_frame"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "sub task"}],
        topic_id=None,
        identity=Identity(user_id="u1", agent_id="sub_agent"),
    )

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
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=_call_mtp_exec_result()
    )

    frame_scheduler = MagicMock()
    frame_scheduler.suspend_frame = MagicMock()
    frame_scheduler.resume_frame = MagicMock()
    frame_scheduler.fork_sub_frame = AsyncMock(return_value=sub_frame)
    frame_scheduler.create_main_frame = MagicMock(return_value=main_frame)

    profile_resolver = MagicMock()
    profile_resolver.resolve = AsyncMock(return_value=OMNI_DOLL_PROFILE)
    alias_resolver = MagicMock()

    orchestrator = AgentOrchestrator(
        agent_runtime=AgentRuntime(
            mtp_executor=MagicMock(), config=MagicMock(), loop_executor=executor
        ),
        frame_scheduler=frame_scheduler,
        agent_profile_resolver=profile_resolver,
        alias_resolver=alias_resolver,
    )

    result = await orchestrator.run_agent(
        messages=[{"role": "user", "content": "hello"}],
        identity=Identity(user_id="u1"),
        topic_id="t1",
    )

    call_events = [
        ev for ev in result.turn_events
        if ev.kind == "tool_result" and ev.tool_kind == "CALL"
    ]
    assert len(call_events) == 1, f"应有 1 个 CALL tool_result 事件，实际: {result.turn_events}"
    call_ev = call_events[0]
    assert call_ev.role == "user"
    assert call_ev.status == "success"
    assert call_ev.render_as == "system_call_response"
    assert call_ev.content.startswith("[System MTP Call Response]\n")


@pytest.mark.asyncio
async def test_context_refs_fetch_uses_runtime_alias_resolver():
    from hivememory.alice.runtime.orchestrator import AgentOrchestrator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime

    executor, _kernel = _build_executor([])
    atom = _make_context_atom("Fact A", "ctx")
    resolved = ResolveResult(
        kind="atom",
        requested_alias="fact_a",
        atom=atom,
    )
    alias_resolver = MagicMock()
    alias_resolver.resolve = AsyncMock(return_value=resolved)

    orchestrator = AgentOrchestrator(
        agent_runtime=AgentRuntime(
            mtp_executor=MagicMock(), config=MagicMock(), loop_executor=executor
        ),
        frame_scheduler=MagicMock(),
        agent_profile_resolver=MagicMock(),
        alias_resolver=alias_resolver,
    )
    identity = Identity(user_id="u1", agent_id="agent_a")

    result = await orchestrator._fetch_context_refs_content(
        ["fact_a"],
        identity,
        language="en",
    )

    assert result.startswith("[Shared Context from Parent Agent]")
    assert "Use READ" in result
    assert '<memory alias="' in result
    assert "Fact A" in result
    assert "ctx" in result
    alias_resolver.resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_refs_fetch_renders_redirected_alias_as_canonical_atom():
    from hivememory.alice.runtime.orchestrator import AgentOrchestrator
    from hivememory.alice.runtime.agent.runtime import AgentRuntime

    executor, _kernel = _build_executor([])
    atom = _make_context_atom("Canonical Fact", "canonical ctx")
    resolved = ResolveResult(
        kind="redirect",
        requested_alias="draft_ctx_1234",
        canonical_alias="fact_canonical",
        atom=atom,
    )
    alias_resolver = MagicMock()
    alias_resolver.resolve = AsyncMock(return_value=resolved)

    orchestrator = AgentOrchestrator(
        agent_runtime=AgentRuntime(
            mtp_executor=MagicMock(), config=MagicMock(), loop_executor=executor
        ),
        frame_scheduler=MagicMock(),
        agent_profile_resolver=MagicMock(),
        alias_resolver=alias_resolver,
    )
    identity = Identity(user_id="u1", agent_id="agent_a")

    result = await orchestrator._fetch_context_refs_content(["draft_ctx_1234"], identity)

    assert result.startswith("[Shared Context from Parent Agent]")
    assert "Canonical Fact" in result
    assert "canonical ctx" in result
    assert "<memory alias=" in result
    alias_resolver.resolve.assert_awaited_once()


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
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=_mtp_exec_result("RUN")
    )

    await executor.execute_frame(frame, max_iterations=5)

    run_commands = [
        ev for ev in frame.progress.turn_events
        if ev.kind == "tool_call" and ev.tool_kind == "RUN"
    ]
    assert len(run_commands) == 1
    assert run_commands[0].status == "success"
