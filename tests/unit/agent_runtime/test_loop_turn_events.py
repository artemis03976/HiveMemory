"""
LoopExecutor TurnEvent 采集单测

验证结构化事件采集行为:
1. 自然停止 -> 1 个 assistant_message 事件
2. 单次 MTP -> 指令前正文、只含指令片段的 tool_call、tool_result，sequence 递增
3. 指令无法解析 -> 没有动词的 tool_call 与 error 结果
4. ACK -> 产出的 pending alias 只记在结果事件上
5. CALL 挂起只写 tool_call；回填由编排侧写出结果事件，不改写调用
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.agent_runtime.frame import (
    ExecutionFrame,
    FrameExecutionPolicy,
    FrameExecutionStatus,
)
from hivememory.agent_runtime.loop import AgentLoopExecutor
from hivememory.agent_runtime.model.budget import TurnContextBudgetExceededError
from hivememory.agent_runtime.model.generator import GenerationResult, StreamChunk
from hivememory.agent_runtime.model.resolution import ResolvedFrameModel
from hivememory.agent_runtime.mtp.context import MTPExecutionContext, MTPExecutionResult
from hivememory.agent_runtime.runtime import AgentRuntime
from hivememory.alice.orchestration.call import CallContextProvider, CallCoordinator
from hivememory.alice.orchestration.executor import RunExecutor
from hivememory.alice.orchestration.frames import FrameFactory
from hivememory.alice.orchestration.session import RunSession
from hivememory.config.alice import AgentRuntimeConfig, KoakumaConfig
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    TurnEvent,
)
from hivememory.core.mtp import MTP_RIGHT_DELIMITER, MTPCallRequest
from hivememory.workspace.contracts import DeriveCallThreadRequest
from tests.helpers.call import StubOperationSubmitter, make_callee
from tests.helpers.llm import DEFAULT_LLM_PARAMS
from tests.helpers.outputs import NullAgentRunOutput, NullFrameOutputSink
from tests.helpers.workspace import make_runtime_scope


def _natural_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        was_mtp_interrupted=False,
        prefix_text="",
        mtp_fragment="",
    )


def _mtp_result(prefix: str, mtp_text: str) -> GenerationResult:
    """与生成器一致：全文是指令前正文加补齐右定界符的指令片段。"""
    fragment = (
        mtp_text if mtp_text.endswith(MTP_RIGHT_DELIMITER) else f"{mtp_text} {MTP_RIGHT_DELIMITER}"
    )
    return GenerationResult(
        text=prefix + fragment,
        was_mtp_interrupted=True,
        prefix_text=prefix,
        mtp_fragment=fragment,
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
        execution_policy=FrameExecutionPolicy.from_verbs(
            {"SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"}
        ),
        submit_operation=StubOperationSubmitter(),
        runtime_scope=make_runtime_scope(
            run_id="run_test_1",
            frame_id="test_frame",
        ),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        topic_id="topic_1",
        language="zh",
        resolved_model=ResolvedFrameModel(DEFAULT_LLM_PARAMS, input_token_limit=65536),
    )


class _ScriptedGenerator:
    """按顺序把预置结果作为流的终态 chunk 交出；元素为异常时在拉取时抛出。

    预置结果用尽后交出 ``fallback``（未提供时显式失败）。
    """

    def __init__(self, results, *, fallback: GenerationResult | None = None) -> None:
        self._results = list(results)
        self._fallback = fallback
        self.calls = 0

    def generate(self, messages, params, *, stream: bool = False):
        assert stream, "帧生成总是流式"
        self.calls += 1
        if self._results:
            return self._chunks(self._results.pop(0))
        if self._fallback is None:
            raise AssertionError("_ScriptedGenerator 没有剩余的预置结果")
        return self._chunks(self._fallback)

    @staticmethod
    async def _chunks(result):
        if isinstance(result, BaseException):
            raise result
        yield StreamChunk(is_final=True, result=result)


def _build_executor(results, *, fallback: GenerationResult | None = None) -> AgentLoopExecutor:
    mtp_executor = MagicMock()
    mtp_executor.intercept_and_execute = AsyncMock(return_value=None)

    executor = AgentLoopExecutor(
        generator=_ScriptedGenerator(results, fallback=fallback),
        mtp_executor=mtp_executor,
        config=AgentRuntimeConfig(max_loop_iterations=10),
    )
    return executor


@pytest.mark.asyncio
async def test_natural_stop_produces_one_assistant_message_event():
    """自然停止: 1 个 assistant_message 事件，sequence=0，role=assistant"""
    frame = _make_frame()
    executor = _build_executor([_natural_result("Hello world")])

    engine_result = await executor.execute_frame(
        frame, max_iterations=5, output_sink=NullFrameOutputSink()
    )

    assert engine_result.status == FrameExecutionStatus.COMPLETED
    assert "".join(frame.progress.text_segments) == "Hello world"
    assert frame.progress.turn_events == [
        TurnEvent(kind="assistant_message", sequence=0, content="Hello world")
    ]


@pytest.mark.asyncio
async def test_iteration_limit_returns_budget_exhausted():
    frame = _make_frame()
    executor = _build_executor([_mtp_result("", "<< READ | alias_x >>")])
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    engine_result = await executor.execute_frame(
        frame, max_iterations=1, output_sink=NullFrameOutputSink()
    )

    assert engine_result.status == FrameExecutionStatus.BUDGET_EXHAUSTED
    assert frame.progress.iteration == 1


@pytest.mark.asyncio
async def test_tool_result_growth_stops_before_next_generation():
    """巨大工具结果保留在事件中，但不会再发送已经超预算的模型请求。"""
    frame = _make_frame()
    frame.resolved_model = ResolvedFrameModel(DEFAULT_LLM_PARAMS, input_token_limit=40)
    executor = _build_executor([_mtp_result("正在读取", "<< READ | alias_x >>")])
    tool_result = _mtp_exec_result("READ")
    tool_result.formatted_response = "x" * 400
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=tool_result)

    result = await executor.execute_frame(
        frame, max_iterations=3, output_sink=NullFrameOutputSink()
    )

    assert result.status == FrameExecutionStatus.FAILED
    assert isinstance(result.error, TurnContextBudgetExceededError)
    assert frame.progress.iteration == 1
    assert frame.progress.turn_events[-1].content == "x" * 400
    assert executor.generator.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "status"), [(5, "completed"), (4, "failed")])
async def test_context_guard_counts_message_overhead_and_allows_exact_budget(limit, status):
    """文本一 token 加消息开销，恰好等于上限时允许生成。"""
    frame = _make_frame()
    frame.resolved_model = ResolvedFrameModel(DEFAULT_LLM_PARAMS, input_token_limit=limit)
    executor = _build_executor([_natural_result("完成")])

    result = await executor.execute_frame(
        frame, max_iterations=2, output_sink=NullFrameOutputSink()
    )

    assert result.status == status
    assert frame.progress.text_segments == (["完成"] if status == "completed" else [])


@pytest.mark.asyncio
async def test_missing_mtp_result_fails_explicitly_without_tool_events():
    """执行器对中断于指令的文本没有给出结果时显式失败，不补写失败的调用与结果。"""
    frame = _make_frame()
    executor = _build_executor([_mtp_result("", "<< READ | alias_x >>")])

    with pytest.raises(RuntimeError, match="no result"):
        await executor.execute_frame(frame, max_iterations=2, output_sink=NullFrameOutputSink())

    assert frame.progress.turn_events == []


@pytest.mark.asyncio
async def test_stream_generation_failure_returns_failed_frame_outcome():
    frame = _make_frame()
    executor = _build_executor([])
    error = RuntimeError("stream unavailable")

    async def fail_stream(*_args, **_kwargs):
        if False:
            yield None
        raise error

    class StreamingSink:

        async def send(self, _output):
            return None

    executor.generator.generate = fail_stream

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
    executor = _build_executor([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_frame(frame, max_iterations=2, output_sink=NullFrameOutputSink())


@pytest.mark.asyncio
async def test_single_mtp_produces_four_events():
    """单次 MTP: 指令前正文 + 只含指令片段的调用 + 结果 + 最终回复，working history 不变。"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("查找中", "<< READ | alias_x >>"),
        _natural_result("找到了"),
    ]
    executor = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())

    assert "".join(frame.progress.text_segments) == "查找中找到了"
    fragment = f"<< READ | alias_x >> {MTP_RIGHT_DELIMITER}"
    assert frame.progress.turn_events == [
        TurnEvent(kind="assistant_message", sequence=0, content="查找中"),
        TurnEvent(
            kind="tool_call",
            sequence=1,
            content=fragment,
            action_id="action_1_1",
            verb="READ",
            targets=("alias_x",),
        ),
        TurnEvent(
            kind="tool_result",
            sequence=2,
            content='<mtp_response status="success">READ result</mtp_response>',
            action_id="action_1_1",
            status="success",
        ),
        TurnEvent(kind="assistant_message", sequence=3, content="找到了"),
    ]
    # 运行时的助手消息仍是指令前正文加指令片段的全文。
    assert frame.working_history[-2] == {"role": "assistant", "content": f"查找中{fragment}"}


@pytest.mark.asyncio
async def test_unparseable_mtp_records_call_without_verb_and_error_result():
    """指令无法解析时调用事件只有原文，动词与作用对象为空，随后是 error 结果。"""
    frame = _make_frame()
    executor = _build_executor(
        [_mtp_result("", "<< READ alias_x >>"), _natural_result("重试后完成")]
    )
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        return_value=MTPExecutionResult(
            command=None,
            response_status="error",
            formatted_response='<mtp_response status="error">parse error</mtp_response>',
        )
    )

    await executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())

    call, result = frame.progress.turn_events[:2]
    assert call == TurnEvent(
        kind="tool_call",
        sequence=0,
        content=f"<< READ alias_x >> {MTP_RIGHT_DELIMITER}",
        action_id="action_1_0",
    )
    assert (result.kind, result.action_id, result.status) == ("tool_result", "action_1_0", "error")


@pytest.mark.asyncio
async def test_mtp_execution_receives_frame_context():
    frame = _make_frame()
    # 帧语言与 Profile 语言不同，MTP 上下文只能取帧语言。
    frame.language = "en"
    gen_results = [_mtp_result("", "READ alias_x"), _natural_result("done")]
    executor = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())

    _, kwargs = executor._mtp_executor.intercept_and_execute.await_args
    context = kwargs["context"]
    assert isinstance(context, MTPExecutionContext)
    assert context.runtime_scope.labels == frame.runtime_scope.labels
    assert context.language == "en"
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
        yield StreamChunk()

    executor = _build_executor([])
    executor.generator.generate = slow_generate

    task = asyncio.create_task(
        executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    executor._mtp_executor.intercept_and_execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_cancellation_during_mtp_propagates():
    frame = _make_frame()
    gen_results = [_mtp_result("前缀", "<< READ | alias_x >>")]
    executor = _build_executor(gen_results)
    started = asyncio.Event()

    async def slow_execute(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    executor._mtp_executor.intercept_and_execute = slow_execute

    task = asyncio.create_task(
        executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())
    )
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
    executor = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(
        side_effect=[
            _mtp_exec_result("SEARCH"),
            _mtp_exec_result("READ"),
        ]
    )

    await executor.execute_frame(frame, max_iterations=10, output_sink=NullFrameOutputSink())

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
    executor = _build_executor(gen_results)
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_mtp_exec_result("READ"))

    await executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())

    # prefix 为空：不产生前缀 assistant_message，只保留 tool_call/tool_result/最终回复
    kinds = [ev.kind for ev in frame.progress.turn_events]
    assert kinds == ["tool_call", "tool_result", "assistant_message"]
    assistant = [ev for ev in frame.progress.turn_events if ev.kind == "assistant_message"]
    assert len(assistant) == 1
    assert assistant[0].content == "done"


@pytest.mark.asyncio
async def test_call_suspension_records_only_the_call_event():
    """CALL 挂起时只写调用事件：结局要等回填，挂起中的调用没有结果事件。"""
    frame = _make_frame()
    executor = _build_executor([_mtp_result("正在调用", '<< CALL | sub_agent | task="do work" >>')])
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_call_mtp_exec_result())

    result = await executor.execute_frame(
        frame, max_iterations=5, output_sink=NullFrameOutputSink()
    )

    assert result.status == FrameExecutionStatus.SUSPENDED
    assert [ev.kind for ev in frame.progress.turn_events] == ["assistant_message", "tool_call"]
    call = frame.progress.turn_events[-1]
    assert (call.verb, call.targets, call.action_id) == ("CALL", ("sub_agent",), "action_1_1")


@pytest.mark.asyncio
async def test_call_path_produces_mtp_result_event_with_call_verb():
    """CALL 回填: 编排侧写出配对的结果事件，状态为子帧结局，调用事件保持原样。"""
    executor = _build_executor(
        [_mtp_result("正在调用", '<< CALL | sub_agent | task="do work" >>')],
        fallback=_natural_result("完成"),
    )
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=_call_mtp_exec_result())

    async def submit_operation(request):
        """仅替代能力提交端口；事件采集与 CALL 编排仍执行生产实现。"""
        if isinstance(request, DeriveCallThreadRequest):
            return make_callee(OMNI_DOLL_PROFILE.model_copy(deep=True))
        raise AssertionError(f"本场景不应提交其他资源操作: {request}")

    agent_runtime = AgentRuntime(
        runtime_config=AgentRuntimeConfig(),
        loop_executor=executor,
    )
    frame_factory = FrameFactory()
    coordinator = CallCoordinator(
        agent_runtime,
        CallContextProvider(),
        frame_factory=frame_factory,
        koakuma_config=KoakumaConfig(enabled=False),
    )
    frame = _make_frame()
    frame.submit_operation = StubOperationSubmitter(derive=submit_operation)
    session = RunSession(agent_run_id="run_test_1", process_id="process-1")
    session.register_root_frame(frame)
    executor = RunExecutor(
        agent_runtime,
        session=session,
        call_coordinator=coordinator,
    )

    # 未注入模型注册表：被调用帧的模型由运行的生成参数直接给出。
    executor_result = await executor.run(
        frame, generation_options={"model": "test-model"}, run_output=NullAgentRunOutput()
    )

    assert executor_result.status == FrameExecutionStatus.COMPLETED
    events = frame.progress.turn_events
    assert [ev.kind for ev in events] == [
        "assistant_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    call, result = events[1], events[2]
    assert (call.verb, call.targets, call.status) == ("CALL", ("sub_agent",), None)
    assert (result.action_id, result.status) == (call.action_id, "success")
    assert result.content.startswith("[System MTP Call Response]\n")


@pytest.mark.asyncio
async def test_ack_records_pending_alias_only_on_result_event():
    """ACK 的 pending alias 记在结果事件的产出上；调用事件不带结局。"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("", '<< WRITE | * | content="记住" >>'),
        _natural_result("done"),
    ]
    executor = _build_executor(gen_results)
    ack = _mtp_exec_result("WRITE", status="ack")
    ack.pending_alias = "draft_note_1234"
    executor._mtp_executor.intercept_and_execute = AsyncMock(return_value=ack)

    await executor.execute_frame(frame, max_iterations=5, output_sink=NullFrameOutputSink())

    call, result = frame.progress.turn_events[:2]
    assert (call.kind, call.status, call.produced_aliases) == ("tool_call", None, ())
    assert (result.status, result.produced_aliases) == ("ack", ("draft_note_1234",))
    assert frame.harvested_aliases == ["draft_note_1234"]
