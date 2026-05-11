"""
LoopExecutor TurnEvent 采集单测

验证 Phase 1 新增的结构化事件采集行为:
1. 自然停止 → 1 个 assistant_text 事件
2. 单次 MTP → prefix + mtp_command + mtp_result，sequence 递增
3. CALL 路径 → 父 frame 只有 kind=mtp_result verb=CALL 事件，无子 frame 事件
4. 无 MTP 时 ChatResult.turn_events 正常，final_text 正确
"""

import json
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
from hivememory.engines.perception.models import TurnEvent
from hivememory.patchouli.kernel.runtime.execution_frame import ExecutionFrame
from hivememory.patchouli.kernel.runtime.loop_executor import KernelLoopExecutor
from hivememory.patchouli.protocol.models import MTPExecutionResult
from hivememory.patchouli.worker_agent import GenerationResult


def _natural_result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        was_mtp_interrupted=False,
        prefix_text="",
        mtp_fragment="",
    )


def _mtp_result(prefix: str, mtp_text: str) -> GenerationResult:
    return GenerationResult(
        text=mtp_text,
        was_mtp_interrupted=True,
        prefix_text=prefix,
        mtp_fragment=mtp_text,
    )


def _mtp_exec_result(verb: str, status: str = "success") -> MTPExecutionResult:
    cmd = MagicMock()
    cmd.verb = MagicMock()
    cmd.verb.value = verb
    cmd.target = MagicMock()
    cmd.target.is_wildcard = False
    cmd.target.aliases = ["alias_x"]
    cmd.args = {}
    cmd.raw_text = f"⟪ {verb} | alias_x ⟫"

    return MTPExecutionResult(
        command=cmd,
        response_status=status,
        response_content="",
        formatted_response=f"<mtp_response status=\"{status}\">{verb} result</mtp_response>",
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
    cmd.raw_text = '⟪ CALL | sub_agent | task="do work" ⟫'

    return MTPExecutionResult(
        command=cmd,
        response_status="suspend",
        response_content=json.dumps({
            "target_alias": "sub_agent",
            "task": "do work",
            "context_refs": [],
        }),
        formatted_response="",
        success=True,
        execution_time_ms=1.0,
    )


def _make_frame(depth: int = 0) -> ExecutionFrame:
    return ExecutionFrame(
        process_id="test_pid",
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "hello"}],
        depth=depth,
        topic_id="topic_1",
        identity=Identity(user_id="u1", agent_id="agent_a"),
    )


def _build_executor(generate_async_side_effect) -> KernelLoopExecutor:
    kernel = MagicMock()
    kernel.koakuma = MagicMock()
    kernel.koakuma._current_traces = []
    kernel.koakuma.atom_cache = MagicMock()
    kernel.koakuma.atom_cache.get_atom_by_alias = MagicMock(return_value=None)
    kernel.config = MagicMock()
    kernel.config.koakuma.max_recursion_depth = 10

    worker_agent = MagicMock()
    worker_agent.generate_async = AsyncMock(side_effect=generate_async_side_effect)

    executor = KernelLoopExecutor(kernel=kernel, worker_agent=worker_agent)
    return executor, kernel


# ============ 自然停止场景 ============

@pytest.mark.asyncio
async def test_natural_stop_produces_one_assistant_text_event():
    """自然停止: 1 个 assistant_text 事件，sequence=0，role=assistant"""
    frame = _make_frame()
    executor, kernel = _build_executor([_natural_result("Hello world")])
    kernel.handle_mtp = AsyncMock(return_value=None)

    result = await executor.execute_frame(frame, max_iterations=5)

    assert result.final_text == "Hello world"
    assert len(result.turn_events) == 1

    ev: TurnEvent = result.turn_events[0]
    assert ev.kind == "assistant_text"
    assert ev.sequence == 0
    assert ev.role == "assistant"
    assert ev.content == "Hello world"
    assert ev.verb is None


@pytest.mark.asyncio
async def test_natural_stop_no_prefix_no_extra_events():
    """没有 MTP 的情况下，turn_events 只有一个事件"""
    frame = _make_frame()
    executor, kernel = _build_executor([_natural_result("Simple reply")])
    kernel.handle_mtp = AsyncMock(return_value=None)

    result = await executor.execute_frame(frame, max_iterations=5)

    assert len(result.turn_events) == 1
    assert result.turn_events[0].kind == "assistant_text"


# ============ 单次 MTP 场景 ============

@pytest.mark.asyncio
async def test_single_mtp_produces_four_events():
    """单次 MTP: prefix(assistant_text) + mtp_command + mtp_result + final(assistant_text)"""
    frame = _make_frame()

    gen_results = [
        _mtp_result("查找中", "⟪ READ | alias_x ⟫"),
        _natural_result("找到了"),
    ]
    executor, kernel = _build_executor(gen_results)
    kernel.handle_mtp = AsyncMock(return_value=_mtp_exec_result("READ"))

    result = await executor.execute_frame(frame, max_iterations=5)

    assert result.final_text == "查找中找到了"
    events = result.turn_events
    # prefix + mtp_command + mtp_result + final natural text
    assert len(events) == 4

    prefix_ev, cmd_ev, res_ev, final_ev = events
    assert prefix_ev.kind == "assistant_text"
    assert prefix_ev.sequence == 0
    assert prefix_ev.role == "assistant"
    assert prefix_ev.content == "查找中"

    assert cmd_ev.kind == "mtp_command"
    assert cmd_ev.sequence == 1
    assert cmd_ev.role == "assistant"
    assert cmd_ev.verb == "READ"

    assert res_ev.kind == "mtp_result"
    assert res_ev.sequence == 2
    assert res_ev.role == "user"
    assert res_ev.verb == "READ"
    assert res_ev.status == "success"
    assert res_ev.render_as == "system_mtp_result"

    assert final_ev.kind == "assistant_text"
    assert final_ev.sequence == 3
    assert final_ev.content == "找到了"


@pytest.mark.asyncio
async def test_sequence_is_monotonically_increasing_across_iterations():
    """多次 MTP: sequence 单调递增"""
    frame = _make_frame()

    gen_results = [
        _mtp_result("", "⟪ SEARCH | * | query=\"x\" ⟫"),
        _mtp_result("", "⟪ READ | alias_y ⟫"),
        _natural_result("done"),
    ]
    executor, kernel = _build_executor(gen_results)
    kernel.handle_mtp = AsyncMock(side_effect=[
        _mtp_exec_result("SEARCH"),
        _mtp_exec_result("READ"),
    ])

    result = await executor.execute_frame(frame, max_iterations=10)

    seqs = [ev.sequence for ev in result.turn_events]
    assert seqs == sorted(seqs), "sequence 必须单调递增"
    assert len(set(seqs)) == len(seqs), "sequence 不能重复"


@pytest.mark.asyncio
async def test_empty_prefix_text_not_recorded():
    """prefix_text 为空时，不生成 assistant_text 事件"""
    frame = _make_frame()

    gen_results = [
        _mtp_result("", "⟪ READ | alias_x ⟫"),  # empty prefix
        _natural_result("done"),
    ]
    executor, kernel = _build_executor(gen_results)
    kernel.handle_mtp = AsyncMock(return_value=_mtp_exec_result("READ"))

    result = await executor.execute_frame(frame, max_iterations=5)

    kinds = [ev.kind for ev in result.turn_events]
    # 不应有来自空 prefix 的 assistant_text
    assert "assistant_text" not in kinds or all(
        ev.content != "" for ev in result.turn_events if ev.kind == "assistant_text"
    )
    assert "mtp_command" in kinds
    assert "mtp_result" in kinds


# ============ CALL 路径场景 ============

@pytest.mark.asyncio
async def test_call_path_produces_mtp_result_event_with_call_verb():
    """CALL 路径: 父 frame 产出 kind=mtp_result, verb=CALL, role=user"""
    main_frame = _make_frame(depth=0)
    sub_frame = ExecutionFrame(
        process_id="sub_pid",
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "sub task"}],
        depth=1,
        topic_id=None,
        parent_frame_id=main_frame.process_id,
        identity=Identity(user_id="u1", agent_id="sub_agent"),
    )

    # 主帧: CALL → 自然停止
    call_counter = {"n": 0}

    async def gen_async_side(*args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            # 主帧第一次生成: CALL 触发
            return _mtp_result("正在调用", '⟪ CALL | sub_agent | task="do work" ⟫')
        else:
            # 子帧或主帧第二次: 自然停止
            return _natural_result("完成")

    executor, kernel = _build_executor([])
    worker_agent = MagicMock()
    worker_agent.generate_async = AsyncMock(side_effect=gen_async_side)
    executor.worker_agent = worker_agent

    kernel.handle_mtp = AsyncMock(return_value=_call_mtp_exec_result())

    # frame_scheduler mock
    kernel.frame_scheduler = MagicMock()
    kernel.frame_scheduler.suspend_frame = MagicMock()
    kernel.frame_scheduler.resume_frame = MagicMock()
    kernel.frame_scheduler.fork_sub_frame = AsyncMock(return_value=sub_frame)

    result = await executor.execute_frame(main_frame, max_iterations=5)

    call_events = [ev for ev in result.turn_events if ev.kind == "mtp_result" and ev.verb == "CALL"]
    assert len(call_events) == 1, f"应有 1 个 CALL mtp_result 事件，实际: {result.turn_events}"
    call_ev = call_events[0]
    assert call_ev.role == "user"
    assert call_ev.status == "success"
    assert call_ev.render_as == "system_ipc_return"

    # 子帧自己的事件不应污染主帧
    sub_kinds = [ev.kind for ev in result.turn_events if ev.verb not in ("CALL", None)]
    # 所有 verb 为 CALL 的来自主帧，子帧事件不透传
    assert all(ev.verb in (None, "CALL") for ev in result.turn_events if ev.kind == "mtp_result")


# ============ turn_events 在 ChatResult 中的默认值 ============

def test_chat_result_default_turn_events():
    """ChatResult 新字段有默认值，不破坏现有代码"""
    from hivememory.patchouli.protocol.models import ChatResult
    r = ChatResult(final_text="hi")
    assert r.turn_events == []


@pytest.mark.asyncio
async def test_run_command_event_carries_execution_status_for_reducer():
    """RUN 指令的 mtp_command 事件应带上执行状态，避免 reducer 降级为 unknown"""
    frame = _make_frame()
    gen_results = [
        _mtp_result("", "⟪ RUN | tool_x | cmd=\"echo hi\" ⟫"),
        _natural_result("done"),
    ]
    executor, kernel = _build_executor(gen_results)
    kernel.handle_mtp = AsyncMock(return_value=_mtp_exec_result("RUN"))

    result = await executor.execute_frame(frame, max_iterations=5)

    run_commands = [ev for ev in result.turn_events if ev.kind == "mtp_command" and ev.verb == "RUN"]
    assert len(run_commands) == 1
    assert run_commands[0].status == "success"
