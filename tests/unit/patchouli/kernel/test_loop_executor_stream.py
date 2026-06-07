"""
AgentLoopExecutor 流式事件测试

聚焦子代理调用与 CALL response 相关的流式事件链路：
    1. CALL suspend 后主/子帧事件命名空间 (scope) 正确
    2. 子帧失败时仍产出 sub_agent_end(error) 且主循环可继续完成
"""

from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
from hivememory.agent_runtime.models import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    GenerationResult,
    RuntimeScope,
    StreamChunk,
)
from hivememory.agent_runtime.loop_executor import AgentLoopExecutor
from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.orchestrator import AgentOrchestrator
from hivememory.core.mtp import MTPCallRequest
from hivememory.core.protocol.models import MTPExecutionResult


def _make_call_mtp_result() -> MTPExecutionResult:
    cmd = MagicMock()
    cmd.verb = MagicMock()
    cmd.verb.value = "CALL"
    cmd.target = MagicMock()
    cmd.target.is_wildcard = False
    cmd.target.aliases = ["coder_doll"]
    cmd.args = {"task": "帮我处理子任务"}
    cmd.raw_text = '⟪ CALL | coder_doll | task="帮我处理子任务" ⟫'

    return MTPExecutionResult(
        command=cmd,
        response_status="suspend",
        response_content="",
        formatted_response="",
        success=True,
        execution_time_ms=1.0,
        call_request=MTPCallRequest(
            target_alias="coder_doll",
            task="帮我处理子任务",
            context_refs=[],
        ),
    )


def _make_frames():
    main_frame = ExecutionFrame(
        runtime_scope=RuntimeScope(frame_id="frame_main_test", depth=0),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "主任务"}],
        topic_id="topic_1",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
    )
    sub_frame = ExecutionFrame(
        runtime_scope=main_frame.runtime_scope.for_child("frame_sub_test"),
        agent_profile=OMNI_DOLL_PROFILE,
        working_history=[{"role": "user", "content": "子任务"}],
        topic_id=None,
        identity=Identity(user_id="u1", agent_id="coder_doll"),
    )
    return main_frame, sub_frame


def _build_orchestrator(worker_stream_impl, main_frame, sub_frame):
    """Build an AgentOrchestrator wired with a streaming worker and mock scheduler."""
    config = MagicMock(max_loop_iterations=10)
    mtp_executor = MagicMock()
    mtp_executor.intercept_and_execute = AsyncMock(return_value=_make_call_mtp_result())

    worker_agent = MagicMock()
    worker_agent.generate_stream = worker_stream_impl

    executor = AgentLoopExecutor(
        worker_agent=worker_agent,
        mtp_executor=mtp_executor,
        config=config,
    )

    agent_runtime = AgentRuntime(
        mtp_executor=mtp_executor,
        config=MagicMock(),
        loop_executor=executor,
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
        agent_runtime=agent_runtime,
        frame_scheduler=frame_scheduler,
        agent_profile_resolver=profile_resolver,
        alias_resolver=alias_resolver,
    )
    return orchestrator


@pytest.mark.asyncio
async def test_execute_frame_stream_emits_scoped_events_for_call():
    """CALL 场景下，token/mtp 事件通过 scope 区分主/子帧。"""
    main_frame, sub_frame = _make_frames()

    call_counter = {"n": 0}

    async def fake_generate_stream(_messages, **_kwargs):
        call_counter["n"] += 1
        n = call_counter["n"]
        if n == 1:
            yield StreamChunk(delta="主帧前缀", full_text="主帧前缀")
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text='主帧前缀⟪ CALL | coder_doll | task="帮我处理子任务" ⟫',
                    finish_reason="stop",
                    was_mtp_interrupted=True,
                    prefix_text="主帧前缀",
                    mtp_fragment='⟪ CALL | coder_doll | task="帮我处理子任务" ⟫',
                ),
            )
            return
        if n == 2:
            yield StreamChunk(delta="子帧输出", full_text="子帧输出")
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text="子帧输出完成",
                    finish_reason="stop",
                    was_mtp_interrupted=False,
                    prefix_text="子帧输出完成",
                    mtp_fragment="",
                ),
            )
            return
        if n == 3:
            yield StreamChunk(delta="主帧收尾", full_text="主帧收尾")
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text="主帧收尾完成",
                    finish_reason="stop",
                    was_mtp_interrupted=False,
                    prefix_text="主帧收尾完成",
                    mtp_fragment="",
                ),
            )
            return
        raise AssertionError("generate_stream 被调用超过预期次数")

    orchestrator = _build_orchestrator(fake_generate_stream, main_frame, sub_frame)

    events: List[Dict[str, Any]] = []
    async for event in orchestrator.run_agent_stream(
        messages=[{"role": "user", "content": "主任务"}],
        identity=Identity(user_id="u1"),
        topic_id="topic_1",
    ):
        events.append(event)

    event_types = [e["event"] for e in events]
    assert "sub_agent_start" in event_types
    assert "sub_agent_end" in event_types
    assert "sub_token" not in event_types
    assert "sub_mtp_start" not in event_types
    assert "sub_mtp_result" not in event_types

    main_tokens = [
        e for e in events
        if e["event"] == "token" and e["data"].get("scope") == "main"
    ]
    sub_tokens = [
        e for e in events
        if e["event"] == "token" and e["data"].get("scope") == "sub"
    ]
    assert any("主帧前缀" in e["data"]["content"] for e in main_tokens)
    assert any("主帧收尾" in e["data"]["content"] for e in main_tokens)
    assert any("子帧输出" in e["data"]["content"] for e in sub_tokens)

    suspend_event = next(
        e for e in events
        if e["event"] == "mtp_result" and e["data"].get("status") == "suspend"
    )
    assert suspend_event["data"]["scope"] == "main"
    assert suspend_event["data"]["verb"] == "CALL"

    sub_end = next(e for e in events if e["event"] == "sub_agent_end")
    assert sub_end["data"]["status"] == "success"

    done_event = next(e for e in events if e["event"] == "done")
    assert any(
        event["kind"] == "tool_result" and event.get("tool_kind") == "CALL"
        for event in done_event["data"]["turn_events"]
    )


@pytest.mark.asyncio
async def test_execute_frame_stream_subframe_error_still_emits_sub_agent_end():
    """子帧执行失败时，应发出 sub_agent_end(error) 且主帧继续结束。"""
    main_frame, sub_frame = _make_frames()

    call_counter = {"n": 0}

    async def fake_generate_stream(_messages, **_kwargs):
        call_counter["n"] += 1
        n = call_counter["n"]
        if n == 1:
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text='主帧⟪ CALL | coder_doll | task="帮我处理子任务" ⟫',
                    finish_reason="stop",
                    was_mtp_interrupted=True,
                    prefix_text="主帧",
                    mtp_fragment='⟪ CALL | coder_doll | task="帮我处理子任务" ⟫',
                ),
            )
            return
        if n == 2:
            raise RuntimeError("sub frame crashed")
        if n == 3:
            yield StreamChunk(
                is_final=True,
                result=GenerationResult(
                    text="主帧恢复并结束",
                    finish_reason="stop",
                    was_mtp_interrupted=False,
                    prefix_text="主帧恢复并结束",
                    mtp_fragment="",
                ),
            )
            return
        raise AssertionError("generate_stream 被调用超过预期次数")

    orchestrator = _build_orchestrator(fake_generate_stream, main_frame, sub_frame)

    events: List[Dict[str, Any]] = []
    async for event in orchestrator.run_agent_stream(
        messages=[{"role": "user", "content": "主任务"}],
        identity=Identity(user_id="u1"),
        topic_id="topic_1",
    ):
        events.append(event)

    sub_end = next(e for e in events if e["event"] == "sub_agent_end")
    assert sub_end["data"]["status"] == "error"

    done_event = next(e for e in events if e["event"] == "done")
    assert done_event["data"]["final_text"].endswith("主帧恢复并结束")
