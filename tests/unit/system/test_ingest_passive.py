"""
被动观测模式 (Passive Observer Mode) 测试

覆盖:
    A. ObserverSessionBuffer 单元测试 — 状态机、flush 触发器、边界情况
    B. ObserverBufferManager 单元测试 — 多 session 隔离、idle timeout
    C. PatchouliSystem.ingest() 集成测试 — 完整 user→assistant→user 流程

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import time
import types
import threading
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from hivememory.core.models import Identity
from hivememory.engines.perception.models import InteractionPayload
from hivememory.engines.gateway.observer_buffer import (
    ObserverBufferState,
    ObserverSessionBuffer,
    ObserverBufferManager,
)
from hivememory.patchouli.protocol.models import (
    EyeGazeResult, KernelHotResult,
)
from hivememory.engines.gateway.models import GatewayIntent


# ========== Helpers ==========

def _make_identity(user_id="u1", agent_id="default", session_id=None) -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id)


def _make_gaze_result(
    raw_query="hello",
    rewritten="hello rewritten",
    worth_saving=True,
    user_id="u1",
) -> EyeGazeResult:
    return EyeGazeResult(
        raw_query=raw_query,
        rewritten_query=rewritten,
        intent=GatewayIntent.CHAT,
        search_keywords=[],
        worth_saving=worth_saving,
        identity=Identity(user_id=user_id),
    )


def _make_hot_result(memory=None) -> KernelHotResult:
    return KernelHotResult(
        intent="Chat",
        rewritten="hello rewritten",
        keywords=[],
        worth_saving=True,
        memory=memory,
    )


# ============================================================
# A. ObserverSessionBuffer 单元测试
# ============================================================

class TestObserverSessionBufferStateMachine:
    """状态机转换: IDLE → AWAITING → SEALED → flush → IDLE"""

    def test_initial_state_is_idle(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        assert buf.state == ObserverBufferState.IDLE
        assert buf.is_idle
        assert not buf.has_pending_round

    def test_accept_user_transitions_to_awaiting(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        result = buf.accept_user("hi")
        assert buf.state == ObserverBufferState.AWAITING_RESPONSE
        assert buf.is_awaiting
        assert result is None  # 首次 user，无上一轮 flush

    def test_accept_assistant_transitions_to_sealed(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("hi")
        buf.accept_assistant("hello!")
        assert buf.state == ObserverBufferState.SEALED
        assert buf.is_sealed

    def test_flush_sealed_returns_payload_and_resets(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("hi")
        buf.accept_assistant("hello!")
        payload = buf.flush()

        assert payload is not None
        assert isinstance(payload, InteractionPayload)
        assert payload.user_message == "hi"
        assert payload.assistant_message == "hello!"
        assert buf.state == ObserverBufferState.IDLE

    def test_flush_idle_returns_none(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        assert buf.flush() is None


class TestObserverNextUserTurnTrigger:
    """'Next User Turn' 触发器: 第二个 user 消息自动 flush 上一轮"""

    def test_second_user_flushes_previous_sealed_round(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("q1")
        buf.accept_assistant("a1")
        # 第二个 user 消息到达
        flushed = buf.accept_user("q2")

        assert flushed is not None
        assert flushed.user_message == "q1"
        assert flushed.assistant_message == "a1"
        # buffer 现在持有 q2
        assert buf.state == ObserverBufferState.AWAITING_RESPONSE

    def test_second_user_flushes_previous_awaiting_round(self):
        """连续 user 消息 → flush 上一轮 (user-only payload)"""
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("q1")
        # 没有 assistant，直接来第二个 user
        flushed = buf.accept_user("q2")

        assert flushed is not None
        assert flushed.user_message == "q1"
        assert flushed.assistant_message == ""  # 无 assistant
        assert buf.is_awaiting

class TestObserverMultiAssistant:
    """多段 assistant 拼接"""

    def test_multiple_assistant_parts_joined(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("q")
        buf.accept_assistant("part1")
        buf.accept_assistant("part2")
        buf.accept_assistant("part3")

        payload = buf.flush()
        assert payload.assistant_message == "part1\npart2\npart3"

    def test_assistant_without_user_ignored(self):
        """孤立 assistant（无配对 user）→ 忽略"""
        buf = ObserverSessionBuffer(identity=_make_identity())
        assert buf.is_idle
        buf.accept_assistant("orphan")
        # 状态不变，仍然 IDLE
        assert buf.is_idle
        assert buf.flush() is None


class TestObserverGazeResultPropagation:
    """EyeGazeResult 元数据正确传递到 InteractionPayload"""

    def test_gaze_result_fields_in_payload(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        gaze = _make_gaze_result(rewritten="resolved query", worth_saving=True)
        buf.accept_user("raw q", gaze_result=gaze)
        buf.accept_assistant("answer")
        payload = buf.flush()

        assert payload.rewritten_query == "resolved query"
        assert payload.worth_saving is True

    def test_no_gaze_result_defaults_to_none(self):
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("q")
        buf.accept_assistant("a")
        payload = buf.flush()

        assert payload.rewritten_query is None
        assert payload.worth_saving is None

    def test_passive_payload_has_empty_mtp_fields(self):
        """被动模式 payload: mtp_traces 空, write/update_focus None"""
        buf = ObserverSessionBuffer(identity=_make_identity())
        buf.accept_user("q", gaze_result=_make_gaze_result())
        buf.accept_assistant("a")
        payload = buf.flush()

        assert payload.mtp_traces == []
        assert payload.write_focus is None
        assert payload.update_focus is None

    def test_identity_preserved_in_payload(self):
        identity = _make_identity(user_id="u99", agent_id="bot", session_id="s1")
        buf = ObserverSessionBuffer(identity=identity)
        buf.accept_user("q")
        buf.accept_assistant("a")
        payload = buf.flush()

        assert payload.identity.user_id == "u99"
        assert payload.identity.agent_id == "bot"


# ============================================================
# B. ObserverBufferManager 单元测试
# ============================================================

class TestObserverBufferManagerMultiSession:
    """多 session 隔离"""

    def test_different_sessions_get_different_buffers(self):
        mgr = ObserverBufferManager()
        id1 = _make_identity(user_id="u1", agent_id="a1", session_id="s1")
        id2 = _make_identity(user_id="u1", agent_id="a2", session_id="s2")

        buf1 = mgr.get_buffer(id1)
        buf2 = mgr.get_buffer(id2)

        assert buf1 is not buf2

    def test_same_session_returns_same_buffer(self):
        mgr = ObserverBufferManager()
        identity = _make_identity(user_id="u1", session_id="s1")

        buf1 = mgr.get_buffer(identity)
        buf2 = mgr.get_buffer(identity)

        assert buf1 is buf2

    def test_remove_buffer(self):
        mgr = ObserverBufferManager()
        identity = _make_identity(user_id="u1", session_id="s1")
        mgr.get_buffer(identity)
        mgr.remove_buffer(identity)

        # 再次 get 应返回新 buffer
        buf = mgr.get_buffer(identity)
        assert buf.is_idle

    def test_list_active_buffers(self):
        mgr = ObserverBufferManager()
        id1 = _make_identity(user_id="u1", session_id="s1")
        id2 = _make_identity(user_id="u2", session_id="s2")
        mgr.get_buffer(id1)
        mgr.get_buffer(id2)

        active = mgr.list_active_buffers()
        assert len(active) == 2


class TestObserverBufferManagerIdleTimeout:
    """flush_idle_buffers() 超时检测"""

    def test_flush_idle_buffers_respects_timeout(self):
        mgr = ObserverBufferManager()
        identity = _make_identity()
        buf = mgr.get_buffer(identity)
        buf.accept_user("q")
        buf.accept_assistant("a")

        # 伪造 last_activity 为很久以前
        buf._last_activity = datetime.now().timestamp() - 60

        payloads = mgr.flush_idle_buffers(timeout_seconds=10)
        assert len(payloads) == 1
        assert payloads[0].user_message == "q"
        # flush 后 buffer 回到 IDLE
        assert buf.is_idle

    def test_flush_idle_buffers_skips_recent(self):
        mgr = ObserverBufferManager()
        identity = _make_identity()
        buf = mgr.get_buffer(identity)
        buf.accept_user("q")
        buf.accept_assistant("a")
        # last_activity 是刚刚，不应被 flush

        payloads = mgr.flush_idle_buffers(timeout_seconds=30)
        assert len(payloads) == 0
        assert buf.is_sealed  # 未被 flush

    def test_flush_idle_skips_idle_buffers(self):
        """IDLE 状态的 buffer 不应被 flush"""
        mgr = ObserverBufferManager()
        identity = _make_identity()
        mgr.get_buffer(identity)  # 空 buffer

        payloads = mgr.flush_idle_buffers(timeout_seconds=0)
        assert len(payloads) == 0


class TestObserverBufferManagerThreadSafety:
    """线程安全基本验证"""

    def test_concurrent_get_buffer(self):
        mgr = ObserverBufferManager()
        results = []
        errors = []

        def worker(uid):
            try:
                identity = _make_identity(user_id=uid)
                buf = mgr.get_buffer(identity)
                buf.accept_user(f"msg from {uid}")
                results.append(uid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20


# ============================================================
# C. PatchouliSystem.ingest() 集成测试
# ============================================================

@pytest.fixture
def sys_passive():
    """
    构建最小化 PatchouliSystem mock (被动模式):
    mock Eye (gaze), Kernel — 绑定真实 ingest() / flush_observer_session()
    Eye 持有真实的 ObserverBufferManager，绑定真实的被动观测方法
    """
    from hivememory.patchouli.system import PatchouliSystem
    from hivememory.patchouli.eye import TheEye

    s = MagicMock(spec=PatchouliSystem)

    # Eye — mock spec，但持有真实 buffer 池和真实被动方法
    s.eye = MagicMock(spec=TheEye)
    s.eye.gaze.return_value = _make_gaze_result()
    s.eye._observer_buffers = ObserverBufferManager()
    s.eye._observer_idle_timeout = 30.0
    s.eye.ingest_user = types.MethodType(TheEye.ingest_user, s.eye)
    s.eye.ingest_assistant = types.MethodType(TheEye.ingest_assistant, s.eye)
    s.eye.flush_session = types.MethodType(TheEye.flush_session, s.eye)
    s.eye.flush_idle_sessions = types.MethodType(TheEye.flush_idle_sessions, s.eye)

    # Kernel
    s.kernel = MagicMock()
    s.kernel.handle_hot.return_value = _make_hot_result(memory="<mem>ctx</mem>")
    s.kernel.submit_interaction = AsyncMock(return_value=None)

    # 绑定真实方法
    from hivememory.patchouli.system import PatchouliSystem as Real
    _ingest_async = types.MethodType(Real.ingest, s)
    _flush_async = types.MethodType(Real.flush_observer_session, s)
    s.ingest = lambda *args, **kwargs: asyncio.run(_ingest_async(*args, **kwargs))
    s.flush_observer_session = lambda *args, **kwargs: asyncio.run(
        _flush_async(*args, **kwargs)
    )

    return s


class TestIngestUserFlow:
    """ingest(role='user') 流程"""

    def test_user_ingest_returns_expected_keys(self, sys_passive):
        result = sys_passive.ingest(
            role="user", content="hello", user_id="u1",
        )

        assert "intent" in result
        assert "rewritten" in result
        assert "keywords" in result
        assert "worth_saving" in result
        assert "memory" in result

    def test_user_ingest_calls_eye_gaze(self, sys_passive):
        sys_passive.ingest(
            role="user", content="test query", user_id="u1",
        )

        sys_passive.eye.gaze.assert_called_once()
        call_kwargs = sys_passive.eye.gaze.call_args.kwargs
        assert call_kwargs["query"] == "test query"

    def test_user_ingest_calls_handle_hot_passive(self, sys_passive):
        sys_passive.ingest(
            role="user", content="q", user_id="u1",
        )

        sys_passive.kernel.handle_hot.assert_called_once()
        call_kwargs = sys_passive.kernel.handle_hot.call_args
        assert call_kwargs.kwargs.get("mode") == "passive"

    def test_user_ingest_returns_memory(self, sys_passive):
        sys_passive.kernel.handle_hot.return_value = _make_hot_result(
            memory="<memory>relevant</memory>"
        )

        result = sys_passive.ingest(
            role="user", content="q", user_id="u1",
        )

        assert result["memory"] == "<memory>relevant</memory>"

    def test_identity_constructed_correctly(self, sys_passive):
        sys_passive.ingest(
            role="user", content="q",
            user_id="ux", agent_id="ax", session_id="sx",
        )

        call_kwargs = sys_passive.eye.gaze.call_args.kwargs
        identity = call_kwargs["identity"]
        assert identity.user_id == "ux"
        assert identity.agent_id == "ax"


class TestIngestAssistantFlow:
    """ingest(role='assistant') 流程"""

    def test_assistant_ingest_returns_buffered(self, sys_passive):
        # 先 ingest user 建立配对
        sys_passive.ingest(role="user", content="q", user_id="u1")

        result = sys_passive.ingest(
            role="assistant", content="answer", user_id="u1",
        )

        assert result["intent"] == "buffered"
        assert result["worth_saving"] is True

    def test_assistant_ingest_does_not_submit(self, sys_passive):
        """assistant 消息仅缓冲，不立即提交"""
        sys_passive.ingest(role="user", content="q", user_id="u1")
        sys_passive.ingest(role="assistant", content="a", user_id="u1")

        sys_passive.kernel.submit_interaction.assert_not_called()

    def test_other_role_returns_ignored(self, sys_passive):
        result = sys_passive.ingest(
            role="system", content="sys msg", user_id="u1",
        )

        assert result["intent"] == "ignored"
        assert result["worth_saving"] is False


class TestIngestFullRoundTrip:
    """完整 user → assistant → user 流程，验证 submit_interaction"""

    def test_next_user_triggers_submit(self, sys_passive):
        """第二个 user 消息触发上一轮 payload 提交"""
        # Round 1
        sys_passive.ingest(role="user", content="q1", user_id="u1")
        sys_passive.ingest(role="assistant", content="a1", user_id="u1")

        # Round 2 — 触发 Round 1 的 flush
        sys_passive.ingest(role="user", content="q2", user_id="u1")

        sys_passive.kernel.submit_interaction.assert_called_once()
        payload = sys_passive.kernel.submit_interaction.call_args[0][0]
        assert payload.user_message == "q1"
        assert payload.assistant_message == "a1"

    def test_explicit_flush_submits_payload(self, sys_passive):
        """flush_observer_session() 显式提交当前轮"""
        sys_passive.ingest(role="user", content="q", user_id="u1")
        sys_passive.ingest(role="assistant", content="a", user_id="u1")

        flushed = sys_passive.flush_observer_session(user_id="u1")

        assert flushed is True
        sys_passive.kernel.submit_interaction.assert_called_once()
        payload = sys_passive.kernel.submit_interaction.call_args[0][0]
        assert payload.user_message == "q"
        assert payload.assistant_message == "a"

    def test_explicit_flush_empty_returns_false(self, sys_passive):
        """空 session flush 返回 False"""
        flushed = sys_passive.flush_observer_session(user_id="u1")
        assert flushed is False
        sys_passive.kernel.submit_interaction.assert_not_called()

    def test_multi_round_submits_each_round(self, sys_passive):
        """多轮对话，每轮都被正确提交"""
        # Round 1
        sys_passive.ingest(role="user", content="q1", user_id="u1")
        sys_passive.ingest(role="assistant", content="a1", user_id="u1")
        # Round 2 — flush Round 1
        sys_passive.ingest(role="user", content="q2", user_id="u1")
        sys_passive.ingest(role="assistant", content="a2", user_id="u1")
        # Explicit flush Round 2
        sys_passive.flush_observer_session(user_id="u1")

        assert sys_passive.kernel.submit_interaction.call_count == 2
        p1 = sys_passive.kernel.submit_interaction.call_args_list[0][0][0]
        p2 = sys_passive.kernel.submit_interaction.call_args_list[1][0][0]
        assert p1.user_message == "q1"
        assert p2.user_message == "q2"

    def test_payload_carries_gaze_metadata(self, sys_passive):
        """提交的 payload 携带 Eye 分析的元数据"""
        gaze = _make_gaze_result(rewritten="resolved", worth_saving=True)
        sys_passive.eye.gaze.return_value = gaze

        sys_passive.ingest(role="user", content="q1", user_id="u1")
        sys_passive.ingest(role="assistant", content="a1", user_id="u1")
        sys_passive.flush_observer_session(user_id="u1")

        payload = sys_passive.kernel.submit_interaction.call_args[0][0]
        assert payload.rewritten_query == "resolved"
        assert payload.worth_saving is True
        assert payload.mtp_traces == []
