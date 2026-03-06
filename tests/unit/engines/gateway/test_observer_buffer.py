"""
ObserverBuffer 单元测试

测试覆盖:
- ObserverSessionBuffer 状态机: IDLE / AWAITING / SEALED 转换
- Payload 构建: 字段填充 / gaze_result 缓存 / 多段 assistant
- ObserverBufferManager: 创建 / 复用 / 移除 / idle flush
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from hivememory.core.models import Identity
from hivememory.engines.gateway.observer_buffer import (
    ObserverBufferState,
    ObserverSessionBuffer,
    ObserverBufferManager,
)


def _make_identity(session_id="s1", user_id="u1", agent_id="a1") -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id, session_id=session_id)


def _make_gaze_result(rewritten_query="重写查询", worth_saving=True):
    """构建 mock EyeGazeResult"""
    gaze = Mock()
    gaze.rewritten_query = rewritten_query
    gaze.worth_saving = worth_saving
    return gaze


class TestObserverSessionBufferStates:
    """状态机转换测试"""

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverSessionBuffer(self.identity)

    def test_initial_state_idle(self):
        """初始状态为 IDLE"""
        assert self.buf.state == ObserverBufferState.IDLE
        assert self.buf.is_idle is True
        assert self.buf.has_pending_round is False

    def test_accept_user_transitions_to_awaiting(self):
        """IDLE → AWAITING"""
        result = self.buf.accept_user("你好")
        assert self.buf.state == ObserverBufferState.AWAITING_RESPONSE
        assert self.buf.is_awaiting is True
        assert result is None  # 首条消息无 flush

    def test_accept_assistant_transitions_to_sealed(self):
        """AWAITING → SEALED"""
        self.buf.accept_user("你好")
        self.buf.accept_assistant("回复")
        assert self.buf.state == ObserverBufferState.SEALED
        assert self.buf.is_sealed is True

    def test_accept_assistant_sealed_appends(self):
        """SEALED → SEALED（追加 assistant）"""
        self.buf.accept_user("你好")
        self.buf.accept_assistant("第一段")
        self.buf.accept_assistant("第二段")
        assert self.buf.state == ObserverBufferState.SEALED

    def test_accept_user_flushes_previous_from_awaiting(self):
        """AWAITING 状态收到新 user 消息时 flush 上一轮"""
        self.buf.accept_user("第一轮")
        payload = self.buf.accept_user("第二轮")
        assert payload is not None
        assert payload.user_message == "第一轮"
        assert self.buf.state == ObserverBufferState.AWAITING_RESPONSE

    def test_accept_user_flushes_previous_from_sealed(self):
        """SEALED 状态收到新 user 消息时 flush 上一轮"""
        self.buf.accept_user("第一轮")
        self.buf.accept_assistant("回复")
        payload = self.buf.accept_user("第二轮")
        assert payload is not None
        assert payload.user_message == "第一轮"
        assert payload.assistant_message == "回复"

    def test_accept_assistant_idle_ignored(self):
        """IDLE 状态收到 assistant 忽略（不崩溃）"""
        self.buf.accept_assistant("孤儿回复")
        assert self.buf.is_idle is True

    def test_flush_idle_returns_none(self):
        """IDLE 状态 flush 返回 None"""
        result = self.buf.flush()
        assert result is None

    def test_flush_awaiting_returns_payload(self):
        """AWAITING 状态 flush 返回 payload（无 assistant）"""
        self.buf.accept_user("问题")
        payload = self.buf.flush()
        assert payload is not None
        assert payload.user_message == "问题"
        assert payload.assistant_message == ""
        assert self.buf.is_idle is True

    def test_flush_sealed_returns_payload(self):
        """SEALED 状态 flush 返回完整 payload"""
        self.buf.accept_user("问题")
        self.buf.accept_assistant("回答")
        payload = self.buf.flush()
        assert payload is not None
        assert payload.user_message == "问题"
        assert payload.assistant_message == "回答"
        assert self.buf.is_idle is True

    def test_full_cycle(self):
        """IDLE → AWAITING → SEALED → flush → IDLE 完整循环"""
        assert self.buf.is_idle
        self.buf.accept_user("Q")
        assert self.buf.is_awaiting
        self.buf.accept_assistant("A")
        assert self.buf.is_sealed
        payload = self.buf.flush()
        assert payload is not None
        assert self.buf.is_idle


class TestObserverSessionBufferPayload:
    """Payload 构建测试"""

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverSessionBuffer(self.identity)

    def test_payload_fields(self):
        """InteractionPayload 字段正确填充"""
        self.buf.accept_user("用户消息")
        self.buf.accept_assistant("助手回复")
        payload = self.buf.flush()
        assert payload.user_message == "用户消息"
        assert payload.assistant_message == "助手回复"
        assert payload.identity is self.identity

    def test_payload_no_gaze_result(self):
        """无 gaze_result 时 rewritten_query/worth_saving 为 None"""
        self.buf.accept_user("消息")
        self.buf.accept_assistant("回复")
        payload = self.buf.flush()
        assert payload.rewritten_query is None
        assert payload.worth_saving is None

    def test_payload_with_gaze_result(self):
        """有 gaze_result 时字段正确提取"""
        gaze = _make_gaze_result("重写后的查询", True)
        self.buf.accept_user("消息", gaze_result=gaze)
        self.buf.accept_assistant("回复")
        payload = self.buf.flush()
        assert payload.rewritten_query == "重写后的查询"
        assert payload.worth_saving is True

    def test_payload_multiple_assistant_parts(self):
        """多段 assistant 用 \\n 连接"""
        self.buf.accept_user("消息")
        self.buf.accept_assistant("第一段")
        self.buf.accept_assistant("第二段")
        payload = self.buf.flush()
        assert payload.assistant_message == "第一段\n第二段"

    def test_payload_passive_mode_fields(self):
        """被动模式字段: mtp_traces=[], write_focus=None, update_focus=None"""
        self.buf.accept_user("消息")
        self.buf.accept_assistant("回复")
        payload = self.buf.flush()
        assert payload.mtp_traces == []
        assert payload.write_focus is None
        assert payload.update_focus is None


class TestObserverBufferManager:
    """ObserverBufferManager 池管理测试"""

    def setup_method(self):
        self.manager = ObserverBufferManager()

    def test_get_buffer_creates_new(self):
        """首次访问创建新 buffer"""
        identity = _make_identity("s1")
        buf = self.manager.get_buffer(identity)
        assert buf is not None
        assert buf.is_idle

    def test_get_buffer_returns_existing(self):
        """重复访问返回同一 buffer"""
        identity = _make_identity("s1")
        buf1 = self.manager.get_buffer(identity)
        buf2 = self.manager.get_buffer(identity)
        assert buf1 is buf2

    def test_remove_buffer(self):
        """移除后再获取是新 buffer"""
        identity = _make_identity("s1")
        buf1 = self.manager.get_buffer(identity)
        buf1.accept_user("test")
        self.manager.remove_buffer(identity)
        buf2 = self.manager.get_buffer(identity)
        assert buf2 is not buf1
        assert buf2.is_idle

    def test_list_active_buffers(self):
        """返回快照"""
        id1 = _make_identity("s1", user_id="u1", agent_id="a1")
        id2 = _make_identity("s2", user_id="u2", agent_id="a2")
        self.manager.get_buffer(id1)
        self.manager.get_buffer(id2)
        buffers = self.manager.list_active_buffers()
        assert len(buffers) == 2

    def test_flush_idle_buffers(self):
        """超时 buffer 被 flush，活跃 buffer 保留"""
        id1 = _make_identity("s1", user_id="u1", agent_id="a1")
        buf1 = self.manager.get_buffer(id1)
        buf1.accept_user("消息")
        buf1.accept_assistant("回复")
        # 手动设置 last_activity 为很久以前
        buf1._last_activity = 0.0

        id2 = _make_identity("s2", user_id="u2", agent_id="a2")
        buf2 = self.manager.get_buffer(id2)
        buf2.accept_user("新消息")
        # buf2 的 last_activity 是刚刚，不会超时

        payloads = self.manager.flush_idle_buffers(timeout_seconds=1.0)
        assert len(payloads) == 1
        assert payloads[0].user_message == "消息"
