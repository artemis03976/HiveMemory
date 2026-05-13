"""
PassiveObserverIngressor + ObserverTurnBuffer 单元测试

测试覆盖:
- ObserverTurnBuffer 状态机: IDLE / AWAITING / SEALED 转换
- target_topic 与轮次绑定 (§3.4 修复)
- flush 返回 (payload, target_topic) 元组
- turn_events 结构化事件流 (Phase P2)
- accept_tool_call / accept_tool_result 工具事件
- assistant_final_text 构建 (仅 assistant_message, 不含 tool 事件)
- PassiveObserverIngressor: ingest_user / ingest_assistant / ingest_tool_* / flush / idle monitor
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent, GatewayResult
from hivememory.patchouli.protocol.models import InteractionPayload
from hivememory.patchouli.protocol.models import EyeGazeResult
from hivememory.patchouli.passive_ingest.observer_turn_buffer import (
    ObserverBufferState,
    ObserverTurnBuffer,
    ObserverTurnBufferManager,
)
from hivememory.patchouli.passive_ingest.ingressor import (
    PassiveObserverIngressor,
)
from hivememory.patchouli.passive_ingest.models import PassiveIngressEvent


def _make_identity(user_id="u1", agent_id="a1", session_id="s1") -> Identity:
    return Identity(user_id=user_id, agent_id=agent_id, session_id=session_id)


def _make_gaze_result(
    raw_query="hello",
    rewritten="hello rewritten",
    worth_saving=True,
    target_topic="topic_001",
) -> EyeGazeResult:
    return EyeGazeResult(
        raw_query=raw_query,
        rewritten_query=rewritten,
        intent=GatewayIntent.RAG,
        search_keywords=["kw1"],
        worth_saving=worth_saving,
        identity=Identity(user_id="u1"),
        target_topic=target_topic,
    )


def _make_gateway_result(**kwargs) -> GatewayResult:
    defaults = dict(
        intent=GatewayIntent.RAG,
        rewritten_query="重写查询",
        search_keywords=["kw1"],
        worth_saving=True,
        reason="有价值",
        target_topic="topic_001",
        new_topic_title=None,
        new_topic_summary=None,
        processing_time_ms=0.0,
    )
    defaults.update(kwargs)
    result = Mock(spec=GatewayResult)
    for k, v in defaults.items():
        setattr(result, k, v)
    return result


# ============================================================
# A. ObserverTurnBuffer 状态机测试
# ============================================================

class TestObserverTurnBufferStates:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_initial_state_idle(self):
        assert self.buf.state == ObserverBufferState.IDLE
        assert self.buf.is_idle is True
        assert self.buf.has_pending_round is False

    def test_accept_user_transitions_to_awaiting(self):
        result = self.buf.accept_user("你好")
        assert self.buf.state == ObserverBufferState.AWAITING_RESPONSE
        assert self.buf.is_awaiting is True
        assert result is None

    def test_accept_assistant_transitions_to_sealed(self):
        self.buf.accept_user("你好")
        self.buf.accept_assistant("回复")
        assert self.buf.state == ObserverBufferState.SEALED
        assert self.buf.is_sealed is True

    def test_accept_assistant_sealed_appends(self):
        self.buf.accept_user("你好")
        self.buf.accept_assistant("第一段")
        self.buf.accept_assistant("第二段")
        assert self.buf.state == ObserverBufferState.SEALED

    def test_accept_user_flushes_previous_from_awaiting(self):
        self.buf.accept_user("第一轮")
        flushed = self.buf.accept_user("第二轮")
        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "第一轮"
        assert self.buf.state == ObserverBufferState.AWAITING_RESPONSE

    def test_accept_user_flushes_previous_from_sealed(self):
        self.buf.accept_user("第一轮")
        self.buf.accept_assistant("回复")
        flushed = self.buf.accept_user("第二轮")
        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "第一轮"
        assert payload.assistant_final_text == "回复"

    def test_accept_assistant_idle_ignored(self):
        self.buf.accept_assistant("孤儿回复")
        assert self.buf.is_idle is True

    def test_flush_idle_returns_none(self):
        result = self.buf.flush()
        assert result is None

    def test_flush_awaiting_returns_payload(self):
        self.buf.accept_user("问题")
        flushed = self.buf.flush()
        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "问题"
        assert payload.assistant_final_text is None
        assert self.buf.is_idle is True

    def test_flush_sealed_returns_payload(self):
        self.buf.accept_user("问题")
        self.buf.accept_assistant("回答")
        flushed = self.buf.flush()
        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "问题"
        assert payload.assistant_final_text == "回答"
        assert self.buf.is_idle is True

    def test_full_cycle(self):
        assert self.buf.is_idle
        self.buf.accept_user("Q")
        assert self.buf.is_awaiting
        self.buf.accept_assistant("A")
        assert self.buf.is_sealed
        flushed = self.buf.flush()
        assert flushed is not None
        assert self.buf.is_idle


# ============================================================
# B. target_topic 绑定测试 (§3.4 修复)
# ============================================================

class TestObserverTurnBufferTargetTopic:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_target_topic_cached_from_gaze_result(self):
        """accept_user 缓存 gaze_result.target_topic"""
        gaze = _make_gaze_result(target_topic="topic_aaa")
        self.buf.accept_user("q", gaze_result=gaze)
        self.buf.accept_assistant("a")
        flushed = self.buf.flush()
        assert flushed is not None
        _, target_topic = flushed
        assert target_topic == "topic_aaa"

    def test_flushed_previous_uses_previous_target_topic(self):
        """
        §3.4 修复验证：flush 上一轮时使用上一轮的 target_topic，
        而不是当前新 user 的 gaze_result.target_topic
        """
        gaze1 = _make_gaze_result(target_topic="topic_round1")
        gaze2 = _make_gaze_result(target_topic="topic_round2")

        self.buf.accept_user("q1", gaze_result=gaze1)
        self.buf.accept_assistant("a1")

        flushed = self.buf.accept_user("q2", gaze_result=gaze2)
        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "q1"
        assert target_topic == "topic_round1"

    def test_no_gaze_result_target_topic_none(self):
        """无 gaze_result 时 target_topic 为 None"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("a")
        flushed = self.buf.flush()
        assert flushed is not None
        _, target_topic = flushed
        assert target_topic is None


# ============================================================
# C. Payload 构建测试
# ============================================================

class TestObserverTurnBufferPayload:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_payload_fields(self):
        self.buf.accept_user("用户消息")
        self.buf.accept_assistant("助手回复")
        flushed = self.buf.flush()
        payload, _ = flushed
        assert payload.user_message == "用户消息"
        assert payload.assistant_final_text == "助手回复"
        assert payload.identity is self.identity

    def test_payload_with_gaze_result(self):
        gaze = _make_gaze_result(rewritten="重写后的查询", worth_saving=True)
        self.buf.accept_user("消息", gaze_result=gaze)
        self.buf.accept_assistant("回复")
        payload, _ = self.buf.flush()
        assert payload.rewritten_query == "重写后的查询"
        assert payload.worth_saving is True

    def test_payload_no_gaze_result(self):
        self.buf.accept_user("消息")
        self.buf.accept_assistant("回复")
        payload, _ = self.buf.flush()
        assert payload.rewritten_query is None
        assert payload.worth_saving is None

    def test_payload_multiple_assistant_parts(self):
        self.buf.accept_user("消息")
        self.buf.accept_assistant("第一段")
        self.buf.accept_assistant("第二段")
        payload, _ = self.buf.flush()
        assert payload.assistant_final_text == "第一段\n第二段"

    def test_payload_passive_mode_fields(self):
        self.buf.accept_user("消息")
        self.buf.accept_assistant("回复")
        payload, _ = self.buf.flush()
        assert payload.mtp_traces == []
        assert payload.write_focus is None
        assert payload.update_focus is None

    def test_identity_preserved_in_payload(self):
        identity = _make_identity(user_id="u99", agent_id="bot", session_id="s1")
        buf = ObserverTurnBuffer(identity=identity)
        buf.accept_user("q")
        buf.accept_assistant("a")
        payload, _ = buf.flush()
        assert payload.identity.user_id == "u99"
        assert payload.identity.agent_id == "bot"


# ============================================================
# D. ObserverTurnBufferManager 测试
# ============================================================

class TestObserverTurnBufferManager:

    def setup_method(self):
        self.manager = ObserverTurnBufferManager()

    def test_get_buffer_creates_new(self):
        identity = _make_identity("u1")
        buf = self.manager.get_buffer(identity)
        assert buf is not None
        assert buf.is_idle

    def test_get_buffer_returns_existing(self):
        identity = _make_identity("u1")
        buf1 = self.manager.get_buffer(identity)
        buf2 = self.manager.get_buffer(identity)
        assert buf1 is buf2

    def test_same_user_agent_different_sessions_get_different_buffers(self):
        id1 = _make_identity(user_id="u1", agent_id="a1", session_id="s1")
        id2 = _make_identity(user_id="u1", agent_id="a1", session_id="s2")
        buf1 = self.manager.get_buffer(id1)
        buf2 = self.manager.get_buffer(id2)
        assert buf1 is not buf2

    def test_remove_buffer(self):
        identity = _make_identity("u1")
        buf1 = self.manager.get_buffer(identity)
        buf1.accept_user("test")
        self.manager.remove_buffer(identity)
        buf2 = self.manager.get_buffer(identity)
        assert buf2 is not buf1
        assert buf2.is_idle

    def test_list_active_buffers(self):
        id1 = _make_identity(user_id="u1", agent_id="a1")
        id2 = _make_identity(user_id="u2", agent_id="a2")
        self.manager.get_buffer(id1)
        self.manager.get_buffer(id2)
        buffers = self.manager.list_active_buffers()
        assert len(buffers) == 2

    def test_flush_idle_buffers(self):
        id1 = _make_identity(user_id="u1", agent_id="a1")
        buf1 = self.manager.get_buffer(id1)
        buf1.accept_user("消息")
        buf1.accept_assistant("回复")
        buf1._last_activity = 0.0

        id2 = _make_identity(user_id="u2", agent_id="a2")
        buf2 = self.manager.get_buffer(id2)
        buf2.accept_user("新消息")

        results = self.manager.flush_idle_buffers(timeout_seconds=1.0)
        assert len(results) == 1
        payload, target_topic = results[0]
        assert payload.user_message == "消息"

    def test_flush_idle_buffers_returns_target_topic(self):
        identity = _make_identity()
        buf = self.manager.get_buffer(identity)
        gaze = _make_gaze_result(target_topic="topic_idle")
        buf.accept_user("q", gaze_result=gaze)
        buf.accept_assistant("a")
        buf._last_activity = 0.0

        results = self.manager.flush_idle_buffers(timeout_seconds=1.0)
        assert len(results) == 1
        _, target_topic = results[0]
        assert target_topic == "topic_idle"


# ============================================================
# E. PassiveObserverIngressor 测试
# ============================================================

class TestPassiveObserverIngressorIngest:

    def setup_method(self):
        self.mock_eye = MagicMock()
        self.mock_eye.gaze = AsyncMock(
            return_value=_make_gaze_result(target_topic="topic_001")
        )
        self.ingressor = PassiveObserverIngressor(
            eye=self.mock_eye, bus=None,
        )

    @pytest.mark.asyncio
    async def test_ingest_user_first_message(self):
        identity = _make_identity()
        gaze_result, flushed = await self.ingressor.ingest_user_async(
            content="你好", identity=identity,
        )
        assert isinstance(gaze_result, EyeGazeResult)
        assert flushed is None

    @pytest.mark.asyncio
    async def test_ingest_user_triggers_flush(self):
        identity = _make_identity()
        gaze1 = _make_gaze_result(target_topic="topic_round1")
        gaze2 = _make_gaze_result(target_topic="topic_round2")
        self.mock_eye.gaze.side_effect = [gaze1, gaze2]

        await self.ingressor.ingest_user_async("第一轮", identity)
        self.ingressor.ingest_assistant("回复", identity)

        gaze_result, flushed = await self.ingressor.ingest_user_async(
            "第二轮", identity,
        )

        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "第一轮"
        assert payload.assistant_final_text == "回复"
        assert target_topic == "topic_round1"

    @pytest.mark.asyncio
    async def test_ingest_assistant_buffers(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("问题", identity)
        self.ingressor.ingest_assistant("回答", identity)

        buf = self.ingressor.buffers.get_buffer(identity)
        assert buf.is_sealed

    @pytest.mark.asyncio
    async def test_flush_session(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("问题", identity)
        self.ingressor.ingest_assistant("回答", identity)

        flushed = self.ingressor.flush_session(identity)

        assert flushed is not None
        payload, target_topic = flushed
        assert payload.user_message == "问题"
        assert target_topic == "topic_001"

    @pytest.mark.asyncio
    async def test_flush_all_pending_sessions(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("消息", identity)
        self.ingressor.ingest_assistant("回复", identity)

        results = self.ingressor.flush_all_pending_sessions()

        assert len(results) == 1
        payload, _ = results[0]
        assert payload.user_message == "消息"
        assert payload.assistant_final_text == "回复"

    @pytest.mark.asyncio
    async def test_route_event_user_returns_user_outcome(self):
        identity = _make_identity()

        outcome = await self.ingressor.route_event(
            PassiveIngressEvent(role="user", content="你好"),
            identity,
        )

        assert outcome.kind == "user"
        assert isinstance(outcome.gaze_result, EyeGazeResult)
        assert outcome.flushed is None

    @pytest.mark.asyncio
    async def test_route_event_tool_call_returns_buffered_outcome(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("请查天气", identity)

        outcome = await self.ingressor.route_event(
            PassiveIngressEvent(
                role="tool_call",
                content="get_weather",
                action_id="a1",
                tool_name="weather_api",
                tool_kind="function_call",
            ),
            identity,
        )

        assert outcome.kind == "buffered"
        assert outcome.gaze_result is None
        assert outcome.flushed is None


# ============================================================
# F. Idle Monitor 测试
# ============================================================

class TestPassiveObserverIngressorIdleMonitor:

    def setup_method(self):
        self.mock_eye = MagicMock()
        self.mock_eye.gaze = AsyncMock(
            return_value=_make_gaze_result(target_topic="topic_001")
        )

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_start_idle_monitor(self, MockScheduler):
        mock_sched = MockScheduler.return_value
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=None)
        ingressor.start_idle_monitor(timeout_seconds=10.0)

        MockScheduler.assert_called_once()
        mock_sched.add_job.assert_called_once()
        mock_sched.start.assert_called_once()

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_start_idle_monitor_double_guard(self, MockScheduler):
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=None)
        ingressor.start_idle_monitor()
        ingressor.start_idle_monitor()

        assert MockScheduler.call_count == 1

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_stop_idle_monitor(self, MockScheduler):
        mock_sched = MockScheduler.return_value
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=None)
        ingressor.start_idle_monitor()
        ingressor.stop_idle_monitor()

        mock_sched.shutdown.assert_called_once_with(wait=False)
        assert ingressor._idle_scheduler is None

    @pytest.mark.asyncio
    async def test_scan_auto_stop_after_global_idle(self):
        mock_bus = Mock()
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=mock_bus)
        ingressor._idle_monitor_enabled = True
        ingressor._idle_scheduler = Mock()
        ingressor._idle_timeout = 9999.0
        ingressor._monitor_idle_shutdown_seconds = 1.0

        identity = _make_identity()
        await ingressor.ingest_user_async("消息", identity)
        ingressor.ingest_assistant("回复", identity)
        ingressor._last_message_ts = 0.0

        ingressor._scan_idle_buffers()

        mock_bus.emit.assert_called_once()
        assert ingressor._idle_scheduler is None

    @pytest.mark.asyncio
    async def test_scan_with_bus(self):
        mock_bus = Mock()
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=mock_bus)

        identity = _make_identity()
        await ingressor.ingest_user_async("消息", identity)
        ingressor.ingest_assistant("回复", identity)
        buf = ingressor.buffers.get_buffer(identity)
        buf._last_activity = 0.0
        ingressor._idle_timeout = 1.0

        ingressor._scan_idle_buffers()

        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args
        assert call_args[0][0] == "observer.idle_flushed"
        assert "target_topic" in call_args[1]

    @pytest.mark.asyncio
    async def test_scan_with_callback(self):
        ingressor = PassiveObserverIngressor(eye=self.mock_eye, bus=None)
        mock_cb = Mock()
        ingressor._on_flush_callback = mock_cb

        identity = _make_identity()
        await ingressor.ingest_user_async("消息", identity)
        ingressor.ingest_assistant("回复", identity)
        buf = ingressor.buffers.get_buffer(identity)
        buf._last_activity = 0.0
        ingressor._idle_timeout = 1.0

        ingressor._scan_idle_buffers()

        mock_cb.assert_called_once()
        call_args = mock_cb.call_args[0]
        assert isinstance(call_args[0], InteractionPayload)
        assert call_args[1] is not None  # target_topic


# ============================================================
# G. turn_events 结构化事件流测试 (Phase P2)
# ============================================================

class TestObserverTurnBufferTurnEvents:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_user_assistant_produces_turn_events(self):
        """user → assistant → flush 产出 turn_events"""
        self.buf.accept_user("问题")
        self.buf.accept_assistant("回答")
        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 2
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[0].role == "user"
        assert payload.turn_events[0].content == "问题"
        assert payload.turn_events[0].sequence == 0
        assert payload.turn_events[1].kind == "assistant_message"
        assert payload.turn_events[1].role == "assistant"
        assert payload.turn_events[1].content == "回答"
        assert payload.turn_events[1].sequence == 1

    def test_user_only_produces_single_event(self):
        """user-only flush 只产出一个 user_message 事件"""
        self.buf.accept_user("问题")
        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 1
        assert payload.turn_events[0].kind == "user_message"

    def test_multi_assistant_produces_multiple_events(self):
        """多段 assistant 产出多个 assistant_message 事件"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("part1")
        self.buf.accept_assistant("part2")
        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 3
        assert payload.turn_events[1].kind == "assistant_message"
        assert payload.turn_events[1].content == "part1"
        assert payload.turn_events[2].kind == "assistant_message"
        assert payload.turn_events[2].content == "part2"

    def test_sequence_numbers_monotonic(self):
        """sequence 编号单调递增"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("a1")
        self.buf.accept_assistant("a2")
        payload, _ = self.buf.flush()

        sequences = [e.sequence for e in payload.turn_events]
        assert sequences == [0, 1, 2]

    def test_flush_resets_turn_events(self):
        """flush 后新一轮的 turn_events 从空开始"""
        self.buf.accept_user("q1")
        self.buf.accept_assistant("a1")
        self.buf.flush()

        self.buf.accept_user("q2")
        self.buf.accept_assistant("a2")
        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 2
        assert payload.turn_events[0].sequence == 0

    def test_next_user_flush_carries_previous_turn_events(self):
        """Next User Turn flush 的 payload 包含上一轮的 turn_events"""
        self.buf.accept_user("q1")
        self.buf.accept_assistant("a1")

        flushed = self.buf.accept_user("q2")
        payload, _ = flushed

        assert len(payload.turn_events) == 2
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[0].content == "q1"
        assert payload.turn_events[1].kind == "assistant_message"
        assert payload.turn_events[1].content == "a1"


# ============================================================
# H. Tool 事件测试 (Phase P2)
# ============================================================

class TestObserverTurnBufferToolEvents:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_tool_call_transitions_to_sealed(self):
        """AWAITING → SEALED via tool_call"""
        self.buf.accept_user("请查天气")
        self.buf.accept_tool_call(
            "get_weather",
            tool_name="weather_api",
            action_id="act_1",
        )
        assert self.buf.is_sealed

    def test_tool_result_transitions_to_sealed(self):
        """AWAITING → SEALED via tool_result"""
        self.buf.accept_user("请查天气")
        self.buf.accept_tool_result(
            "北京 25°C",
            action_id="act_1",
            status="success",
        )
        assert self.buf.is_sealed

    def test_tool_call_idle_ignored(self):
        """IDLE 状态收到 tool_call 忽略"""
        self.buf.accept_tool_call("orphan", tool_name="t")
        assert self.buf.is_idle

    def test_tool_result_idle_ignored(self):
        """IDLE 状态收到 tool_result 忽略"""
        self.buf.accept_tool_result("orphan")
        assert self.buf.is_idle

    def test_full_tool_flow_turn_events(self):
        """user → tool_call → tool_result → assistant → flush 完整事件流"""
        self.buf.accept_user("请查天气")
        self.buf.accept_tool_call(
            "get_weather(city='北京')",
            action_id="act_1",
            tool_name="weather_api",
            tool_kind="function_call",
            tool_args={"city": "北京"},
        )
        self.buf.accept_tool_result(
            "北京 25°C 晴",
            action_id="act_1",
            status="success",
        )
        self.buf.accept_assistant("北京现在是25度，天气晴朗。")

        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 4

        # user_message
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[0].content == "请查天气"

        # tool_call
        tc = payload.turn_events[1]
        assert tc.kind == "tool_call"
        assert tc.role == "assistant"
        assert tc.action_id == "act_1"
        assert tc.tool_name == "weather_api"
        assert tc.tool_kind == "function_call"
        assert tc.tool_args == {"city": "北京"}

        # tool_result
        tr = payload.turn_events[2]
        assert tr.kind == "tool_result"
        assert tr.role == "system"
        assert tr.action_id == "act_1"
        assert tr.status == "success"
        assert tr.content == "北京 25°C 晴"

        # assistant_message
        assert payload.turn_events[3].kind == "assistant_message"
        assert "25度" in payload.turn_events[3].content

    def test_tool_events_not_in_assistant_final_text(self):
        """tool_call / tool_result 不进入 assistant_final_text"""
        self.buf.accept_user("请查天气")
        self.buf.accept_tool_call("get_weather", tool_name="w")
        self.buf.accept_tool_result("25°C")
        self.buf.accept_assistant("天气是25度。")

        payload, _ = self.buf.flush()

        assert payload.assistant_final_text == "天气是25度。"
        assert "get_weather" not in (payload.assistant_final_text or "")

    def test_tool_only_no_assistant_text(self):
        """仅有 tool 事件、无 assistant 时，assistant_final_text 为 None"""
        self.buf.accept_user("查天气")
        self.buf.accept_tool_call("get_weather", tool_name="w")
        self.buf.accept_tool_result("25°C")

        payload, _ = self.buf.flush()

        assert payload.assistant_final_text is None
        assert payload.assistant_final_text is None
        assert len(payload.turn_events) == 3

    def test_multiple_tool_calls_in_one_turn(self):
        """单轮多次工具调用"""
        self.buf.accept_user("查两个城市的天气")
        self.buf.accept_tool_call("get_weather", action_id="a1", tool_name="w")
        self.buf.accept_tool_result("北京 25°C", action_id="a1")
        self.buf.accept_tool_call("get_weather", action_id="a2", tool_name="w")
        self.buf.accept_tool_result("上海 28°C", action_id="a2")
        self.buf.accept_assistant("北京25度，上海28度。")

        payload, _ = self.buf.flush()

        assert len(payload.turn_events) == 6
        tool_calls = [e for e in payload.turn_events if e.kind == "tool_call"]
        tool_results = [e for e in payload.turn_events if e.kind == "tool_result"]
        assert len(tool_calls) == 2
        assert len(tool_results) == 2
        assert tool_calls[0].action_id == "a1"
        assert tool_calls[1].action_id == "a2"

    def test_tool_result_render_as_propagated(self):
        """render_as 参数正确传递到 TurnEvent"""
        self.buf.accept_user("q")
        self.buf.accept_tool_result(
            "result",
            action_id="a1",
            render_as="system_tool_result",
        )
        payload, _ = self.buf.flush()

        tr = payload.turn_events[1]
        assert tr.render_as == "system_tool_result"

    def test_tool_call_target_propagated(self):
        """target 参数正确传递到 TurnEvent"""
        self.buf.accept_user("q")
        self.buf.accept_tool_call(
            "call sub-agent",
            action_id="a1",
            tool_name="delegate",
            target="sub_agent_1",
        )
        payload, _ = self.buf.flush()

        tc = payload.turn_events[1]
        assert tc.target == "sub_agent_1"


# ============================================================
# I. assistant_final_text 构建规则 (Phase P2 §5.3)
# ============================================================

class TestAssistantFinalText:

    def setup_method(self):
        self.identity = _make_identity()
        self.buf = ObserverTurnBuffer(self.identity)

    def test_simple_assistant_final_text(self):
        """单条 assistant → assistant_final_text"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("回答")
        payload, _ = self.buf.flush()

        assert payload.assistant_final_text == "回答"

    def test_multi_part_assistant_final_text(self):
        """多段 assistant 拼接为 assistant_final_text"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("第一段")
        self.buf.accept_assistant("第二段")
        payload, _ = self.buf.flush()

        assert payload.assistant_final_text == "第一段\n第二段"

    def test_no_assistant_final_text_none(self):
        """无 assistant 消息时 assistant_final_text 为 None"""
        self.buf.accept_user("q")
        payload, _ = self.buf.flush()

        assert payload.assistant_final_text is None

    def test_assistant_final_text_matches_joined_assistant_parts(self):
        """assistant_final_text 等于按顺序拼接的 assistant 文本"""
        self.buf.accept_user("q")
        self.buf.accept_assistant("回答")
        payload, _ = self.buf.flush()

        assert payload.assistant_final_text == "回答"


# ============================================================
# J. PassiveObserverIngressor Tool 事件测试 (Phase P2)
# ============================================================

class TestPassiveObserverIngressorToolIngest:

    def setup_method(self):
        self.mock_eye = MagicMock()
        self.mock_eye.gaze = AsyncMock(
            return_value=_make_gaze_result(target_topic="topic_001")
        )
        self.ingressor = PassiveObserverIngressor(
            eye=self.mock_eye, bus=None,
        )

    @pytest.mark.asyncio
    async def test_ingest_tool_call_buffers(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("请查天气", identity)
        self.ingressor.ingest_tool_call(
            "get_weather", identity,
            action_id="a1", tool_name="weather_api",
        )

        buf = self.ingressor.buffers.get_buffer(identity)
        assert buf.is_sealed

    @pytest.mark.asyncio
    async def test_ingest_tool_result_buffers(self):
        identity = _make_identity()
        await self.ingressor.ingest_user_async("q", identity)
        self.ingressor.ingest_tool_result(
            "25°C", identity,
            action_id="a1", status="success",
        )

        buf = self.ingressor.buffers.get_buffer(identity)
        assert buf.is_sealed

    @pytest.mark.asyncio
    async def test_full_tool_flow_via_ingressor(self):
        """ingressor 层完整 tool 流: user → tool_call → tool_result → assistant → flush"""
        identity = _make_identity()
        await self.ingressor.ingest_user_async("请查天气", identity)
        self.ingressor.ingest_tool_call(
            "get_weather(city='北京')", identity,
            action_id="a1", tool_name="weather",
            tool_kind="function_call", tool_args={"city": "北京"},
        )
        self.ingressor.ingest_tool_result(
            "北京 25°C", identity,
            action_id="a1", status="success",
        )
        self.ingressor.ingest_assistant("北京25度。", identity)

        flushed = self.ingressor.flush_session(identity)
        payload, target_topic = flushed

        assert len(payload.turn_events) == 4
        assert payload.turn_events[0].kind == "user_message"
        assert payload.turn_events[1].kind == "tool_call"
        assert payload.turn_events[2].kind == "tool_result"
        assert payload.turn_events[3].kind == "assistant_message"
        assert payload.assistant_final_text == "北京25度。"
        assert target_topic == "topic_001"
