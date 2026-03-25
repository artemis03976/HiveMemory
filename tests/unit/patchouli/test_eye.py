"""
TheEye 单元测试

测试覆盖:
- gaze: 正常 / fallback / identity 默认值 / active_topics_menu 传递
- 被动模式: ingest_user / ingest_assistant / flush_session / flush_idle
- idle monitor: 启动 / 重复启动 / 停止 / scan 分支
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent, GatewayResult
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.protocol.models import EyeGazeResult


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1", session_id="s1")


def _make_gateway_result(**kwargs) -> GatewayResult:
    """构建 mock GatewayResult"""
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


@pytest.mark.asyncio
class TestTheEyeGaze:
    """gaze() 方法测试"""

    def setup_method(self):
        self.mock_engine = Mock()
        self.mock_engine.process = AsyncMock()
        self.eye = TheEye(engine=self.mock_engine, bus=None)

    async def test_gaze_success(self):
        """正常调用 engine.process，返回 EyeGazeResult(is_fallback=False)"""
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        self.mock_engine.process.assert_called_once()
        assert isinstance(result, EyeGazeResult)
        assert result.is_fallback is False
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "重写查询"

    async def test_gaze_fallback_on_exception(self):
        """engine.process 抛异常时返回 fallback"""
        self.mock_engine.process.side_effect = RuntimeError("boom")

        result = await self.eye.gaze("测试查询", identity=_make_identity())

        assert result.is_fallback is True
        assert result.intent == GatewayIntent.RAG
        assert result.rewritten_query == "测试查询"
        assert result.target_topic == "NEW_TOPIC"

    async def test_gaze_identity_default(self):
        """identity=None 时使用 Identity() 默认值"""
        self.mock_engine.process.return_value = _make_gateway_result()

        result = await self.eye.gaze("查询", identity=None)

        assert result.identity is not None

    async def test_gaze_forwards_active_topics_menu(self):
        """engine.process 接收 active_topics_menu 关键字参数"""
        self.mock_engine.process.return_value = _make_gateway_result()

        await self.eye.gaze(
            "查询",
            topic_snapshots=[],
            identity=_make_identity(),
        )

        call_args = self.mock_engine.process.call_args
        assert call_args[0][0] == "查询"
        assert call_args[1]["active_topics_menu"] is None


class TestTheEyePassiveMode:
    """被动观测模式测试"""

    def setup_method(self):
        self.mock_engine = Mock()
        self.mock_engine.process.return_value = _make_gateway_result()
        self.eye = TheEye(engine=self.mock_engine, bus=None)
        self.identity = _make_identity()

    def test_ingest_user_first_message(self):
        """首条 user 消息，无 flush"""
        gaze_result, flushed = self.eye.ingest_user("你好", self.identity)

        assert isinstance(gaze_result, EyeGazeResult)
        assert flushed is None

    def test_ingest_user_triggers_flush(self):
        """第二条 user 消息触发上一轮 flush"""
        self.eye.ingest_user("第一轮", self.identity)
        self.eye.ingest_assistant("回复", self.identity)

        gaze_result, flushed = self.eye.ingest_user("第二轮", self.identity)

        assert flushed is not None
        assert flushed.user_message == "第一轮"
        assert flushed.assistant_message == "回复"

    def test_ingest_assistant(self):
        """assistant 消息正确缓冲"""
        self.eye.ingest_user("问题", self.identity)
        self.eye.ingest_assistant("回答", self.identity)

        buf = self.eye.observer_buffers.get_buffer(self.identity)
        assert buf.is_sealed

    def test_flush_session(self):
        """显式 flush 返回 payload"""
        self.eye.ingest_user("问题", self.identity)
        self.eye.ingest_assistant("回答", self.identity)

        payload = self.eye.flush_session(self.identity)

        assert payload is not None
        assert payload.user_message == "问题"

    def test_flush_idle_sessions(self):
        """委托给 ObserverBufferManager"""
        self.eye.ingest_user("消息", self.identity)
        self.eye.ingest_assistant("回复", self.identity)
        # 手动设置超时
        buf = self.eye.observer_buffers.get_buffer(self.identity)
        buf._last_activity = 0.0

        payloads = self.eye.flush_idle_sessions(timeout_seconds=1.0)
        assert len(payloads) == 1


class TestTheEyeIdleMonitor:
    """idle monitor 测试"""

    def setup_method(self):
        self.mock_engine = Mock()
        self.eye = TheEye(engine=self.mock_engine, bus=None)

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_start_idle_monitor(self, MockScheduler):
        """启动调度器"""
        mock_sched = MockScheduler.return_value
        self.eye.start_observer_idle_monitor(timeout_seconds=10.0)

        MockScheduler.assert_called_once()
        mock_sched.add_job.assert_called_once()
        mock_sched.start.assert_called_once()

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_start_idle_monitor_double_guard(self, MockScheduler):
        """重复启动不创建新调度器"""
        self.eye.start_observer_idle_monitor()
        self.eye.start_observer_idle_monitor()

        assert MockScheduler.call_count == 1

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_stop_idle_monitor(self, MockScheduler):
        """停止调度器"""
        mock_sched = MockScheduler.return_value
        self.eye.start_observer_idle_monitor()
        self.eye.stop_observer_idle_monitor()

        mock_sched.shutdown.assert_called_once_with(wait=False)
        assert self.eye._observer_idle_scheduler is None

    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_lazy_start_on_first_message(self, MockScheduler):
        """惰性启动：收到消息后才启动调度器"""
        mock_sched = MockScheduler.return_value
        self.mock_engine.process.return_value = _make_gateway_result()
        self.eye.start_observer_idle_monitor(lazy_start=True)

        MockScheduler.assert_not_called()
        self.eye.ingest_user("消息", _make_identity())

        MockScheduler.assert_called_once()
        mock_sched.start.assert_called_once()

    def test_scan_auto_stop_after_global_idle(self):
        """全局无新消息超时后 flush 并自动停表"""
        mock_bus = Mock()
        eye = TheEye(engine=self.mock_engine, bus=mock_bus)
        self.mock_engine.process.return_value = _make_gateway_result()
        eye._observer_idle_monitor_enabled = True
        eye._observer_idle_scheduler = Mock()
        eye._observer_idle_timeout = 9999.0
        eye._observer_monitor_idle_shutdown_seconds = 1.0

        identity = _make_identity()
        eye.ingest_user("消息", identity)
        eye.ingest_assistant("回复", identity)
        eye._observer_last_message_ts = 0.0

        eye._scan_observer_idle_buffers()

        mock_bus.emit.assert_called_once()
        assert eye._observer_idle_scheduler is None

    def test_scan_with_bus(self):
        """有 bus 时 emit 事件"""
        mock_bus = Mock()
        eye = TheEye(engine=self.mock_engine, bus=mock_bus)
        eye._mock_engine = self.mock_engine
        self.mock_engine.process.return_value = _make_gateway_result()

        identity = _make_identity()
        eye.ingest_user("消息", identity)
        eye.ingest_assistant("回复", identity)
        buf = eye.observer_buffers.get_buffer(identity)
        buf._last_activity = 0.0
        eye._observer_idle_timeout = 1.0

        eye._scan_observer_idle_buffers()

        mock_bus.emit.assert_called_once()
        call_kwargs = mock_bus.emit.call_args
        assert call_kwargs[0][0] == "observer.idle_flushed"

    def test_scan_with_callback(self):
        """无 bus 时调用 callback"""
        eye = TheEye(engine=self.mock_engine, bus=None)
        self.mock_engine.process.return_value = _make_gateway_result()
        mock_cb = Mock()
        eye._on_flush_callback = mock_cb

        identity = _make_identity()
        eye.ingest_user("消息", identity)
        eye.ingest_assistant("回复", identity)
        buf = eye.observer_buffers.get_buffer(identity)
        buf._last_activity = 0.0
        eye._observer_idle_timeout = 1.0

        eye._scan_observer_idle_buffers()

        mock_cb.assert_called_once()
