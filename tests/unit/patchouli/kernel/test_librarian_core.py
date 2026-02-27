"""
LibrarianCore 单元测试

测试覆盖:
- 初始化: 有/无 bus 时的订阅行为
- 观察者: 添加 / 移除 / 移除不存在的
- ingest_interaction: 委托给 bus
- _on_perception_flush: Mode A/B/C 分支 / 空消息 / 异常隔离
"""

import pytest
from unittest.mock import Mock, call
from uuid import uuid4

from hivememory.core.models import StreamMessage, StreamMessageType, Identity
from hivememory.engines.perception.models import FlushReason, InteractionPayload
from hivememory.engines.generation.models import GenerationRequest, WriteFocus
from hivememory.patchouli.kernel.librarian_core import LibrarianCore


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1", session_id="s1")


def _make_messages(n=2):
    identity = _make_identity()
    msgs = []
    for i in range(n):
        msg_type = StreamMessageType.USER if i % 2 == 0 else StreamMessageType.ASSISTANT
        msgs.append(StreamMessage(message_type=msg_type, content=f"msg_{i}", identity=identity))
    return msgs


class TestLibrarianCoreInit:
    """初始化测试"""

    def test_init_with_bus_subscribes(self):
        """有 bus 时订阅 perception.flushed"""
        mock_bus = Mock()
        core = LibrarianCore(storage=Mock(), bus=mock_bus)

        mock_bus.subscribe.assert_called_once_with(
            "perception.flushed", core._on_perception_flush
        )

    def test_init_without_bus(self):
        """无 bus 时不报错"""
        core = LibrarianCore(storage=Mock(), bus=None)
        assert core._bus is None

class TestLibrarianCoreObservers:
    """观察者管理测试"""

    def setup_method(self):
        self.core = LibrarianCore(storage=Mock(), bus=None)

    def test_add_flush_observer(self):
        """添加观察者"""
        observer = Mock()
        self.core.add_flush_observer(observer)
        assert observer in self.core._flush_observers

    def test_remove_flush_observer(self):
        """移除观察者"""
        observer = Mock()
        self.core.add_flush_observer(observer)
        self.core.remove_flush_observer(observer)
        assert observer not in self.core._flush_observers

    def test_remove_nonexistent_observer(self):
        """移除不存在的观察者不报错"""
        self.core.remove_flush_observer(Mock())


class TestLibrarianCoreIngest:
    """ingest_interaction 测试"""

    def test_ingest_interaction_delegates_to_bus(self):
        """委托给 bus.request"""
        mock_bus = Mock()
        core = LibrarianCore(storage=Mock(), bus=mock_bus)
        payload = Mock(spec=InteractionPayload)
        payload.user_message = "test message for logging"
        payload.mtp_traces = []
        payload.write_focus = None
        payload.update_focus = None

        core.ingest_interaction(payload, target_topic="topic_1")

        mock_bus.request.assert_called_with(
            "perception.route_and_ingest", "topic_1", payload
        )


class TestLibrarianCoreFlushCallback:
    """_on_perception_flush 回调测试"""

    def setup_method(self):
        self.mock_bus = Mock()
        self.mock_storage = Mock()
        self.core = LibrarianCore(
            storage=self.mock_storage, bus=self.mock_bus
        )

    def test_flush_mode_a_default(self):
        """普通 flush，构建 GenerationRequest(context_messages=msgs)"""
        msgs = _make_messages(2)
        self.mock_bus.request.return_value = []

        self.core._on_perception_flush(msgs, FlushReason.MESSAGE_COUNT)

        # 第二次 request 调用是 generation.process（第一次是 subscribe）
        gen_call = None
        for c in self.mock_bus.request.call_args_list:
            if c[0][0] == "generation.process":
                gen_call = c
                break
        assert gen_call is not None
        request = gen_call[0][1]
        assert isinstance(request, GenerationRequest)
        assert len(request.context_messages) == 2
        assert request.write_focus is None
        assert request.update_focus is None

    def test_flush_mode_b_write(self):
        """MTP_WRITE flush，构建带 write_focus 的 request"""
        msgs = _make_messages(2)
        write_focus = WriteFocus(content="测试写入内容")
        self.mock_bus.request.return_value = []

        self.core._on_perception_flush(
            msgs, FlushReason.MTP_WRITE, write_focus=write_focus
        )

        gen_call = None
        for c in self.mock_bus.request.call_args_list:
            if c[0][0] == "generation.process":
                gen_call = c
                break
        assert gen_call is not None
        request = gen_call[0][1]
        assert request.write_focus is write_focus

    def test_flush_mode_c_update_success(self):
        """MTP_UPDATE flush，加载 existing memory 成功"""
        msgs = _make_messages(2)
        update_focus = Mock()
        update_focus.target_uuid = str(uuid4())
        update_focus.target_alias = "fact_test"
        existing_memory = Mock()

        def bus_request_side_effect(route, *args, **kwargs):
            if route == "storage.get_memory":
                return existing_memory
            return []

        self.mock_bus.request.side_effect = bus_request_side_effect

        self.core._on_perception_flush(
            msgs, FlushReason.MTP_UPDATE, update_focus=update_focus
        )

        assert update_focus.existing_memory is existing_memory

    def test_flush_mode_c_update_memory_not_found(self):
        """existing memory 不存在时 early return"""
        msgs = _make_messages(2)
        update_focus = Mock()
        update_focus.target_uuid = str(uuid4())
        update_focus.target_alias = "fact_test"

        def bus_request_side_effect(route, *args, **kwargs):
            if route == "storage.get_memory":
                return None
            return []

        self.mock_bus.request.side_effect = bus_request_side_effect

        self.core._on_perception_flush(
            msgs, FlushReason.MTP_UPDATE, update_focus=update_focus
        )

        # generation.process 不应被调用
        gen_calls = [
            c for c in self.mock_bus.request.call_args_list
            if c[0][0] == "generation.process"
        ]
        assert len(gen_calls) == 0

    def test_flush_empty_messages(self):
        """空消息列表 early return"""
        self.core._on_perception_flush([], FlushReason.MESSAGE_COUNT)

        # 不应调用 generation.process
        gen_calls = [
            c for c in self.mock_bus.request.call_args_list
            if c[0][0] == "generation.process"
        ]
        assert len(gen_calls) == 0

    def test_flush_generation_exception(self):
        """generation.process 抛异常时不崩溃"""
        msgs = _make_messages(2)

        def bus_request_side_effect(route, *args, **kwargs):
            if route == "generation.process":
                raise RuntimeError("generation failed")
            return []

        self.mock_bus.request.side_effect = bus_request_side_effect

        # 不应抛异常
        self.core._on_perception_flush(msgs, FlushReason.MESSAGE_COUNT)

    def test_flush_generation_returns_empty(self):
        """generation.process 返回空列表（正常日志，不崩溃）"""
        msgs = _make_messages(2)
        self.mock_bus.request.return_value = []

        self.core._on_perception_flush(msgs, FlushReason.MESSAGE_COUNT)
