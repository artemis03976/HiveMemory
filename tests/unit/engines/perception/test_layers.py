"""
感知层 (Perception Layers) 单元测试

测试覆盖:
- SimplePerceptionLayer:
    - ingest_payload 消息添加流程
    - 触发器调用
    - Flush 回调
- SemanticFlowPerceptionLayer:
    - ingest_payload 载荷摄入流程
    - Block 管理
    - 语义吸附流程
    - Flush 回调

Note:
    v3.0 重构：
    - perceive() 已移除，统一使用 ingest_payload()
    - SemanticFlowPerceptionLayer 不再依赖 UnifiedStreamParser
    - Adsorber.should_adsorb() 返回 Optional[FlushEvent]
    - Relay.should_relay() 返回 Optional[FlushEvent]
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.perception.simple_perception_layer import SimplePerceptionLayer
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.models import (
    FlushEvent,
    SimpleBuffer,
    SemanticBuffer,
    LogicalBlock,
    FlushReason,
    InteractionPayload,
)
from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.patchouli.config import SimplePerceptionConfig, SemanticFlowPerceptionConfig


def _make_payload(user_msg="msg", assistant_msg="reply", identity=None):
    """辅助: 构建 InteractionPayload"""
    if identity is None:
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
    return InteractionPayload(
        user_message=user_msg,
        assistant_message=assistant_msg,
        identity=identity,
    )


class TestSimplePerceptionLayer:
    """测试简单感知层"""

    def setup_method(self):
        self.mock_trigger_manager = Mock()
        self.mock_callback = Mock()
        self.config = SimplePerceptionConfig()
        self.layer = SimplePerceptionLayer(
            config=self.config,
            trigger_manager=self.mock_trigger_manager,
            on_flush_callback=self.mock_callback
        )

    def test_add_message_flow(self):
        """测试 ingest_payload 消息添加流程"""
        # ingest_payload 添加 user + assistant 后执行一次触发检查
        self.mock_trigger_manager.should_trigger.return_value = (True, FlushReason.MESSAGE_COUNT)

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        payload = _make_payload("msg1", "msg2", identity)

        self.layer.ingest_payload(payload)

        # 验证 Flush 被调用
        self.mock_callback.assert_called_once()
        args, kwargs = self.mock_callback.call_args
        messages, reason = args
        assert len(messages) == 2
        assert reason == FlushReason.MESSAGE_COUNT

    def test_flush_buffer_manual(self):
        """测试手动 Flush"""
        self.mock_trigger_manager.should_trigger.return_value = (False, None)

        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        self.layer.ingest_payload(_make_payload("msg1", "reply1", identity))

        self.layer.flush_buffer(identity)

        self.mock_callback.assert_called_once()
        args, kwargs = self.mock_callback.call_args
        messages, reason = args
        assert len(messages) == 2
        assert reason == FlushReason.MANUAL


class TestSemanticFlowPerceptionLayer:
    """测试语义流感知层"""

    def setup_method(self):
        self.mock_adsorber = Mock()
        self.mock_relay = Mock()
        self.mock_callback = Mock()

        self.config = SemanticFlowPerceptionConfig()

        # v3.0: should_adsorb 返回 None 表示继续吸附
        self.mock_adsorber.should_adsorb.return_value = None
        self.mock_adsorber.compute_new_topic_kernel.return_value = [0.1, 0.2, 0.3]

        # v3.0: should_relay 返回 None 表示不需要接力
        self.mock_relay.should_relay.return_value = None

        self.layer = SemanticFlowPerceptionLayer(
            config=self.config,
            adsorber=self.mock_adsorber,
            relay_controller=self.mock_relay,
            on_flush_callback=self.mock_callback
        )

    def test_process_new_block_flow(self):
        """测试 ingest_payload 新 Block 处理流程"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        payload = _make_payload("hi", "hello", identity)

        self.layer.ingest_payload(payload)

        # Verify: block 应该已完成并加入 buffer
        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

    def test_semantic_drift_flush(self):
        """测试话题路由 (Phase 4.5 MMU: 由 TheEye 路由替代 Adsorber 漂移检测)"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 第一轮：正常摄入
        self.layer.ingest_payload(_make_payload("old topic", "old response", identity))

        # 验证第一个 block 已加入
        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 第二轮：继续摄入（MMU 模式下话题路由由 TheEye 完成，ingest_payload 只做添加）
        self.layer.ingest_payload(_make_payload("new topic", "new response", identity))

        # 验证两个 block 都在同一 buffer 中（无漂移检测）
        assert len(buffer.blocks) == 2

    def test_token_overflow_relay(self):
        """测试 Token 溢出 (Phase 4.5: Relay 已断开，仅验证多 block 累积)"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        self.layer.ingest_payload(_make_payload("first", "response1", identity))

        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 第二轮：继续摄入（Relay 已断开，不再触发 Token 溢出 flush）
        self.layer.ingest_payload(_make_payload("second", "response2", identity))

        # 验证两个 block 都在 buffer 中
        assert len(buffer.blocks) == 2

    def test_no_flush_callback_when_no_blocks(self):
        """测试无 blocks 时不调用回调"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        # 手动 flush 空 buffer
        result = self.layer.flush_buffer(identity)

        assert result == []
        self.mock_callback.assert_not_called()

    def test_clear_buffer(self):
        """测试清理 buffer"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

        self.layer.ingest_payload(_make_payload("hi", "hello", identity))

        buffer = self.layer.get_buffer(identity)
        assert len(buffer.blocks) == 1

        # 清理
        result = self.layer.clear_buffer(identity)
        assert result is True

        # 验证 buffer 已清空
        assert len(buffer.blocks) == 0
        assert buffer.topic_kernel_vector is None
