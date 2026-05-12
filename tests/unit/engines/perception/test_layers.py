"""
感知层 (Perception Layers) 单元测试

测试覆盖:
- SemanticFlowPerceptionLayer:
    - ingest_payload 载荷摄入流程
    - Block 管理
    - Flush 回调

Note:
    v3.0 重构：
    - perceive() 已移除，统一使用 ingest_payload()
    - SemanticFlowPerceptionLayer 不再依赖 UnifiedStreamParser
    - Phase 4.5: 移除 SimplePerceptionLayer，仅保留 SemanticFlowPerceptionLayer
    - Phase 4.5 重构：使用 topic_id 替代 session_id
"""

import pytest
from unittest.mock import Mock, MagicMock

from hivememory.core.models import Identity, TurnEvent
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.engines.perception.models import (
    FlushEvent,
    SemanticBuffer,
    LogicalBlock,
    FlushReason,
    InteractionPayload,
)
from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.patchouli.config import SemanticFlowPerceptionConfig


def _make_payload(user_msg="msg", assistant_msg="reply", identity=None):
    """辅助: 构建 InteractionPayload"""
    if identity is None:
        identity = Identity(user_id="u1", agent_id="a1")
    return InteractionPayload(
        user_message=user_msg,
        assistant_final_text=assistant_msg,
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content=assistant_msg,
            )
        ],
        identity=identity,
    )


class TestSemanticFlowPerceptionLayer:
    """测试语义流感知层"""

    def setup_method(self):
        self.mock_relay = Mock()
        self.mock_callback = Mock()

        self.config = SemanticFlowPerceptionConfig()

        # v3.0: should_relay 返回 None 表示不需要接力
        self.mock_relay.should_relay.return_value = None

        self.layer = SemanticFlowPerceptionLayer(
            config=self.config,
            relay_controller=self.mock_relay,
        )
        self.layer.set_generation_callback(self.mock_callback)

    @pytest.mark.asyncio
    async def test_process_new_block_flow(self):
        """测试 ingest_payload 新 Block 处理流程"""
        identity = Identity(user_id="u1", agent_id="a1")
        payload = _make_payload("hi", "hello", identity)

        # 路由到新话题并摄入
        await self.layer.route_and_ingest("NEW_TOPIC", payload)

        # 获取 topic_id
        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        # Verify: block 应该已完成并加入 buffer
        buffer = self.layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0].identity.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_semantic_drift_flush(self):
        """测试话题路由 (Phase 4.5 MMU: 由 TheEye 路由替代 Adsorber 漂移检测)"""
        identity = Identity(user_id="u1", agent_id="a1")

        # 第一轮：路由到新话题
        await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("old topic", "old response", identity))

        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        # 验证第一个 block 已加入
        buffer = self.layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 1

        # 第二轮：继续摄入（MMU 模式下话题路由由 TheEye 完成，ingest_payload 只做添加）
        await self.layer.route_and_ingest(topic_id, _make_payload("new topic", "new response", identity))

        # 验证两个 block 都在同一 buffer 中（无漂移检测）
        assert len(buffer.blocks) == 2

    @pytest.mark.asyncio
    async def test_token_overflow_relay(self):
        """测试 Token 溢出 (Phase 4.5: Relay 已断开，仅验证多 block 累积)"""
        identity = Identity(user_id="u1", agent_id="a1")

        await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("first", "response1", identity))

        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        buffer = self.layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 1

        # 第二轮：继续摄入（Relay 已断开，不再触发 Token 溢出 flush）
        await self.layer.route_and_ingest(topic_id, _make_payload("second", "response2", identity))

        # 验证两个 block 都在 buffer 中
        assert len(buffer.blocks) == 2

    @pytest.mark.asyncio
    async def test_manual_trigger_without_active_topic(self):
        """测试无活跃话题时 manual_trigger 抛出异常"""
        with pytest.raises(ValueError):
            await self.layer.manual_trigger()
        self.mock_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        """测试清理 buffer"""
        identity = Identity(user_id="u1", agent_id="a1")

        await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("hi", "hello", identity))

        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        buffer = self.layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 1

        # 清理
        result = self.layer.clear_buffer(topic_id)
        assert result is True

        # 验证 buffer 已清空
        assert len(buffer.blocks) == 0
        assert buffer.topic_kernel_vector is None
