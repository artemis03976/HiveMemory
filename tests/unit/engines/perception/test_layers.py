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
)
from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.core.protocol import InteractionPayload


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

        # 路由到新话题并摄入，topic_id 由路由入口直接返回。
        topic_id = await self.layer.route_and_ingest("NEW_TOPIC", payload)

        # Verify: block 应该已完成并加入话题数据
        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1
        assert topic_data.blocks[0].identity.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_route_and_ingest_reuses_topic_buffer(self):
        """测试同一 topic_id 下继续写入同一 buffer"""
        identity = Identity(user_id="u1", agent_id="a1")

        # 第一轮：路由到新话题
        topic_id = await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("old topic", "old response", identity))

        # 验证第一个 block 已加入
        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1

        # 第二轮：继续摄入（MMU 模式下话题路由由 TheEye 完成，ingest_payload 只做添加）
        await self.layer.route_and_ingest(topic_id, _make_payload("new topic", "new response", identity))

        # 验证两个 block 都在同一话题中（无自动漂移检测）
        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 2

    @pytest.mark.asyncio
    async def test_token_overflow_relay(self):
        """测试 Token 溢出 (Phase 4.5: Relay 已断开，仅验证多 block 累积)"""
        identity = Identity(user_id="u1", agent_id="a1")

        topic_id = await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("first", "response1", identity))

        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1

        # 第二轮：继续摄入（Relay 已断开，不再触发 Token 溢出 flush）
        await self.layer.route_and_ingest(topic_id, _make_payload("second", "response2", identity))

        # 验证两个 block 都在话题数据中
        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 2

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

        topic_id = await self.layer.route_and_ingest("NEW_TOPIC", _make_payload("hi", "hello", identity))

        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 1

        # 清理
        result = self.layer._short_term_store.clear_buffer(topic_id)
        assert result is not None  # clear_buffer returns cleared blocks list

        # 验证话题 blocks 已清空
        topic_data = self.layer._short_term_store.get_topic_data(topic_id, touch=False)
        assert topic_data is not None
        assert len(topic_data.blocks) == 0
