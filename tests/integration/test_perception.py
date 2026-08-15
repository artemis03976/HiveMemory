"""
Perception 层集成测试

测试感知层与各组件的协作:
- 与 Generation Engine 的协作
- 与 Buffer Manager 的协作

Note:
    Phase 4.5 重构：PerceptionLayer 方法改为使用 topic_id
"""

from unittest.mock import Mock

import pytest

from hivememory.core.models import Identity, LogicalBlock, TurnEvent, TurnRecord
from hivememory.core.protocol import InteractionPayload
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import SemanticFlowPerceptionConfig


def _make_payload(user_msg: str, assistant_msg: str, identity: Identity) -> InteractionPayload:
    """辅助方法：创建测试用 Payload"""
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


class TestShortTermMemoryStoreCollaboration:
    """测试感知层与短期记忆 Store 的协作"""

    def test_short_term_store_creates_topic(self):
        """测试短期话题创建"""
        manager = ShortTermMemoryStore()

        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建 buffer
        buffer = manager.create_buffer(user_id=identity.user_id, topic_title="新建话题")

        # 验证 buffer 创建
        assert buffer is not None
        assert buffer.user_id == identity.user_id
        assert buffer.topic_id is not None

    def test_short_term_store_keeps_topics_isolated(self):
        """测试话题隔离"""
        manager = ShortTermMemoryStore()
        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建两个话题
        buf1 = manager.create_buffer(user_id=identity.user_id, topic_title="Topic 1")
        buf2 = manager.create_buffer(user_id=identity.user_id, topic_title="Topic 2")

        # 验证话题隔离
        assert buf1.topic_id != buf2.topic_id
        assert buf1.user_id == buf2.user_id  # 同一个用户


class TestSemanticFlowPerceptionLayerOrchestration:
    """测试语义流感知层的编排"""

    def _create_perception(self, on_flush_callback=None):
        """辅助方法：创建感知层实例"""
        config = SemanticFlowPerceptionConfig()

        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary.return_value = ""

        short_term_store = ShortTermMemoryStore()

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
            short_term_store=short_term_store,
            interaction_journal=InMemoryInteractionApplyJournal(),
        )
        if on_flush_callback is not None:
            perception.set_generation_callback(on_flush_callback)
        return perception

    @pytest.mark.asyncio
    async def test_semantic_flow_route_and_ingest(self):
        """测试语义流路由和摄入"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        topic_id, _ = await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 验证话题创建
        topic_data = perception._short_term_store.get_topic_data(topic_id)
        assert topic_data is not None
        assert topic_data.topic_title == "新建话题"

    @pytest.mark.asyncio
    async def test_semantic_flow_buffer_info(self):
        """测试语义流 Buffer 信息获取"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题
        topic_id, _ = await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 通过短期 Store 获取话题信息
        info = perception._short_term_store.get_buffer_info(topic_id)

        assert info['exists'] is True

    @pytest.mark.asyncio
    async def test_semantic_flow_manual_trigger(self):
        """测试语义流 settle_topic"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        topic_id, _ = await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("消息1", "回复1", identity)
        )

        result = await perception.settle_topic(topic_id)
        assert result.topic_id == topic_id


class TestTokenManagement:
    """测试 Token 管理"""

    def test_token_estimation(self):
        """测试 Token 估算"""
        from hivememory.utils.token_estimator import estimate_tokens

        text = "Hello, world!"
        tokens = estimate_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_block_token_count(self):
        """测试 Block Token 计数"""
        from hivememory.utils.token_estimator import estimate_tokens

        user_query = "Test query"
        assistant_final_text = "Test response"
        block = LogicalBlock(
            turn=TurnRecord(
                user_query=user_query,
                assistant_final_text=assistant_final_text,
            ),
            total_tokens=(
                estimate_tokens(user_query)
                + estimate_tokens(assistant_final_text)
            ),
        )

        assert block.total_tokens > 0
