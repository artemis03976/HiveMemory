"""
Perception 层集成测试

测试感知层与各组件的协作:
- 与 Adsorber 的协作
- 与 Generation Engine 的协作
- 与 Buffer Manager 的协作

Note:
    Phase 4.5 重构：PerceptionLayer 方法改为使用 topic_id
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from hivememory.core.models import Identity, TurnEvent, TurnRecord
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.models import (
    BufferState,
    LogicalBlock,
    SemanticBuffer,
)
from hivememory.engines.perception.semantic_flow_perception_layer import SemanticFlowPerceptionLayer
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.core.protocol import InteractionPayload


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


class TestAdsorberAndBufferCollaboration:
    """测试感知层与 Buffer Manager 的协作"""

    def test_adsorber_computes_buffer_kernel(self):
        """测试 Buffer Kernel 计算"""
        manager = SemanticBufferManager()

        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建 buffer
        buffer = manager.create_buffer(user_id=identity.user_id, title="新建话题")

        # 验证 buffer 创建
        assert buffer is not None
        assert buffer.user_id == identity.user_id
        assert buffer.topic_id is not None

    def test_adsorber_detects_topic_shift(self):
        """测试话题切换检测"""
        manager = SemanticBufferManager()
        identity = Identity(user_id="user1", agent_id="agent1")

        # 创建两个话题
        buf1 = manager.create_buffer(user_id=identity.user_id, title="Topic 1")
        buf2 = manager.create_buffer(user_id=identity.user_id, title="Topic 2")

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

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
        )
        if on_flush_callback is not None:
            perception.set_generation_callback(on_flush_callback)
        return perception

    @pytest.mark.asyncio
    async def test_semantic_flow_initialization(self):
        """测试语义流初始化"""
        perception = self._create_perception()
        assert perception is not None

    @pytest.mark.asyncio
    async def test_semantic_flow_route_and_ingest(self):
        """测试语义流路由和摄入"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 验证话题创建
        snapshots = perception.get_active_topics_snapshots(identity)
        assert len(snapshots) == 1
        assert snapshots[0].title == "新建话题"

    @pytest.mark.asyncio
    async def test_semantic_flow_buffer_info(self):
        """测试语义流 Buffer 信息获取"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("测试消息", "测试回复", identity)
        )

        # 获取菜单以获取 topic_id
        snapshots = perception.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        # 通过 topic_id 获取信息
        info = perception.get_buffer_info(topic_id)

        assert info['exists'] is True

    @pytest.mark.asyncio
    async def test_semantic_flow_manual_trigger(self):
        """测试语义流 manual_trigger"""
        perception = self._create_perception()

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由到新话题并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("消息1", "回复1", identity)
        )

        # 获取 topic_id
        snapshots = perception.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        result = await perception.manual_trigger(topic_id)
        assert result["success"] is True
        assert result["topic_id"] == topic_id
        assert result["blocks_archived"] == 1


class TestPerceptionAndGenerationCollaboration:
    """测试感知层与生成层的协作"""

    @pytest.mark.asyncio
    async def test_messages_converted_to_stream_messages(self):
        """测试消息转换为 StreamMessage"""
        archive_payloads = []

        async def on_generate(payload):
            archive_payloads.append(payload)

        config = SemanticFlowPerceptionConfig()
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary.return_value = ""

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
        )
        perception.set_generation_callback(on_generate)

        identity = Identity(user_id="test_user", agent_id="test_agent")

        # 路由并摄入
        await perception.route_and_ingest(
            "NEW_TOPIC",
            _make_payload("用户消息", "助手回复", identity)
        )

        # 获取 topic_id 并手动触发结算
        snapshots = perception.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        await perception.manual_trigger(topic_id)
        await asyncio.sleep(0)

        assert len(archive_payloads) > 0
        assert archive_payloads[0].topic_id == topic_id
        assert len(archive_payloads[0].blocks) > 0

    @pytest.mark.asyncio
    async def test_manual_trigger_archives_identity_from_payload(self):
        """场景D：manual_trigger 后归档块保留 identity 溯源"""
        archive_payloads = []

        async def on_generate(payload):
            archive_payloads.append(payload)

        config = SemanticFlowPerceptionConfig()
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary.return_value = ""

        perception = SemanticFlowPerceptionLayer(
            config=config,
            relay_controller=mock_relay,
        )
        perception.set_generation_callback(on_generate)

        identity = Identity(user_id="test_user", agent_id="reviewer_doll")
        buffer = perception._buffer_manager.create_buffer(
            user_id=identity.user_id,
            title="新建话题",
        )
        topic_id = buffer.topic_id
        perception._buffer_manager.set_last_active_topic(topic_id)
        await perception.ingest_payload(
            _make_payload("请 review 一下上面的代码", "建议补充边界条件测试", identity),
            topic_id=topic_id,
        )

        result = await perception.manual_trigger(topic_id)
        assert result["success"] is True
        await asyncio.sleep(0)

        assert len(archive_payloads) > 0
        archived_blocks = archive_payloads[0].blocks
        assert len(archived_blocks) == 1
        assert archived_blocks[0].identity.agent_id == "reviewer_doll"


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
        block = LogicalBlock(
            turn=TurnRecord(
                user_query="Test query",
                assistant_final_text="Test response",
            ),
            total_tokens=0,
        )

        from hivememory.utils.token_estimator import estimate_tokens
        block.total_tokens = (
            estimate_tokens(block.user_query) +
            estimate_tokens(block.assistant_final_text)
        )

        assert block.total_tokens > 0
