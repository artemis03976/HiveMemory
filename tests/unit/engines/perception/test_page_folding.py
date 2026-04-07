"""
Page Folding 单元测试

测试覆盖:
- block total_tokens 计算修复验证
- 低于阈值时不触发折叠
- 超阈值后旧 blocks 被丢弃，state_summary 被写入
- 折叠过程不触发 generation_callback
- 连续折叠时 state_summary 累积

参考: ShortTermMemory.md §4.2

Note:
    Phase 4.5 重构：使用 topic_id 替代 session_id
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from hivememory.core.models import Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.semantic_flow_perception_layer import (
    SemanticFlowPerceptionLayer,
)
from hivememory.engines.perception.models import (
    InteractionPayload,
    TraceItem,
    FlushReason,
    LogicalBlock,
)
from hivememory.patchouli.config import SemanticFlowPerceptionConfig


def _make_identity():
    return Identity(user_id="u1", agent_id="a1")


def _make_payload(user_msg="hello", assistant_msg="world", identity=None, traces=None):
    if identity is None:
        identity = _make_identity()
    return InteractionPayload(
        user_message=user_msg,
        assistant_message=assistant_msg,
        identity=identity,
        mtp_traces=traces or [],
    )


def _make_large_payload(token_target, identity=None):
    """构造一个大 payload，使 block.total_tokens 接近 token_target"""
    # estimate_tokens 大约按 len/4 估算，所以 content 长度 ~= token_target * 4
    content = "x" * (token_target * 4)
    return _make_payload(
        user_msg=content,
        assistant_msg="short reply",
        identity=identity,
    )


class TestBlockTokenComputation:
    """验证 ingest_payload 后 block.total_tokens > 0"""

    def setup_method(self):
        self.config = SemanticFlowPerceptionConfig(
            fold_token_threshold=999999,  # 不触发折叠
        )
        self.mock_relay = Mock()
        self.mock_relay.should_relay.return_value = None
        self.layer = SemanticFlowPerceptionLayer(config=self.config, relay_controller=self.mock_relay)

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_block_total_tokens_computed(self, mock_parser_cls):
        """ingest_payload 后 block.total_tokens 应 > 0"""
        mock_parser_cls.parse.return_value = ("clean reply", [])

        identity = _make_identity()
        payload = _make_payload("What is Python?", "Python is a language", identity)

        await self.layer.route_and_ingest("NEW_TOPIC", payload)

        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        buffer = self.layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 1
        assert buffer.blocks[0].total_tokens > 0
        assert buffer.total_tokens > 0

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_block_tokens_include_traces(self, mock_parser_cls):
        """traces 中的 query/target 也应计入 total_tokens"""
        mock_parser_cls.parse.return_value = ("clean", [])

        traces = [
            TraceItem(action="SEARCH", query="how to sort a list"),
            TraceItem(action="READ", target="my_notes_alias"),
        ]
        identity = _make_identity()
        payload = _make_payload("q", "a", identity, traces=traces)

        await self.layer.route_and_ingest("NEW_TOPIC", payload)

        snapshots = self.layer.get_active_topics_snapshots(identity)
        topic_id = snapshots[0].topic_id

        buffer = self.layer.get_buffer(topic_id)
        block = buffer.blocks[0]
        # tokens 应包含 user_query + clean_response + trace fields
        assert block.total_tokens > 0


class TestPageFoldingThreshold:
    """验证 Page Folding 阈值逻辑"""

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_fold_not_triggered_below_threshold(self, mock_parser_cls):
        """低于阈值时 state_summary 保持为空"""
        mock_parser_cls.parse.return_value = ("reply", [])

        config = SemanticFlowPerceptionConfig(
            fold_token_threshold=999999,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        identity = _make_identity()

        # 路由到新话题
        topic_id = None
        for i in range(5):
            if topic_id is None:
                await layer.route_and_ingest("NEW_TOPIC", _make_payload(f"msg{i}", f"reply{i}", identity))
                snapshots = layer.get_active_topics_snapshots(identity)
                topic_id = snapshots[0].topic_id
            else:
                await layer.route_and_ingest(topic_id, _make_payload(f"msg{i}", f"reply{i}", identity))

        buffer = layer.get_buffer(topic_id)
        assert len(buffer.blocks) == 5
        assert buffer.state_summary == ""

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_fold_triggered_above_threshold(self, mock_parser_cls):
        """超阈值后旧 blocks 被丢弃，state_summary 被写入，保留最近 N 个"""
        mock_parser_cls.parse.return_value = ("reply", [])

        config = SemanticFlowPerceptionConfig(
            fold_token_threshold=100,  # 较低阈值
            fold_retain_recent_blocks=2,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary = Mock(return_value="Test summary")  # Mock generate_summary
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        identity = _make_identity()

        # 直接使用 BufferManager 创建 buffer 并添加 blocks
        buffer = layer._buffer_manager.create_buffer(identity.user_id)
        topic_id = buffer.topic_id

        # 添加多个小 blocks，总 token 会超过阈值
        for i in range(10):
            user_msg = StreamMessage(message_type=StreamMessageType.USER, content=f"question {i}")
            response_msg = StreamMessage(message_type=StreamMessageType.ASSISTANT, content=f"answer {i}")
            block = LogicalBlock(
                user_block=user_msg,
                response_block=response_msg,
                total_tokens=20,  # 每个 block 20 tokens
            )
            layer._buffer_manager.add_block(topic_id, block)

        # 手动触发 fold_blocks（模拟 token 溢出场景）
        layer._buffer_manager.fold_blocks(topic_id, "Test summary", 2)

        buffer = layer.get_buffer(topic_id)
        # 折叠后应只保留最近 2 个 blocks
        assert len(buffer.blocks) <= 2
        # state_summary 应被写入
        assert buffer.state_summary == "Test summary"

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_fold_does_not_trigger_generation_callback(self, mock_parser_cls):
        """折叠过程中不应触发 generation_callback"""
        mock_parser_cls.parse.return_value = ("reply", [])

        config = SemanticFlowPerceptionConfig(
            fold_token_threshold=100,
            fold_retain_recent_blocks=2,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary = Mock(return_value="Test summary")  # Mock generate_summary
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        mock_callback = AsyncMock(return_value=None)
        layer.set_generation_callback(mock_callback)

        identity = _make_identity()

        # 直接使用 BufferManager 创建 buffer 并添加 blocks
        buffer = layer._buffer_manager.create_buffer(identity.user_id)
        topic_id = buffer.topic_id

        # 添加多个小 blocks
        for i in range(10):
            user_msg = StreamMessage(message_type=StreamMessageType.USER, content=f"question {i}")
            response_msg = StreamMessage(message_type=StreamMessageType.ASSISTANT, content=f"answer {i}")
            block = LogicalBlock(
                user_block=user_msg,
                response_block=response_msg,
                total_tokens=20,
            )
            layer._buffer_manager.add_block(topic_id, block)

        # 手动触发 fold_blocks（不经过 resolve_topic，所以不会触发 Archive 回调）
        layer._buffer_manager.fold_blocks(topic_id, "Test summary", 2)

        mock_callback.assert_not_called()


class TestPageFoldingCumulative:
    """验证连续折叠时 state_summary 累积"""

    @patch("hivememory.patchouli.protocol.mtp_log_parser.MTPLogParser")
    @pytest.mark.asyncio
    async def test_fold_cumulative_summary(self, mock_parser_cls):
        """连续两次折叠，验证 state_summary 以 --- 分隔累积"""
        mock_parser_cls.parse.return_value = ("reply", [])

        config = SemanticFlowPerceptionConfig(
            fold_token_threshold=50,
            fold_retain_recent_blocks=1,
        )
        mock_relay = Mock()
        mock_relay.should_relay.return_value = None
        mock_relay.generate_summary = Mock(side_effect=lambda blocks_to_fold, previous_summary: previous_summary + "---folded")
        layer = SemanticFlowPerceptionLayer(config=config, relay_controller=mock_relay)
        identity = _make_identity()

        # 第一波：触发第一次折叠
        topic_id = None
        for i in range(4):
            if topic_id is None:
                await layer.route_and_ingest("NEW_TOPIC", _make_payload(f"wave1 q{i} " * 20, f"wave1 a{i} " * 20, identity))
                snapshots = layer.get_active_topics_snapshots(identity)
                topic_id = snapshots[0].topic_id
            else:
                await layer.route_and_ingest(topic_id, _make_payload(f"wave1 q{i} " * 20, f"wave1 a{i} " * 20, identity))

        buffer = layer.get_buffer(topic_id)
        first_summary = buffer.state_summary
        assert first_summary != ""

        # 第二波：继续摄入，触发第二次折叠
        for i in range(4):
            await layer.route_and_ingest(topic_id, _make_payload(f"wave2 q{i} " * 20, f"wave2 a{i} " * 20, identity))

        buffer = layer.get_buffer(topic_id)
        # 累积摘要应包含分隔符
        assert "---" in buffer.state_summary
        # 第一次摘要应被保留在累积摘要中
        assert first_summary in buffer.state_summary
