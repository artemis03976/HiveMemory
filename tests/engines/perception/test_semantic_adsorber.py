"""
SemanticBoundaryAdsorber 单元测试

测试覆盖:
- 相似度计算 (Mock Embedding)
- 吸附判定逻辑（短文本、语义漂移）
- 话题核心向量计算 (EMA)
- v3.0 新增: 无状态服务
- v3.0 新增: 返回 FlushEvent

Note:
    v3.0 重构：
    - should_adsorb() 返回 Optional[FlushEvent]
    - compute_new_topic_kernel() 是纯函数，返回新向量
    - 移除内部状态 (_anchor_cache)
"""

import pytest
from unittest.mock import Mock, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.perception.semantic_adsorber import (
    SemanticBoundaryAdsorber,
    DEFAULT_SHORT_TEXT_STOP_WORDS,
)
from hivememory.patchouli.config import SemanticAdsorberConfig
from hivememory.engines.perception.models import (
    FlushEvent,
    LogicalBlock,
    SemanticBuffer,
    FlushReason,
)
from hivememory.core.models import StreamMessage, StreamMessageType
from hivememory.engines.perception.grey_area_arbiter import NoOpArbiter


class TestSemanticBoundaryAdsorberBasic:
    """测试语义边界吸附器基础功能"""

    def setup_method(self):
        """每个测试前的设置"""
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            short_text_threshold=10,
            ema_alpha=0.5,
            enable_arbiter=False,
        )
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        # Mock embedding service
        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 1.0

    def test_should_adsorb_returns_none_for_adsorb(self):
        """测试吸附时返回 None"""
        buffer = self._create_buffer_with_kernel([1.0, 0.0])
        new_block = self._create_complete_block("相关话题", 100)

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is None  # None 表示继续吸附

    def test_should_adsorb_returns_flush_event_for_drift(self):
        """测试语义漂移时返回 FlushEvent"""
        buffer = self._create_buffer_with_kernel([0.0, 1.0])
        new_block = self._create_complete_block("不相关话题", 100)

        # 模拟相似度为 0（低于低阈值）
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 0.0

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is not None
        assert isinstance(result, FlushEvent)
        assert result.flush_reason == FlushReason.SEMANTIC_DRIFT
        assert result.triggered_by_block is new_block

    def test_short_text_adsorb(self):
        """测试短文本强吸附"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.topic_kernel_vector = None

        new_block = self._create_complete_block("ok", 2)  # 停用词

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is None  # 短文本强吸附，返回 None

    def test_incomplete_block_returns_none(self):
        """测试未闭合的 Block 返回 None"""
        buffer = Mock(spec=SemanticBuffer)

        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = False

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is None

    def _create_buffer_with_kernel(self, kernel_vector: list) -> SemanticBuffer:
        """创建带有话题核心的 buffer"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = kernel_vector

        # 添加一个 block
        block = LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content="初始话题"
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="回复"
            )
        )
        buffer.blocks.append(block)

        return buffer

    def _create_complete_block(self, content: str, total_tokens: int) -> LogicalBlock:
        """创建已闭合的 LogicalBlock"""
        return LogicalBlock(
            user_block=StreamMessage(
                message_type=StreamMessageType.USER,
                content=content
            ),
            response_block=StreamMessage(
                message_type=StreamMessageType.ASSISTANT,
                content="回复内容"
            ),
            total_tokens=total_tokens,
            rewritten_query=content,
        )


class TestSemanticBoundaryAdsorberThresholds:
    """测试双阈值系统"""

    def setup_method(self):
        """每个测试前的设置"""
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            enable_arbiter=False,
        )
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]

    def test_high_threshold_adsorb(self):
        """测试高阈值强吸附 (sim >= 0.75)"""
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 0.80

        buffer = self._create_buffer_with_kernel([1.0, 0.0])
        new_block = self._create_complete_block("相关话题", 100)

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is None  # 高相似度，继续吸附

    def test_low_threshold_split(self):
        """测试低阈值强制切分 (sim < 0.40)"""
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 0.30

        buffer = self._create_buffer_with_kernel([0.0, 1.0])
        new_block = self._create_complete_block("不相关话题", 100)

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is not None
        assert result.flush_reason == FlushReason.SEMANTIC_DRIFT

    def test_grey_area_default_adsorb(self):
        """测试灰度区间默认吸附 (0.40 <= sim < 0.75, 无仲裁器)"""
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 0.55

        buffer = self._create_buffer_with_kernel([1.0, 0.0])
        new_block = self._create_complete_block("可能相关", 100)

        result = self.adsorber.should_adsorb(buffer, new_block)

        # 无仲裁器时，灰度区间默认吸附
        assert result is None

    def _create_buffer_with_kernel(self, kernel_vector: list) -> SemanticBuffer:
        """创建带有话题核心的 buffer"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = kernel_vector
        buffer.blocks.append(LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="初始"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复")
        ))
        return buffer

    def _create_complete_block(self, content: str, total_tokens: int) -> LogicalBlock:
        """创建已闭合的 LogicalBlock"""
        return LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content=content),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
            total_tokens=total_tokens,
            rewritten_query=content,
        )


class TestSemanticBoundaryAdsorberArbiter:
    """测试灰度仲裁"""

    def test_grey_area_with_arbiter_continue(self):
        """测试灰度区间仲裁器决定继续"""
        mock_arbiter = Mock()
        mock_arbiter.is_available.return_value = True
        mock_arbiter.should_continue_topic.return_value = True

        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            enable_arbiter=True,
        )
        adsorber = SemanticBoundaryAdsorber(
            config=config,
            arbiter=mock_arbiter,
        )
        adsorber.embedding_service = Mock()
        adsorber.embedding_service.encode.return_value = [0.7, 0.7]
        adsorber.embedding_service.compute_cosine_similarity.return_value = 0.55

        buffer = self._create_buffer_with_kernel([1.0, 0.0])
        new_block = self._create_complete_block("可能的延续", 100)

        result = adsorber.should_adsorb(buffer, new_block)

        assert result is None  # 仲裁器决定继续
        mock_arbiter.should_continue_topic.assert_called_once()

    def test_grey_area_with_arbiter_split(self):
        """测试灰度区间仲裁器决定切分"""
        mock_arbiter = Mock()
        mock_arbiter.is_available.return_value = True
        mock_arbiter.should_continue_topic.return_value = False

        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            enable_arbiter=True,
        )
        adsorber = SemanticBoundaryAdsorber(
            config=config,
            arbiter=mock_arbiter,
        )
        adsorber.embedding_service = Mock()
        adsorber.embedding_service.encode.return_value = [0.7, 0.7]
        adsorber.embedding_service.compute_cosine_similarity.return_value = 0.55

        buffer = self._create_buffer_with_kernel([1.0, 0.0])
        new_block = self._create_complete_block("可能的延续", 100)

        result = adsorber.should_adsorb(buffer, new_block)

        assert result is not None
        assert result.flush_reason == FlushReason.SEMANTIC_DRIFT

    def _create_buffer_with_kernel(self, kernel_vector: list) -> SemanticBuffer:
        """创建带有话题核心的 buffer"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = kernel_vector
        buffer.blocks.append(LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="初始"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复")
        ))
        return buffer

    def _create_complete_block(self, content: str, total_tokens: int) -> LogicalBlock:
        """创建已闭合的 LogicalBlock"""
        return LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content=content),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
            total_tokens=total_tokens,
            rewritten_query=content,
        )


class TestSemanticBoundaryAdsorberTopicKernel:
    """测试话题核心向量计算"""

    def setup_method(self):
        """每个测试前的设置"""
        config = SemanticAdsorberConfig(ema_alpha=0.5)
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]

    def test_compute_new_topic_kernel_initial(self):
        """测试首次计算话题核心"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = None  # 无现有核心

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="新话题"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
        )

        result = self.adsorber.compute_new_topic_kernel(buffer, new_block)

        assert result is not None
        assert result == [1.0, 0.0]  # 首次直接使用新向量

    def test_compute_new_topic_kernel_ema(self):
        """测试 EMA 更新话题核心"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = [0.0, 1.0]  # 现有核心

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="更新"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
        )

        # encode 返回 [1.0, 0.0]
        # alpha = 0.5
        # 期望: 0.5 * [1, 0] + 0.5 * [0, 1] = [0.5, 0.5]

        result = self.adsorber.compute_new_topic_kernel(buffer, new_block)

        assert result is not None
        assert result[0] == pytest.approx(0.5, abs=1e-6)
        assert result[1] == pytest.approx(0.5, abs=1e-6)

    def test_compute_new_topic_kernel_no_anchor(self):
        """测试无锚点文本时返回 None"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)

        new_block = LogicalBlock()  # 无 user_block，anchor_text 为空

        result = self.adsorber.compute_new_topic_kernel(buffer, new_block)

        assert result is None

    def test_compute_does_not_modify_buffer(self):
        """测试计算不修改 buffer（纯函数）"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        original_kernel = [0.0, 1.0]
        buffer.topic_kernel_vector = original_kernel.copy()

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="更新"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
        )

        self.adsorber.compute_new_topic_kernel(buffer, new_block)

        # buffer 的 topic_kernel_vector 应该保持不变
        assert buffer.topic_kernel_vector == original_kernel


class TestFlushEventContent:
    """测试 FlushEvent 内容"""

    def setup_method(self):
        """每个测试前的设置"""
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
        )
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]
        self.adsorber.embedding_service.compute_cosine_similarity.return_value = 0.0

    def test_flush_event_contains_blocks(self):
        """测试 FlushEvent 包含要刷出的 blocks"""
        identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
        buffer = SemanticBuffer(identity=identity)
        buffer.topic_kernel_vector = [0.0, 1.0]

        # 添加两个 blocks
        block1 = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="话题1"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复1"),
        )
        block2 = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="话题2"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复2"),
        )
        buffer.blocks.extend([block1, block2])

        new_block = LogicalBlock(
            user_block=StreamMessage(message_type=StreamMessageType.USER, content="新话题"),
            response_block=StreamMessage(message_type=StreamMessageType.ASSISTANT, content="回复"),
            total_tokens=100,
            rewritten_query="新话题",
        )

        result = self.adsorber.should_adsorb(buffer, new_block)

        assert result is not None
        assert len(result.blocks_to_flush) == 2
        assert block1 in result.blocks_to_flush
        assert block2 in result.blocks_to_flush
        assert result.triggered_by_block is new_block


class TestDEFAULT_SHORT_TEXT_STOP_WORDS:
    """测试默认停用词列表"""

    def test_stop_words_not_empty(self):
        """测试停用词列表不为空"""
        assert DEFAULT_SHORT_TEXT_STOP_WORDS
        assert len(DEFAULT_SHORT_TEXT_STOP_WORDS) > 10

    def test_common_stop_words_present(self):
        """测试常见停用词存在"""
        assert "ok" in DEFAULT_SHORT_TEXT_STOP_WORDS
        assert "继续" in DEFAULT_SHORT_TEXT_STOP_WORDS
        assert "不对" in DEFAULT_SHORT_TEXT_STOP_WORDS
        assert "continue" in DEFAULT_SHORT_TEXT_STOP_WORDS


class TestSemanticAdsorberConfig:
    """测试配置验证"""

    def test_invalid_thresholds(self):
        """测试无效的阈值参数"""
        with pytest.raises(ValueError):
            SemanticAdsorberConfig(
                semantic_threshold_low=0.8,
                semantic_threshold_high=0.7,  # low > high
            )

    def test_custom_stop_words(self):
        """测试自定义停用词"""
        custom_stop_words = {"custom_stop"}
        config = SemanticAdsorberConfig(stop_words=custom_stop_words)

        assert "custom_stop" in config.stop_words
