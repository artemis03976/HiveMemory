"""
SemanticBoundaryAdsorber 单元测试

测试覆盖:
- 相似度��算 (Mock Embedding)
- 吸附判定逻辑（短文本、语义漂移）
- 话题核心向量更新 (EMA)
- v2.0 新增: 三阶段处理管道
- v2.0 新增: 双阈值系统
- v2.0 新增: 停用词检测
- v2.0 新增: 灰度仲裁
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from hivememory.core.models import FlushReason
from hivememory.engines.perception.semantic_adsorber import (
    SemanticBoundaryAdsorber,
    DEFAULT_SHORT_TEXT_STOP_WORDS,
)
from hivememory.patchouli.config import SemanticAdsorberConfig
from hivememory.engines.perception.models import LogicalBlock, SemanticBuffer, StreamMessage, StreamMessageType
from hivememory.engines.perception.grey_area_arbiter import NoOpArbiter


class TestSemanticBoundaryAdsorberLegacy:
    """测试语义边界吸附器（兼容旧版逻辑 - 单阈值模式）"""

    def setup_method(self):
        """每个测试前的设置"""
        # 使用单阈值配置模拟旧版逻辑
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.6,
            semantic_threshold_low=0.6,
            short_text_threshold=50,
            ema_alpha=0.5,
            enable_arbiter=False,
        )
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        # Mock embedding service
        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]

    def test_compute_similarity(self):
        """测试相似度计算"""
        # 向量 [1, 0] dot [1, 0] = 1.0
        similarity = self.adsorber.compute_similarity("text", [1.0, 0.0])
        assert similarity == 1.0

    def test_should_adsorb_short_text(self):
        """测试短文本强吸附"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 0

        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 10  # < 50
        new_block.anchor_text = "ok"

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True
        assert reason == FlushReason.SHORT_TEXT_ADSORB

    def test_semantic_drift(self):
        """测试语义漂移"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 100
        buffer.topic_kernel_vector = [0.0, 1.0]  # 与 Mock 的 [1.0, 0.0] 正交，sim=0
        buffer.blocks = []  # 修复: ContextBridge 需要访问 blocks
        buffer.relay_summary = None  # 修复: ContextBridge 需要访问 relay_summary

        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 100  # > 50
        new_block.anchor_text = "new topic"
        new_block.rewritten_query = None

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.SEMANTIC_DRIFT

    def test_update_topic_kernel(self):
        """测试话题核心更新 (EMA)"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.buffer_id = "test"
        buffer.topic_kernel_vector = [0.0, 1.0]
        buffer.blocks = []  # ContextBridge 需要
        buffer.relay_summary = None

        new_block = Mock(spec=LogicalBlock)
        new_block.block_id = "b1"
        new_block.anchor_text = "update"
        new_block.rewritten_query = None

        # Mock encode returns [1.0, 0.0]
        # Alpha = 0.5
        # Expected = 0.5*[1,0] + 0.5*[0,1] = [0.5, 0.5]

        self.adsorber.update_topic_kernel(buffer, new_block)

        updated = buffer.topic_kernel_vector
        # 结果应该是 [0.5, 0.5]
        assert updated[0] == pytest.approx(0.5, abs=1e-6)
        assert updated[1] == pytest.approx(0.5, abs=1e-6)


class TestSemanticBoundaryAdsorberV2:
    """测试 v2.0 三阶段处理管道"""

    def setup_method(self):
        """每个测试前的设置"""
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            enable_arbiter=True,
        )
        self.adsorber = SemanticBoundaryAdsorber(config=config)

        # Mock embedding service
        self.adsorber.embedding_service = Mock()
        self.adsorber.embedding_service.encode.return_value = [1.0, 0.0]

    def test_high_threshold_adsorb(self):
        """测试高阈值强吸附"""
        buffer = self._create_buffer_with_kernel([1.0, 0.0])  # sim = 1.0

        new_block = self._create_complete_block(
            anchor_text="相关话题",
            rewritten_query="相关话题",
            total_tokens=100
        )

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True
        assert reason is None

    def test_low_threshold_split(self):
        """测试低阈值强制切分"""
        buffer = self._create_buffer_with_kernel([0.0, 1.0])  # sim = 0.0

        new_block = self._create_complete_block(
            anchor_text="不相关话题",
            rewritten_query="不相关话题",
            total_tokens=100
        )

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.SEMANTIC_DRIFT

    def test_stop_word_detection(self):
        """测试停用词检测"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.topic_kernel_vector = None  # 无话题核心

        # 停用词（token < 10 且在停用词列表中）
        new_block = self._create_complete_block(
            anchor_text="ok",
            rewritten_query="ok",
            total_tokens=2  # < short_text_threshold (default 10)
        )

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True
        assert reason == FlushReason.SHORT_TEXT_ADSORB

    def test_stop_word_not_detected_with_more_tokens(self):
        """测试 token 数较多时不停用停用词检测"""
        buffer = self._create_buffer_with_kernel([1.0, 0.0])

        # 即使是停用词，但 token 数较多，不会触发停用词检测
        new_block = self._create_complete_block(
            anchor_text="ok",
            rewritten_query="ok",
            total_tokens=15  # > short_text_threshold (default 10)
        )

        # 应该走正常的相似度判定
        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True  # sim = 1.0 > 高阈值

    def test_grey_area_with_arbiter_continue(self):
        """测试灰度区间仲裁器决定继续"""
        # 创建仲裁器 mock，返回 True（继续）
        mock_arbiter = Mock(spec=NoOpArbiter)
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
        adsorber.embedding_service.encode.return_value = [0.7, 0.7]  # 归一化后

        # sim 约 0.5（灰度区间）
        buffer = self._create_buffer_with_kernel([1.0, 0.0])

        new_block = self._create_complete_block(
            anchor_text="可能的延续",
            rewritten_query="可能的延续",
            total_tokens=100
        )

        should, reason = adsorber.should_adsorb(new_block, buffer)
        assert should is True
        mock_arbiter.should_continue_topic.assert_called_once()

    def test_grey_area_with_arbiter_split(self):
        """测试灰度区间仲裁器决定切分"""
        # 创建仲裁器 mock，返回 False（切分）
        mock_arbiter = Mock(spec=NoOpArbiter)
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

        buffer = self._create_buffer_with_kernel([1.0, 0.0])

        new_block = self._create_complete_block(
            anchor_text="可能的延续",
            rewritten_query="可能的延续",
            total_tokens=100
        )

        should, reason = adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.SEMANTIC_DRIFT

    def test_grey_area_without_arbiter_default_adsorb(self):
        """测试无仲裁器时灰度区间默认吸附"""
        # 无仲裁器
        config = SemanticAdsorberConfig(
            semantic_threshold_high=0.75,
            semantic_threshold_low=0.40,
            enable_arbiter=True,
        )
        adsorber = SemanticBoundaryAdsorber(
            config=config,
            arbiter=None,
        )
        adsorber.embedding_service = Mock()
        adsorber.embedding_service.encode.return_value = [0.7, 0.7]

        buffer = self._create_buffer_with_kernel([1.0, 0.0])

        new_block = self._create_complete_block(
            anchor_text="可能的延续",
            rewritten_query="可能的延续",
            total_tokens=100
        )

        should, reason = adsorber.should_adsorb(new_block, buffer)
        # 仲裁器不可用时，默认吸附
        assert should is True

    def test_incomplete_block(self):
        """测试未闭合的 Block"""
        buffer = Mock(spec=SemanticBuffer)

        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = False

        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True
        assert reason is None

    def test_custom_stop_words(self):
        """测试自定义停用词"""
        custom_stop_words = {"custom_stop"}
        config = SemanticAdsorberConfig(stop_words=custom_stop_words)

        adsorber = SemanticBoundaryAdsorber(config=config)

        assert "custom_stop" in adsorber.config.stop_words

    def test_invalid_thresholds(self):
        """测试无效的阈值参数"""
        with pytest.raises(ValueError):
            SemanticAdsorberConfig(
                semantic_threshold_low=0.8,
                semantic_threshold_high=0.7,  # low > high
            )

    def _create_buffer_with_kernel(self, kernel_vector: list) -> SemanticBuffer:
        """创建带有话题核心的 buffer"""
        buffer = SemanticBuffer(
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
        )
        buffer.topic_kernel_vector = kernel_vector

        # 添加一个 block（ContextBridge 需要）
        block = LogicalBlock()
        block.user_block = StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content="初始话题"
        )
        block.response_block = StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="这是对初始话题的回复"
        )
        buffer.add_block(block)

        return buffer

    def _create_complete_block(
        self,
        anchor_text: str,
        rewritten_query: str,
        total_tokens: int
    ) -> LogicalBlock:
        """创建已闭合的 LogicalBlock"""
        block = LogicalBlock()
        # 设置 user_block 和 response_block 使其成为 complete
        block.user_block = StreamMessage(
            message_type=StreamMessageType.USER_QUERY,
            content=anchor_text
        )
        block.response_block = StreamMessage(
            message_type=StreamMessageType.ASSISTANT_MESSAGE,
            content="回复内容"
        )
        # is_complete 是一个 property，由 user_block 和 response_block 决定
        block.total_tokens = total_tokens
        block.rewritten_query = rewritten_query
        return block


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
