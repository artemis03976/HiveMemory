"""
SemanticBoundaryAdsorber 单元测试

测试覆盖:
- 相似度计算 (Mock Embedding)
- 吸附判定逻辑 (短文本, Token 溢出, 超时, 语义漂移)
- 话题核心向量更新 (EMA)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from hivememory.core.models import FlushReason
from hivememory.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.perception.models import LogicalBlock, SemanticBuffer

class TestSemanticBoundaryAdsorber:
    """测试语义边界吸附器"""

    def setup_method(self):
        self.adsorber = SemanticBoundaryAdsorber(
            semantic_threshold=0.6,
            short_text_threshold=50,
            idle_timeout_seconds=60,
            ema_alpha=0.5
        )
        
        # Mock embedding service
        self.adsorber._embedding_service = Mock()
        # Mock encode to return a fixed vector
        self.adsorber._embedding_service.encode.return_value = [1.0, 0.0]

    def test_compute_similarity(self):
        """测试相似度计算"""
        # 向量 [1, 0] dot [1, 0] = 1.0
        similarity = self.adsorber._compute_cosine_similarity("text", [1.0, 0.0])
        assert similarity == 1.0

    def test_should_adsorb_short_text(self):
        """测试短文本强吸附"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 0
        buffer.max_tokens = 1000
        buffer.is_idle.return_value = False
        
        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 10  # < 50
        
        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is True
        assert reason == FlushReason.SHORT_TEXT_ADSORB

    def test_should_adsorb_token_overflow(self):
        """测试 Token 溢出不吸附"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 900
        buffer.max_tokens = 1000
        buffer.is_idle.return_value = False
        
        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 200  # 900+200 > 1000
        
        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.TOKEN_OVERFLOW

    def test_should_adsorb_timeout(self):
        """测试超时不吸附"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 100
        buffer.max_tokens = 1000
        buffer.is_idle.return_value = True  # 超时
        buffer.last_update = 1000.0  # Set attribute explicitly
        
        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 100
        
        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.IDLE_TIMEOUT

    def test_semantic_drift(self):
        """测试语义漂移"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.total_tokens = 100
        buffer.max_tokens = 1000
        buffer.is_idle.return_value = False
        buffer.topic_kernel_vector = [0.0, 1.0] # 与 Mock 的 [1.0, 0.0] 正交，sim=0
        
        new_block = Mock(spec=LogicalBlock)
        new_block.is_complete = True
        new_block.total_tokens = 100 # > 50
        new_block.anchor_text = "new topic"
        
        should, reason = self.adsorber.should_adsorb(new_block, buffer)
        assert should is False
        assert reason == FlushReason.SEMANTIC_DRIFT

    def test_update_topic_kernel(self):
        """测试话题核心更新 (EMA)"""
        buffer = Mock(spec=SemanticBuffer)
        buffer.buffer_id = "test"
        buffer.topic_kernel_vector = [0.0, 1.0]
        
        new_block = Mock(spec=LogicalBlock)
        new_block.block_id = "b1"
        new_block.anchor_text = "update"
        
        # Mock encode returns [1.0, 0.0]
        # Alpha = 0.5
        # Expected = 0.5*[1,0] + 0.5*[0,1] = [0.5, 0.5]
        
        self.adsorber.update_topic_kernel(buffer, new_block)
        
        updated = buffer.topic_kernel_vector
        assert pytest.approx(updated[0]) == 0.5
        assert pytest.approx(updated[1]) == 0.5
