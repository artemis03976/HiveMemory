"""
查重与演化管理器 (MemoryDeduplicator) 单元测试

测试覆盖:
- 查重决策逻辑 (CREATE/UPDATE/TOUCH)
- 文本相似度计算
- 内容一致性检测
- 记忆合并与演化逻辑
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.patchouli.config import DeduplicatorConfig
from hivememory.engines.generation.deduplicator import MemoryDeduplicator, DuplicateDecision
from hivememory.engines.generation.extractor import ExtractedMemoryDraft
from hivememory.infrastructure.storage import QdrantMemoryStore


class TestMemoryDeduplicator:
    """测试记忆查重器"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.mock_storage = Mock(spec=QdrantMemoryStore)
        self.config = DeduplicatorConfig()
        self.deduplicator = MemoryDeduplicator(
            storage=self.mock_storage,
            config=self.config
        )
        
        # 构造基础数据
        self.draft = ExtractedMemoryDraft(
            title="Python Quicksort",
            summary="Implementation of quicksort",
            tags=["algorithm", "python"],
            memory_type="CODE_SNIPPET",
            content="def quicksort(): pass",
            confidence_score=0.9,
            has_value=True
        )
        
        self.existing_memory = MemoryAtom(
            id=str(uuid4()),
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                session_id="session1",
                confidence_score=0.8
            ),
            index=IndexLayer(
                title="Python Quicksort",
                summary="Old summary",
                tags=["python", "sort"],
                memory_type=MemoryType.CODE_SNIPPET,
            ),
            payload=PayloadLayer(
                content="def quicksort(): pass"
            )
        )

    def test_calculate_text_similarity(self):
        """测试文本相似度计算"""
        # 完全相同
        assert self.deduplicator._calculate_text_similarity("abc", "abc") == 1.0
        
        # 完全不同
        assert self.deduplicator._calculate_text_similarity("abc", "def") == 0.0
        
        # 部分相同 (Jaccard)
        # "apple" -> {a,p,l,e}, "pear" -> {p,e,a,r}
        # intersection={p,e,a} (3), union={a,p,l,e,r} (5) -> 3/5 = 0.6
        # 但是代码中使用了 re.findall(r'\w+', text.lower())，这会把 "apple" 当作一个词 "apple"
        # 而不是字符集合。
        # 如果要测试字符级相似度，输入应该是空格分隔的词，或者修改测试用例
        
        # 测试用例修正：基于词的 Jaccard 相似度
        text1 = "apple banana orange"
        text2 = "apple banana pear"
        # words1 = {apple, banana, orange} (3)
        # words2 = {apple, banana, pear} (3)
        # intersection = {apple, banana} (2)
        # union = {apple, banana, orange, pear} (4)
        # similarity = 2/4 = 0.5
        
        assert self.deduplicator._calculate_text_similarity(text1, text2) == 0.5

    def test_check_duplicate_create(self):
        """测试判定为 CREATE (无相似记忆)"""
        self.mock_storage.search_memories.return_value = []
        
        decision, memory = self.deduplicator.check_duplicate(self.draft)
        
        assert decision == DuplicateDecision.CREATE
        assert memory is None

    def test_check_duplicate_touch(self):
        """测试判定为 TOUCH (高相似度 + 内容一致)"""
        # 模拟找到高相似度记忆
        self.mock_storage.search_memories.return_value = [{
            "score": 0.98,
            "memory": self.existing_memory
        }]
        
        # 内容一致 (draft 和 existing 内容相同)
        decision, memory = self.deduplicator.check_duplicate(self.draft)
        
        assert decision == DuplicateDecision.TOUCH
        assert memory == self.existing_memory

    def test_check_duplicate_update_high_score_diff_content(self):
        """测试判定为 UPDATE (高相似度 + 内容不同)"""
        self.mock_storage.search_memories.return_value = [{
            "score": 0.98,
            "memory": self.existing_memory
        }]
        
        # 修改 draft 内容
        self.draft.content = "def quicksort_v2(): pass"
        
        decision, memory = self.deduplicator.check_duplicate(self.draft)
        
        assert decision == DuplicateDecision.UPDATE
        assert memory == self.existing_memory

    def test_check_duplicate_update_medium_score(self):
        """测试判定为 UPDATE (中等相似度)"""
        self.mock_storage.search_memories.return_value = [{
            "score": 0.85,  # 0.75 < 0.85 < 0.95
            "memory": self.existing_memory
        }]
        
        decision, memory = self.deduplicator.check_duplicate(self.draft)
        
        assert decision == DuplicateDecision.UPDATE

    def test_check_duplicate_create_low_score(self):
        """测试判定为 CREATE (低相似度)"""
        self.mock_storage.search_memories.return_value = [{
            "score": 0.5,  # < 0.75
            "memory": self.existing_memory
        }]
        
        decision, memory = self.deduplicator.check_duplicate(self.draft)
        
        assert decision == DuplicateDecision.CREATE
        assert memory == self.existing_memory

    def test_merge_memory(self):
        """测试记忆合并"""
        new_draft = ExtractedMemoryDraft(
            title="New Title",  # 应该被忽略
            summary="Better summary",
            tags=["new_tag"],
            memory_type="CODE_SNIPPET",
            content="New content",
            confidence_score=0.9,
            has_value=True
        )
        
        merged = self.deduplicator.merge_memory(self.existing_memory, new_draft)
        
        # 验证合并逻辑
        assert merged.id == self.existing_memory.id
        assert merged.index.title == self.existing_memory.index.title  # 保留旧标题
        assert merged.index.summary == "Better summary"  # 新摘要更长
        assert "new_tag" in merged.index.tags
        assert "python" in merged.index.tags
        
        # 验证内容合并
        assert "def quicksort(): pass" in merged.payload.content
        assert "New content" in merged.payload.content
        assert "## 更新" in merged.payload.content
        
        # 验证置信度加权平均 (0.8 * 0.6 + 0.9 * 0.4 = 0.48 + 0.36 = 0.84)
        assert abs(merged.meta.confidence_score - 0.84) < 0.01

    def test_merge_content_identical(self):
        """测试合并相似内容（不重复追加）"""
        content = self.deduplicator._merge_content("abc", "abc")
        assert content == "abc"  # 直接返回新内容（相同）


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
