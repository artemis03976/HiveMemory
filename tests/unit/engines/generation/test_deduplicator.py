"""
查重与演化管理器 (MemoryDeduplicator) 单元测试

测试覆盖:
- 查重判断逻辑 (DISTINCT/SIMILAR/IDENTICAL)
- 文本相似度计算
- 内容一致性检测
- 记忆合并与演化逻辑
"""

from uuid import uuid4

import pytest

from hivememory.config.patchouli import DeduplicatorConfig
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.extractor import ExtractedMemoryDraft
from hivememory.engines.generation.models import DuplicateDecision
from tests.helpers.memory import make_memory_metadata


class TestMemoryDeduplicator:
    """测试记忆查重器"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.config = DeduplicatorConfig()
        self.deduplicator = MemoryDeduplicator(config=self.config)

        # 构造基础数据
        self.draft = ExtractedMemoryDraft(
            title="Python Quicksort",
            summary="Implementation of quicksort",
            tags=["algorithm", "python"],
            memory_type="CODE_SNIPPET",
            content="def quicksort(): pass",
            confidence_score=0.9,
            has_value=True,
        )

        self.existing_memory = MemoryAtom(
            id=str(uuid4()),
            meta=make_memory_metadata(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
            ),
            index=IndexLayer(
                title="Python Quicksort",
                summary="Old summary",
                tags=["python", "sort"],
                memory_type=MemoryType.CODE_SNIPPET,
            ),
            payload=PayloadLayer(content="def quicksort(): pass"),
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

    def test_check_duplicate_distinct_without_candidates(self):
        """测试判定为 DISTINCT (无相似记忆)"""
        candidates = []

        decision, memory = self.deduplicator.check_duplicate(self.draft, candidates)

        assert decision == DuplicateDecision.DISTINCT
        assert memory is None

    def test_check_duplicate_identical(self):
        """测试判定为 IDENTICAL (高相似度 + 内容一致)"""
        # 模拟找到高相似度记忆
        candidates = [{"score": 0.98, "memory": self.existing_memory}]

        # 内容一致 (draft 和 existing 内容相同)
        decision, memory = self.deduplicator.check_duplicate(self.draft, candidates)

        assert decision == DuplicateDecision.IDENTICAL
        assert memory == self.existing_memory

    def test_check_duplicate_similar_high_score_diff_content(self):
        """测试判定为 SIMILAR (高相似度 + 内容不同)"""
        candidates = [{"score": 0.98, "memory": self.existing_memory}]

        # 修改 draft 内容
        self.draft.content = "def quicksort_v2(): pass"

        decision, memory = self.deduplicator.check_duplicate(self.draft, candidates)

        assert decision == DuplicateDecision.SIMILAR
        assert memory == self.existing_memory

    @pytest.mark.parametrize("score", [0.85, 0.95], ids=["between", "upper-bound"])
    def test_check_duplicate_similar_medium_score(self, score):
        """测试判定为 SIMILAR (中等相似度)"""
        candidates = [{"score": score, "memory": self.existing_memory}]

        decision, memory = self.deduplicator.check_duplicate(self.draft, candidates)

        assert decision == DuplicateDecision.SIMILAR

    @pytest.mark.parametrize("score", [0.5, 0.75], ids=["below", "lower-bound"])
    def test_check_duplicate_distinct_low_score(self, score):
        """测试判定为 DISTINCT (低相似度)"""
        candidates = [{"score": score, "memory": self.existing_memory}]

        decision, memory = self.deduplicator.check_duplicate(self.draft, candidates)

        assert decision == DuplicateDecision.DISTINCT
        assert memory == self.existing_memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
