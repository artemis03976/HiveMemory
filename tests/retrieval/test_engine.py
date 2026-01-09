"""
RetrievalEngine 单元测试

测试覆盖:
- 完整检索流程集成
- 路由逻辑集成
- 错误处理
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from hivememory.core.models import MemoryAtom, MemoryType, IndexLayer, PayloadLayer, MetaData
from hivememory.retrieval.engine import RetrievalEngine, RetrievalResult
from hivememory.retrieval.router import RetrievalRouter
from hivememory.retrieval.query import QueryProcessor, ProcessedQuery
from hivememory.retrieval.searcher import HybridSearcher, SearchResults, SearchResult
from hivememory.retrieval.renderer import ContextRenderer

class TestRetrievalEngine:
    """测试检索引擎"""

    def setup_method(self):
        # Mock 所有组件
        self.mock_storage = Mock()
        self.mock_router = Mock(spec=RetrievalRouter)
        self.mock_processor = Mock(spec=QueryProcessor)
        self.mock_searcher = Mock(spec=HybridSearcher)
        self.mock_renderer = Mock(spec=ContextRenderer)
        
        self.engine = RetrievalEngine(
            storage=self.mock_storage,
            router=self.mock_router,
            processor=self.mock_processor,
            searcher=self.mock_searcher,
            renderer=self.mock_renderer
        )
        
        # 默认 Mock 行为
        self.mock_router.should_retrieve.return_value = True
        self.mock_processor.process.return_value = ProcessedQuery(semantic_query="test", original_query="test")
        
        self.memory = MemoryAtom(
            index=IndexLayer(title="M1", summary="This is a summary of the memory content.", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="C1"),
            meta=MetaData(source_agent_id="test", user_id="u1")
        )
        self.search_results = SearchResults(
            results=[SearchResult(memory=self.memory, score=0.9)]
        )
        self.mock_searcher.search.return_value = self.search_results
        self.mock_renderer.render.return_value = "<context>C1</context>"

    def test_retrieve_context_full_flow(self):
        """测试完整检索流程"""
        result = self.engine.retrieve_context(
            query="test query",
            user_id="u1"
        )
        
        assert result.should_retrieve is True
        assert result.memories_count == 1
        assert result.rendered_context == "<context>C1</context>"
        
        # 验证调用顺序
        self.mock_router.should_retrieve.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_searcher.search.assert_called_once()
        self.mock_renderer.render.assert_called_once()

    def test_retrieve_context_skipped(self):
        """测试路由跳过检索"""
        self.mock_router.should_retrieve.return_value = False
        
        result = self.engine.retrieve_context(
            query="hi",
            user_id="u1"
        )
        
        assert result.should_retrieve is False
        assert result.is_empty()
        # 后续步骤不应执行
        self.mock_processor.process.assert_not_called()
        self.mock_searcher.search.assert_not_called()

    def test_force_retrieve(self):
        """测试强制检索"""
        self.mock_router.should_retrieve.return_value = False
        
        result = self.engine.retrieve_context(
            query="hi",
            user_id="u1",
            force_retrieve=True
        )
        
        # 即使路由说不，也应该检索
        assert result.should_retrieve is True  # 注意：这里 Result 里的 should_retrieve 还是 True，因为逻辑被绕过了
        self.mock_searcher.search.assert_called_once()

    def test_search_error_handling(self):
        """测试检索错误处理"""
        self.mock_searcher.search.side_effect = Exception("Search failed")
        
        # 不应抛出异常，而是返回空结果
        result = self.engine.retrieve_context(
            query="test",
            user_id="u1"
        )
        
        assert result.is_empty()
        assert result.rendered_context == ""

    def test_search_memories_simple(self):
        """测试简化搜索接口"""
        # Mock processor return value
        self.mock_processor.process.return_value = ProcessedQuery(semantic_query="test", original_query="test")
        self.mock_searcher.search.return_value = self.search_results
        
        memories = self.engine.search_memories("test", "u1")
        
        assert len(memories) == 1
        assert memories[0].index.title == "M1"
        self.mock_processor.process.assert_called()
        self.mock_searcher.search.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
