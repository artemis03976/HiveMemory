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
from hivememory.retrieval.engine import MemoryRetrievalEngine, RetrievalResult
from hivememory.retrieval.router import RetrievalRouter
from hivememory.retrieval.query import QueryProcessor, ProcessedQuery
from hivememory.retrieval.retriever import HybridRetriever, SearchResults, SearchResult
from hivememory.retrieval.renderer import ContextRenderer

class TestRetrievalEngine:
    """测试检索引擎"""

    def setup_method(self):
        self.mock_storage = MagicMock()
        self.mock_router = MagicMock(spec=RetrievalRouter)
        self.mock_processor = MagicMock(spec=QueryProcessor)
        self.mock_retriever = MagicMock(spec=HybridRetriever)
        self.mock_renderer = MagicMock(spec=ContextRenderer)
        
        # 构造引擎，手动注入组件
        self.engine = MemoryRetrievalEngine(
            storage=self.mock_storage,
            router=self.mock_router,
            processor=self.mock_processor,
            retriever=self.mock_retriever,
            renderer=self.mock_renderer,
            config=None
        )
        
        # 默认 Mock 行为
        self.mock_router.should_retrieve.return_value = True
        
        self.processed_query = ProcessedQuery(
            original_query="test query",
            semantic_query="test query",
            filters={}
        )
        self.mock_processor.process.return_value = self.processed_query
        
        # 创建一个真实的 MemoryAtom 用于测试，避免 Pydantic 校验错误
        from hivememory.core.models import MemoryAtom, IndexLayer, PayloadLayer, MetaData, MemoryType
        from datetime import datetime
        
        memory = MemoryAtom(
            index=IndexLayer(title="Test Memory", summary="This is a summary of the memory content.", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Memory content"),
            meta=MetaData(source_agent_id="agent", user_id="user", updated_at=datetime.now())
        )
        
        self.search_results = SearchResults(
            results=[
                SearchResult(
                    memory=memory,
                    score=0.9
                )
            ]
        )
        self.mock_retriever.retrieve.return_value = self.search_results
        
        self.mock_renderer.render.return_value = "<context>...</context>"

    def test_retrieve_context_full_flow(self):
        """测试完整检索流程"""
        result = self.engine.retrieve_context("query", "u1")
        
        # 验证各组件调用顺序
        self.mock_router.should_retrieve.assert_called_once()
        self.mock_processor.process.assert_called_once()
        self.mock_retriever.retrieve.assert_called_once()
        self.mock_renderer.render.assert_called_once()
        
        assert result.rendered_context == "<context>...</context>"
        assert result.memories_count == 1

    def test_retrieve_context_skipped(self):
        """测试路由决定跳过检索"""
        self.mock_router.should_retrieve.return_value = False
        
        result = self.engine.retrieve_context("query", "u1")
        
        assert not result.should_retrieve
        self.mock_processor.process.assert_not_called()
        self.mock_retriever.retrieve.assert_not_called()

    def test_force_retrieve(self):
        """测试强制检索"""
        self.mock_router.should_retrieve.return_value = False
        
        # 强制检索应忽略路由结果
        result = self.engine.retrieve_context("query", "u1", force_retrieve=True)
        
        self.mock_retriever.retrieve.assert_called_once()
        assert result.memories_count == 1

    def test_search_error_handling(self):
        """测试检索异常处理"""
        self.mock_retriever.retrieve.side_effect = Exception("Search failed")
        
        result = self.engine.retrieve_context("query", "u1")
        
        # 应该捕获异常并返回空结果
        # 根据 engine.py:183, result.rendered_context 默认为 ""
        # 异常捕获后 result.rendered_context 不会被修改，所以是 ""
        assert result.rendered_context == ""
        assert result.memories_count == 0

    def test_search_memories_simple(self):
        """测试简单搜索接口"""
        memories = self.engine.retrieve_memories("query", "u1")
        
        assert len(memories) == 1
        self.mock_processor.process.assert_called_once()
        self.mock_retriever.retrieve.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
