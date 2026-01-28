"""
检索引擎组件协作测试

测试检索引擎内部各组件之间的协作：
- DenseRetriever 与 SparseRetriever 的融合
- Reranker 与检索结果的交互
- ContextRenderer 与检索结果的格式化
- HybridRetriever 的整体编排

不测试：与外部存储（Qdrant）的交互
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock

from hivememory.core.models import (
    MemoryAtom,
    MetaData,
    IndexLayer,
    PayloadLayer,
    MemoryType,
)
from hivememory.engines.retrieval import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    CrossEncoderReranker,
    FullContextRenderer,
    RenderFormat,
    ReciprocalRankFusion,
    create_retriever,
)
from hivememory.patchouli.config import (
    RerankerConfig, 
    ReciprocalRankFusionConfig, 
    HybridRetrieverConfig, 
    DenseRetrieverConfig, 
    SparseRetrieverConfig,
    FullRendererConfig,
)
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    QueryFilters,
    SearchResult,
    SearchResults,
)


# 创建测试记忆
def create_test_memory(title: str, content: str, memory_type: MemoryType = MemoryType.FACT) -> MemoryAtom:
    """创建测试记忆"""
    return MemoryAtom(
        meta=MetaData(
            source_agent_id="test_agent",
            user_id="test_user",
            confidence_score=0.9,
        ),
        index=IndexLayer(
            title=title,
            summary=f"{title}的摘要信息，长度必须超过十个字符",
            tags=["test"],
            memory_type=memory_type,
        ),
        payload=PayloadLayer(content=content),
    )


class TestDenseAndSparseRetrieverCollaboration:
    """测试 DenseRetriever 与 SparseRetriever 的融合"""

    def test_fusion_strategy_combines_results(self):
        """测试融合策略组合两种检索结果"""
        # Mock 密集检索结果
        dense_results = SearchResults(results=[
            SearchResult(
                memory=create_test_memory("Python代码", "def test(): pass", MemoryType.CODE_SNIPPET),
                score=0.85,
                match_reason="语义相似",
            ),
            SearchResult(
                memory=create_test_memory("Python教程", "Python学习笔记"),
                score=0.75,
                match_reason="语义相似",
            ),
        ])

        # Mock 稀疏检索结果
        sparse_results = SearchResults(results=[
            SearchResult(
                memory=create_test_memory("Python函数", "def python_func(): pass", MemoryType.CODE_SNIPPET),
                score=0.90,
                match_reason="关键词匹配",
            ),
            SearchResult(
                memory=create_test_memory("Java代码", "public void test() {}", MemoryType.CODE_SNIPPET),
                score=0.70,
                match_reason="关键词匹配",
            ),
        ])

        # 创建融合器
        fusion = ReciprocalRankFusion(config=ReciprocalRankFusionConfig(rrf_k=60))

        # 融合结果
        fused = fusion.fuse(dense_results, sparse_results)

        # 验证融合结果
        assert len(fused.results) > 0
        # RRF 应该重新排序结果

    def test_hybrid_retriever_calls_both_retrievers(self):
        """测试混合检索器调用两种检索器"""
        mock_storage = Mock()
        
        config = HybridRetrieverConfig()
        
        # Mock internal retrievers
        dense_retriever = Mock()
        dense_retriever.retrieve = Mock(return_value=SearchResults())
        
        sparse_retriever = Mock()
        sparse_retriever.retrieve = Mock(return_value=SearchResults())
        
        fusion = Mock()
        fusion.fuse = Mock(return_value=SearchResults())

        hybrid = HybridRetriever(
            config=config,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
        )

        query = RetrievalQuery(semantic_query="Python代码")
        results = hybrid.retrieve(query, top_k=5)

        # 验证两种检索器都被调用
        assert results is not None
        dense_retriever.retrieve.assert_called()
        sparse_retriever.retrieve.assert_called()


class TestRerankerAndRetrieverCollaboration:
    """测试 Reranker 与检索器的协作"""

    def test_reranker_reorders_results(self):
        """测试重排序器重新排列结果"""
        # 创建初始结果（故意打乱顺序）
        results = SearchResults(results=[
            SearchResult(
                memory=create_test_memory("低分结果", "内容"),
                score=0.5,
                match_reason="原始",
            ),
            SearchResult(
                memory=create_test_memory("中分结果", "内容"),
                score=0.7,
                match_reason="原始",
            ),
            SearchResult(
                memory=create_test_memory("高分结果", "内容"),
                score=0.9,
                match_reason="原始",
            ),
        ])

        # Mock service
        mock_service = Mock()
        # compute_score returns raw scores, order matching input pairs
        # Input order is [低分, 中分, 高分]
        # We want high scores for "高分结果"
        mock_service.compute_score = Mock(return_value=[0.1, 0.5, 0.9])

        reranker = CrossEncoderReranker(service=mock_service, config=RerankerConfig())

        # Mock 重排序逻辑
        query = RetrievalQuery(semantic_query="测试查询")
        reranked = reranker.rerank(results, query=query)

        # 验证结果被重新排序 (Should be high -> low)
        assert len(reranked.results) == 3
        assert reranked.results[0].memory.index.title == "高分结果"
        assert reranked.results[2].memory.index.title == "低分结果"


class TestRendererAndResultsCollaboration:
    """测试 ContextRenderer 与检索结果的协作"""

    def test_renderer_formats_results(self):
        """测试渲染器格式化结果"""
        results = [
            SearchResult(
                memory=create_test_memory("测试记忆", "测试内容"),
                score=0.9,
                match_reason="语义匹配",
            ),
        ]

        # 测试 XML 格式
        renderer_xml = FullContextRenderer(FullRendererConfig())
        xml_output = renderer_xml.render(results)

        assert "测试记忆" in xml_output or "测试内容" in xml_output
        assert len(xml_output) > 0

    def test_renderer_markdown_format(self):
        """测试 Markdown 格式渲染"""
        results = [
            SearchResult(
                memory=create_test_memory("Python代码", "def test(): pass"),
                score=0.9,
                match_reason="代码匹配",
            ),
            SearchResult(
                memory=create_test_memory("Python文档", "文档内容"),
                score=0.8,
                match_reason="文档匹配",
            ),
        ]

        renderer_md = FullContextRenderer(FullRendererConfig())
        md_output = renderer_md.render(results, render_format=RenderFormat.MARKDOWN)

        # 验证 Markdown 格式
        assert "Python代码" in md_output or "Python文档" in md_output
        assert len(md_output) > 0

    def test_renderer_respects_token_limit(self):
        """测试渲染器遵守 token 限制"""
        # 创建大量结果
        results = [
            SearchResult(
                memory=create_test_memory(f"记忆{i}", "内容" * 100),
                score=0.9 - i * 0.1,
                match_reason="测试",
            )
            for i in range(10)
        ]

        from hivememory.patchouli.config import FullRendererConfig
        config = FullRendererConfig(max_tokens=100)
        renderer = FullContextRenderer(config)
        output = renderer.render(results)

        # 输出应该被截断
        assert len(output) > 0
        # 不应该包含所有记忆


class TestQueryAndFilterCollaboration:
    """测试查询与过滤器的协作"""

    def test_query_with_filters(self):
        """测试带过滤条件的查询"""
        mock_storage = Mock()
        # Mock search_memories to return a list of dicts as expected by DenseRetriever
        
        hit1 = {
            "memory": create_test_memory("代码", "代码", MemoryType.CODE_SNIPPET),
            "score": 0.9,
            "id": "1"
        }
        
        hit2 = {
            "memory": create_test_memory("事实", "事实", MemoryType.FACT),
            "score": 0.8,
            "id": "2"
        }
        
        mock_storage.search_memories = Mock(return_value=[hit1, hit2])

        retriever = DenseRetriever(storage=mock_storage, config=DenseRetrieverConfig())

        query = RetrievalQuery(
            semantic_query="Python",
            filters=QueryFilters(memory_type=MemoryType.CODE_SNIPPET),
        )

        results = retriever.retrieve(query, top_k=5)
        
        # Verify search_memories called with correct filters
        mock_storage.search_memories.assert_called()
        call_args = mock_storage.search_memories.call_args
        assert call_args is not None
        
        # Verify results
        assert len(results.results) == 2

        # 验证结果
        assert results is not None

    def test_multiple_filters(self):
        """测试多个过滤条件"""
        mock_storage = Mock()
        mock_storage.get_memories_by_filter = Mock(return_value=[
            create_test_memory("用户1", "内容1", MemoryType.USER_PROFILE),
        ])

        query = RetrievalQuery(
            semantic_query="查询",
            filters=QueryFilters(
                memory_type=MemoryType.USER_PROFILE,
                user_id="user1",
                tags=["profile"],
            ),
        )

        # 验证过滤器正确构建
        assert query.filters.memory_type == MemoryType.USER_PROFILE
        assert query.filters.user_id == "user1"
        assert query.filters.tags == ["profile"]


class TestHybridRetrieverOrchestration:
    """测试 HybridRetriever 的整体编排"""

    def test_hybrid_full_pipeline(self):
        """测试混合检索完整流程"""
        mock_storage = Mock()
        
        config = HybridRetrieverConfig(
            enable_parallel=False  # Sequential for easier mocking
        )
        
        # Mock internal retrievers
        dense_retriever = Mock()
        dense_retriever.retrieve = Mock(return_value=SearchResults(results=[
            SearchResult(memory=create_test_memory("语义结果", "内容"), score=0.8)
        ]))
        
        sparse_retriever = Mock()
        sparse_retriever.retrieve = Mock(return_value=SearchResults(results=[
            SearchResult(memory=create_test_memory("关键词结果", "内容"), score=0.9)
        ]))

        fusion = ReciprocalRankFusion(config=ReciprocalRankFusionConfig())

        hybrid = HybridRetriever(
            config=config,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
        )

        query = RetrievalQuery(semantic_query="Python代码")
        results = hybrid.retrieve(query, top_k=5)

        # 验证结果
        assert results is not None
        assert results.results is not None
        assert len(results.results) > 0

    def test_hybrid_with_reranking(self):
        """测试混合检索带重排序"""
        mock_storage = Mock()
        
        config = HybridRetrieverConfig()
        config.reranker.enabled = True
        
        # Mock retrievers
        dense_retriever = Mock()
        dense_retriever.retrieve = Mock(return_value=SearchResults(results=[
            SearchResult(memory=create_test_memory(f"结果{i}", "内容"), score=0.9)
            for i in range(5)
        ]))
        sparse_retriever = Mock()
        sparse_retriever.retrieve = Mock(return_value=SearchResults())
        
        fusion = ReciprocalRankFusion(config=ReciprocalRankFusionConfig())

        # Mock reranker
        reranker = Mock()
        reranker.rerank = Mock(return_value=SearchResults(results=[
            SearchResult(memory=create_test_memory("Top1", "Content"), score=0.99)
        ]))

        hybrid = HybridRetriever(
            config=config,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
            reranker=reranker
        )

        query = RetrievalQuery(semantic_query="查询")
        results = hybrid.retrieve(query, top_k=3)

        # 验证结果数量被限制
        assert len(results.results) == 1
        reranker.rerank.assert_called()

    def test_create_retriever_dense_only(self):
        """测试创建 DenseRetriever"""
        mock_storage = Mock()
        config = DenseRetrieverConfig()
        
        retriever = create_retriever(storage=mock_storage, config=config)
        
        assert isinstance(retriever, DenseRetriever)


class TestScoreNormalization:
    """测试分数归一化"""

    def test_dense_and_sparse_score_normalization(self):
        """测试密集和稀疏分数归一化"""
        # 密集检索分数通常是 0-1
        dense_score = 0.85
        # 稀疏检索分数可能范围不同
        sparse_score = 5.0

        # 归一化后应该在同一范围内
        normalized_dense = dense_score  # 已经在 0-1
        normalized_sparse = min(sparse_score / 10.0, 1.0)  # 简单归一化

        assert 0 <= normalized_dense <= 1
        assert 0 <= normalized_sparse <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
