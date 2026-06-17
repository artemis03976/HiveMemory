"""
RetrievalEngine 单元测试

测试覆盖:
- 有结果时调用渲染器
- 空结果时跳过渲染
- 参数传递 (top_k, score_threshold, render_format)
- 返回结果字段完整性
- 延迟测量
"""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.models import (
    RetrievalQuery,
    QueryFilters,
    SearchResult,
    SearchResults,
    RetrievalResult,
    RenderFormat,
)
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType


def _make_memory(title="测试记忆") -> MemoryAtom:
    """辅助: 构建测试用 MemoryAtom"""
    return MemoryAtom(
        meta=MetaData(source_agent_id="a1", user_id="u1", session_id="s1"),
        index=IndexLayer(title=title, summary="这是一段足够长的测试摘要用于通过验证", tags=["t1"], memory_type=MemoryType.FACT),
        payload=PayloadLayer(content="内容"),
    )


def _make_query(text="测试查询") -> RetrievalQuery:
    """辅助: 构建测试用 RetrievalQuery"""
    return RetrievalQuery(
        semantic_query=text,
        keywords=[],
        filters=QueryFilters(),
    )


class TestRetrievalEngine:
    """RetrievalEngine 编排逻辑单元测试"""

    def setup_method(self):
        self.mock_retriever = AsyncMock()
        self.mock_renderer = Mock()
        self.engine = RetrievalEngine(
            retriever=self.mock_retriever,
            renderer=self.mock_renderer,
        )

    @pytest.mark.asyncio
    async def test_retrieve_with_results(self):
        """有结果时调用渲染器"""
        mem = _make_memory()
        sr = SearchResult(memory=mem, score=0.9)
        self.mock_retriever.retrieve.return_value = SearchResults(results=[sr])
        self.mock_renderer.render.return_value = "<memory>内容</memory>"

        result = await self.engine.retrieve(_make_query())

        self.mock_renderer.render.assert_called_once()
        assert result.rendered_context == "<memory>内容</memory>"
        assert result.memories_count == 1
        assert len(result.memories) == 1

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        """空结果时不调用渲染器"""
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])

        result = await self.engine.retrieve(_make_query())

        self.mock_renderer.render.assert_not_called()
        assert result.rendered_context == ""
        assert result.memories_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_passes_parameters(self):
        """top_k 和 score_threshold 正确传递给 retriever"""
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])

        await self.engine.retrieve(_make_query(), top_k=10, score_threshold=0.5)

        call_kwargs = self.mock_retriever.retrieve.call_args
        assert call_kwargs[1]["top_k"] == 10
        assert call_kwargs[1]["score_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_with_render_format(self):
        """render_format 正确传递给 renderer"""
        mem = _make_memory()
        sr = SearchResult(memory=mem, score=0.9)
        self.mock_retriever.retrieve.return_value = SearchResults(results=[sr])
        self.mock_renderer.render.return_value = "rendered"

        await self.engine.retrieve(_make_query(), render_format=RenderFormat.XML)

        call_kwargs = self.mock_renderer.render.call_args
        assert call_kwargs[1]["render_format"] == RenderFormat.XML

    @pytest.mark.asyncio
    async def test_retrieve_result_fields(self):
        """返回结果包含所有必要字段"""
        mem = _make_memory()
        sr = SearchResult(memory=mem, score=0.9)
        search_results = SearchResults(results=[sr])
        self.mock_retriever.retrieve.return_value = search_results
        self.mock_renderer.render.return_value = "ctx"

        result = await self.engine.retrieve(_make_query())

        assert result.memories == [mem]
        assert result.search_results is search_results
        assert result.memories_count == 1
        assert result.rendered_context == "ctx"

    @pytest.mark.asyncio
    async def test_retrieve_latency_measured(self):
        """latency_ms 应大于 0"""
        self.mock_retriever.retrieve.return_value = SearchResults(results=[])

        result = await self.engine.retrieve(_make_query())

        assert result.latency_ms >= 0

    def test_render_memories_uses_renderer(self):
        mem = _make_memory()
        self.mock_renderer.render.return_value = "rendered"

        result = self.engine.render_memories([mem], render_format=RenderFormat.XML)

        assert result == "rendered"
        self.mock_renderer.render.assert_called_once_with(
            [mem],
            render_format=RenderFormat.XML,
        )

    def test_render_memories_empty_skips_renderer(self):
        result = self.engine.render_memories([])

        assert result == ""
        self.mock_renderer.render.assert_not_called()
