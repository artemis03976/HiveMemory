"""
Retrieval Engine - 纯计算执行层

职责：
- 调用检索器执行检索
- 调用渲染器生成可注入上下文
- 统计检索与渲染的耗时与产出规模
"""

from __future__ import annotations

import time
from typing import Optional

from hivememory.engines.retrieval.interfaces import BaseContextRenderer, BaseMemoryRetriever
from hivememory.engines.retrieval.models import RenderFormat, RetrievalQuery, RetrievalResult


class RetrievalEngine:
    def __init__(
        self,
        retriever: BaseMemoryRetriever,
        renderer: BaseContextRenderer,
    ) -> None:
        self.retriever = retriever
        self.renderer = renderer

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        score_threshold: float = 0.75,
        render_format: Optional[RenderFormat] = None,
    ) -> RetrievalResult:
        start_time = time.time()

        search_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        memories = search_results.get_memories()

        rendered_context = ""
        if not search_results.is_empty():
            rendered_context = self.renderer.render(
                search_results.results,
                render_format=render_format,
            )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=memories,
            rendered_context=rendered_context,
            latency_ms=latency_ms,
            memories_count=len(memories),
            search_results=search_results,
        )


__all__ = [
    "RetrievalEngine"
]