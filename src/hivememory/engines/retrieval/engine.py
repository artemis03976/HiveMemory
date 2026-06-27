"""
Retrieval Engine - 纯计算执行层

职责：
- 调用检索器执行检索
- 统计检索耗时与产出规模
"""

from __future__ import annotations

import time
from typing import Optional

from hivememory.engines.retrieval.interfaces import BaseMemoryRetriever
from hivememory.engines.retrieval.models import RetrievalQuery, RetrievalResult


class RetrievalEngine:
    def __init__(self, retriever: BaseMemoryRetriever) -> None:
        self.retriever = retriever

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        score_threshold: float = 0.75,
    ) -> RetrievalResult:
        start_time = time.time()

        search_results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        memories = search_results.get_memories()
        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=memories,
            latency_ms=latency_ms,
            memories_count=len(memories),
            search_results=search_results,
        )


__all__ = ["RetrievalEngine"]
