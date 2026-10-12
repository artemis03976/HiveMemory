"""
Retrieval Engine - 纯计算执行层

职责：
- 编排召回与重排两个阶段：检索器召回候选，重排器对任意检索器的输出精排
- 统计检索耗时与产出规模
"""

from __future__ import annotations

import time

from hivememory.engines.retrieval.interfaces import BaseMemoryRetriever, BaseReranker
from hivememory.engines.retrieval.models import RetrievalQuery, RetrievalResult


class RetrievalEngine:
    def __init__(self, retriever: BaseMemoryRetriever, *, reranker: BaseReranker) -> None:
        # 重排序关闭时由 NoopReranker 顶替，重排阶段始终存在。
        self.retriever = retriever
        self.reranker = reranker

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
        score_threshold: float = 0.75,
    ) -> RetrievalResult:
        start_time = time.time()

        # 召回：按重排器声明的候选池大小取候选
        candidates = await self.retriever.retrieve(
            query=query,
            top_k=self.reranker.candidate_count(top_k),
            score_threshold=score_threshold,
        )
        # 重排：重排后的分数不再套用召回阈值，只截取 top_k
        search_results = self.reranker.rerank(candidates, query)
        search_results.results = search_results.results[:top_k]

        memories = search_results.get_memories()
        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=memories,
            latency_ms=latency_ms,
            memories_count=len(memories),
            search_results=search_results,
        )


__all__ = ["RetrievalEngine"]
