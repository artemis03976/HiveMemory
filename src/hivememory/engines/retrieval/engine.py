"""
Retrieval Engine - 纯计算执行层

职责：
- 编排召回与重排两个阶段：检索器召回候选，重排器对任意检索器的输出精排
- 在重排之后按本次结果条数截取一次（请求未给出时使用配置的默认条数）
- 统计检索耗时与产出规模
"""

from __future__ import annotations

import time

from hivememory.engines.retrieval.interfaces import BaseMemoryRetriever, BaseReranker
from hivememory.engines.retrieval.models import RetrievalQuery, RetrievalResult


class RetrievalEngine:
    def __init__(
        self,
        retriever: BaseMemoryRetriever,
        *,
        reranker: BaseReranker,
        default_top_k: int,
    ) -> None:
        # 重排序关闭时由 NoopReranker 顶替，重排阶段始终存在。
        self.retriever = retriever
        self.reranker = reranker
        self.default_top_k = default_top_k

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
    ) -> RetrievalResult:
        start_time = time.time()
        result_count = self.default_top_k if top_k is None else top_k

        # 召回：按重排器声明的候选池大小取候选；分数阈值只在 Dense 召回内部生效
        candidates = await self.retriever.retrieve(
            query=query,
            top_k=self.reranker.candidate_count(result_count),
        )
        # 重排：重排分数不设阈值；最终截取只在这里进行一次
        search_results = await self.reranker.rerank(candidates, query)
        search_results.results = search_results.results[:result_count]

        memories = search_results.get_memories()
        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            memories=memories,
            latency_ms=latency_ms,
            memories_count=len(memories),
            search_results=search_results,
        )


__all__ = ["RetrievalEngine"]
