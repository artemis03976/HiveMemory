"""User Query Analysis 的保守解析器实现。"""

from __future__ import annotations

from hivememory.core.protocol.gateway import (
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.gateway.analysis.models import (
    UserQueryAnalysisContext,
    UserQueryAnalysisResult,
)
from hivememory.system.config import UserQueryAnalysisConfig


class FallbackUserQueryAnalysisResolver:
    """不调用 Engine，直接生成稳定、保守的查询分析结果。"""

    def __init__(self, config: UserQueryAnalysisConfig) -> None:
        self._config = config

    async def resolve(
        self,
        context: UserQueryAnalysisContext,
    ) -> UserQueryAnalysisResult:
        """始终保留原始查询，并选择 RAG、写入记忆和混合检索。"""

        return UserQueryAnalysisResult(
            intent_type=IntentType.RAG,
            rewritten_query=context.raw_message,
            search_keywords=(),
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(
                mode=RetrievalMode.HYBRID,
                top_k=self._config.default_top_k,
            ),
        )


__all__ = ["FallbackUserQueryAnalysisResolver"]
