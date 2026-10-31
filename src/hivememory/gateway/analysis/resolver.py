"""User Query Analysis 的 Resolver 实现。

- `FallbackUserQueryAnalysisResolver`：不调用 Engine，生成稳定保守结果。
- `LLMUserQueryAnalysisResolver`：第一代实现，本地规则前置 + 单次共享
  LLM 调用（意图/重写/关键词）+ 纯函数派生检索计划。
"""

from __future__ import annotations

import re
from time import perf_counter

from hivememory.components.events.bus import (
    NullRuntimeEventSink,
    RuntimeEventSink,
)
from hivememory.config.gateway import UserQueryAnalysisConfig
from hivememory.core.contracts.runtime_events import (
    RuntimeEvent,
    RuntimeEventType,
)
from hivememory.core.protocol.gateway import (
    IntentType,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.engines.gateway.query_understanding import (
    QueryUnderstandingEngine,
    QueryUnderstandingError,
)
from hivememory.gateway.analysis.models import (
    UserQueryAnalysisContext,
    UserQueryAnalysisResult,
)
from hivememory.gateway.errors import RecoverableGatewayError


class FallbackUserQueryAnalysisResolver:
    """不调用 Engine，直接生成稳定、保守的查询分析结果。"""

    def __init__(self, config: UserQueryAnalysisConfig) -> None:
        self._config = config

    async def resolve(
        self,
        context: UserQueryAnalysisContext,
    ) -> UserQueryAnalysisResult:
        """始终保留原始查询，并选择 RAG 与混合检索。"""

        return UserQueryAnalysisResult(
            intent_type=IntentType.RAG,
            rewritten_query=context.raw_message,
            search_keywords=(),
            retrieval_plan=RetrievalPlan(
                mode=RetrievalMode.HYBRID,
                top_k=self._config.default_top_k,
            ),
        )


class LLMUserQueryAnalysisResolver:
    """第一代 Resolver：规则前置 + 单次共享 LLM 调用 + 纯函数派生。

    技术债说明：意图识别、query rewrite 与意图分解当前共享一次 LLM 调用
    （QueryUnderstandingEngine），任一能力的提示或模型调整都会连带影响其余能力。
    """

    _WRITE_INTENT_PATTERNS: tuple[str, ...] = (
        r"^(请|帮我|帮忙)?记住",
        r"^以后(都|请|帮我|要)?",
        r"^从现在开始",
        r"^remember\b",
        r"^from now on\b",
    )

    def __init__(
        self,
        *,
        config: UserQueryAnalysisConfig,
        engine: QueryUnderstandingEngine,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._config = config
        self._engine = engine
        self._runtime_events = runtime_events or NullRuntimeEventSink()
        self._write_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._WRITE_INTENT_PATTERNS
        ]

    async def resolve(
        self,
        context: UserQueryAnalysisContext,
    ) -> UserQueryAnalysisResult:
        """完成一次完整查询分析；能力失败转换为 RecoverableGatewayError。"""

        rule_intent = self._rule_intent(context)

        started_at = perf_counter()
        try:
            analysis = await self._engine.analyze(
                context.raw_message,
                topic_data=context.routed_topic_data,
                session_turns=context.session_turns,
            )
        except QueryUnderstandingError as exc:
            self._emit_capability(started_at, error=exc)
            raise RecoverableGatewayError("查询分析不可用") from exc
        self._emit_capability(started_at, error=None)

        intent_type = rule_intent or analysis.intent_type
        retrieval_plan = self._derive_retrieval_plan(
            intent_type=intent_type,
            search_keywords=analysis.search_keywords,
        )
        return UserQueryAnalysisResult(
            intent_type=intent_type,
            rewritten_query=analysis.rewritten_query,
            search_keywords=analysis.search_keywords,
            retrieval_plan=retrieval_plan,
        )

    def _rule_intent(self, context: UserQueryAnalysisContext) -> IntentType | None:
        """零成本前置规则：显式记忆指令直接判为 WRITE 意图，覆盖模型的判断。"""

        text = context.raw_message.strip()
        if any(pattern.search(text) for pattern in self._write_patterns):
            return IntentType.WRITE
        return None

    def _derive_retrieval_plan(
        self,
        *,
        intent_type: IntentType,
        search_keywords: tuple[str, ...],
    ) -> RetrievalPlan:
        """检索策略是纯派生：由意图与关键词决定，不单独调用模型。"""

        if intent_type in (IntentType.CHAT, IntentType.WRITE):
            return RetrievalPlan(mode=RetrievalMode.SKIP, top_k=0)
        if not search_keywords:
            return RetrievalPlan(
                mode=RetrievalMode.DENSE,
                top_k=self._config.default_top_k,
            )
        return RetrievalPlan(
            mode=RetrievalMode.HYBRID,
            top_k=self._config.default_top_k,
        )

    def _emit_capability(
        self,
        started_at: float,
        *,
        error: QueryUnderstandingError | None,
    ) -> None:
        """Resolver 内部能力调用的观测事件，失败不得影响业务结果。"""

        try:
            self._runtime_events.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.GATEWAY_ANALYSIS_CAPABILITY_COMPLETED,
                    subsystem="gateway",
                    component="user_query_analysis",
                    severity="warning" if error is not None else "info",
                    data={
                        "capability_id": "query_understanding",
                        "duration_ms": (perf_counter() - started_at) * 1000,
                        "error": "query_understanding_failed" if error is not None else None,
                        "error_class": type(error).__name__ if error is not None else None,
                    },
                )
            )
        except Exception:
            return


__all__ = [
    "FallbackUserQueryAnalysisResolver",
    "LLMUserQueryAnalysisResolver",
]
