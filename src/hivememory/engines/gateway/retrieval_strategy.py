"""
Phase 3 Gateway 检索策略预选择决策原语。

Phase 3A 使用固定默认策略，后续阶段再接入细粒度配置与判断。
"""

from __future__ import annotations

from hivememory.engines.gateway.models import IntentType, RetrievalMode, RetrievalStrategy


class RetrievalStrategyEngine:
    """Phase 3 S4b 检索策略 engine 骨架。"""

    async def pick(
        self,
        *,
        intent_type: IntentType | None = None,
        target_topic: str | None = None,
    ) -> RetrievalStrategy:
        """返回保守默认检索策略。"""

        if intent_type == IntentType.CHAT:
            return RetrievalStrategy(
                mode=RetrievalMode.SKIP,
                top_k=0,
                dense_weight=0.0,
                sparse_weight=0.0,
                reason="闲聊意图默认跳过检索",
            )

        return RetrievalStrategy(
            mode=RetrievalMode.HYBRID,
            top_k=5,
            reason=f"Phase 3A 默认混合检索策略，target_topic={target_topic or 'UNKNOWN'}",
        )


__all__ = ["RetrievalStrategyEngine"]
