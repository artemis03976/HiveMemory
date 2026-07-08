"""
Phase 3 Gateway 意图分类决策原语。

该模块只提供可被 Stage 包裹的纯 engine 骨架，不依赖 SystemBus、
Patchouli 或 Alice。真实窄 Prompt 分类会在后续阶段接入。
"""

from __future__ import annotations

from hivememory.engines.gateway.models import (
    IntentClassificationResult,
    IntentType,
)


class IntentClassifierEngine:
    """Phase 3 S1 意图分类 engine 骨架。"""

    async def classify(
        self,
        message: str,
    ) -> IntentClassificationResult:
        """
        生成最小可用的主意图分类结果。

        Phase 3B 不引入新 Prompt；S0 已经负责 command 短路，S1 只根据
        当前消息给出保守的单主意图默认值。
        """

        if not message.strip():
            return IntentClassificationResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                reason="空消息无法分类",
            )

        return IntentClassificationResult(
            intent_type=IntentType.QUERY,
            confidence=0.5,
            reason="Phase 3B 默认查询意图",
        )


__all__ = ["IntentClassifierEngine"]
