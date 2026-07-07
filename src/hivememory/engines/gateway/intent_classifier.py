"""
Phase 3 Gateway 意图分类决策原语。

该模块只提供可被 Stage 包裹的纯 engine 骨架，不依赖 SystemBus、
Patchouli 或 Alice。真实窄 Prompt 分类会在后续阶段接入。
"""

from __future__ import annotations

from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentClassificationResult,
    IntentType,
)


class IntentClassifierEngine:
    """Phase 3 S1 意图分类 engine 骨架。"""

    async def classify(
        self,
        message: str,
        *,
        gateway_intent: GatewayIntent | None = None,
    ) -> IntentClassificationResult:
        """
        生成最小可用的主意图分类结果。

        Phase 3A 不引入新 Prompt；若调用方提供现有 GatewayIntent，则按兼容映射
        输出 IntentType，否则采用保守 QUERY 默认值。
        """

        if gateway_intent == GatewayIntent.CHAT:
            return IntentClassificationResult(
                intent_type=IntentType.CHAT,
                confidence=1.0,
                reason="由兼容 GatewayIntent.CHAT 映射",
            )

        if gateway_intent == GatewayIntent.SYSTEM:
            return IntentClassificationResult(
                intent_type=IntentType.UNKNOWN,
                confidence=1.0,
                reason="系统指令应由 S0 短路，S1 仅保留 UNKNOWN 兼容映射",
            )

        return IntentClassificationResult(
            intent_type=IntentType.QUERY,
            confidence=0.5 if message.strip() else 0.0,
            reason="Phase 3A 默认查询意图",
        )


__all__ = ["IntentClassifierEngine"]
