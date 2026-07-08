"""
Phase 3 Gateway 记忆价值预判断决策原语。

Phase 3B 只提供保守规则骨架，不执行写入。
"""

from __future__ import annotations

from hivememory.engines.gateway.models import IntentType, MemoryWriteSignal


class MemoryValueJudgeEngine:
    """Phase 3 S4a 记忆价值预判断 engine 骨架。"""

    async def judge(
        self,
        message: str,
        *,
        intent_type: IntentType | None = None,
    ) -> MemoryWriteSignal:
        """基于新 GatewayState 语义输出最小写入信号。"""

        if intent_type == IntentType.CHAT or not message.strip():
            return MemoryWriteSignal.SKIP
        return MemoryWriteSignal.UNKNOWN


__all__ = ["MemoryValueJudgeEngine"]
