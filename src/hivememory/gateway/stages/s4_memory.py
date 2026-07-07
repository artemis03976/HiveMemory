"""S4a 记忆写入价值预判 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway import MemoryValueJudgeEngine
from hivememory.engines.gateway.models import IntentType, MemoryWriteSignal
from hivememory.gateway.pipeline import GatewayState, StageTrace


class MemoryValueJudgeStage:
    """调用 MemoryValueJudgeEngine 产出 memory_write_signal。"""

    stage_name = "S4a.MemoryValueJudge"

    def __init__(self, engine: MemoryValueJudgeEngine | None = None) -> None:
        self._engine = engine or MemoryValueJudgeEngine()

    async def process(self, state: GatewayState) -> GatewayState:
        """S3 已派生写入信号时保持不变，否则使用默认 engine 判断。"""

        if state.memory_write_signal is not None:
            return state

        try:
            signal = await self._engine.judge(
                state.raw_message,
                intent_type=IntentType(state.intent_type) if state.intent_type else None,
            )
        except Exception as exc:  # pragma: no cover - 防御性 fallback
            state.memory_write_signal = MemoryWriteSignal.UNKNOWN
            state.stage_trace.append(
                StageTrace(
                    stage_name=self.stage_name,
                    duration_ms=0.0,
                    is_fallback=True,
                    fallback_reason=f"S4a 价值判断失败：{exc}",
                )
            )
            return state

        state.memory_write_signal = MemoryWriteSignal(signal)
        return state


__all__ = ["MemoryValueJudgeStage"]
