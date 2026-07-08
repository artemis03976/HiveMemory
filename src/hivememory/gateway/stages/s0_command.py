"""S0 入口拦截 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway.interfaces import BaseInterceptor
from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalStrategy,
)
from hivememory.gateway.pipeline import GatewayFlowEnded, GatewayState


class EntryInterceptorStage:
    """
    S0 入口拦截器。

    复用 engines.gateway 的 L1 interceptor，统一处理系统指令与简单闲聊。
    这里只做解析、写入 state 和短路，不执行 dispatcher 副作用。
    """

    stage_name = "S0.EntryInterceptor"

    def __init__(self, interceptor: BaseInterceptor) -> None:
        self._interceptor = interceptor

    async def process(self, state: GatewayState) -> GatewayState:
        """
        命中系统指令或简单闲聊时终止后续 Gateway Workflow。
        """

        result = self._interceptor.intercept(state.raw_message)
        if result is None or not result.hit:
            return state

        state.l1_result = result

        if result.intent == GatewayIntent.SYSTEM:
            state.intent_type = IntentType.UNKNOWN
            state.command_parse_result = result.command
            state.flow_end_reason = "system_command"
            raise GatewayFlowEnded(state)

        if result.intent == GatewayIntent.CHAT:
            state.intent_type = IntentType.CHAT
            state.rewritten_query = state.raw_message
            state.search_keywords = []
            state.memory_write_signal = MemoryWriteSignal.SKIP
            state.retrieval_strategy = RetrievalStrategy(mode=RetrievalMode.SKIP)
            state.flow_end_reason = "simple_chat"
            raise GatewayFlowEnded(state)

        return state


__all__ = ["EntryInterceptorStage"]
