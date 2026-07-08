"""S0 入口拦截 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway.interfaces import BaseInterceptor
from hivememory.engines.gateway.models import (
    GatewayIntent,
    IntentType,
)
from hivememory.gateway.pipeline import GatewayState, StageResult


class EntryInterceptorStage:
    """
    S0 入口拦截器。

    复用 engines.gateway 的 L1 interceptor，统一处理系统指令与简单闲聊。
    这里只做解析、写入 state 和短路，不执行 dispatcher 副作用。
    """

    stage_name = "S0.EntryInterceptor"
    writable_fields = frozenset(
        {
            "l1_result",
            "intent_type",
            "command_parse_result",
        }
    )

    def __init__(self, interceptor: BaseInterceptor) -> None:
        self._interceptor = interceptor

    async def process(self, state: GatewayState) -> StageResult:
        """
        命中系统指令或简单闲聊时终止后续 Gateway Workflow。
        """

        result = self._interceptor.intercept(state.raw_message)
        if result is None or not result.hit:
            return StageResult.empty()

        if result.intent == GatewayIntent.SYSTEM:
            return StageResult.from_updates(
                {
                    "l1_result": result,
                    "intent_type": IntentType.UNKNOWN,
                    "command_parse_result": result.command,
                },
                flow_end_reason="system_command",
            )

        if result.intent == GatewayIntent.CHAT:
            return StageResult.from_updates(
                {
                    "l1_result": result,
                    "intent_type": IntentType.CHAT,
                },
                flow_end_reason="simple_chat",
            )

        return StageResult.from_updates({"l1_result": result})


__all__ = ["EntryInterceptorStage"]
