"""S0 系统指令拦截 Stage。"""

from __future__ import annotations

from hivememory.engines.gateway.models import IntentType
from hivememory.gateway.commands import CommandRegistry
from hivememory.gateway.pipeline import GatewayState, ShortCircuit


class CommandInterceptorStage:
    """
    复用 Phase 2 command registry 的轻量拦截器。

    这里只做解析和短路，不执行 dispatcher 副作用。
    """

    stage_name = "S0.CommandInterceptor"

    def __init__(self, registry: CommandRegistry | None = None) -> None:
        self._registry = registry

    async def process(self, state: GatewayState) -> GatewayState:
        """命中 slash 指令时终止 Gateway Workflow。"""

        if self._registry is None:
            return state

        command = self._registry.match(state.raw_message)
        if command is None:
            return state

        state.command_result = command
        state.intent_type = IntentType.UNKNOWN
        raise ShortCircuit(state)


__all__ = ["CommandInterceptorStage"]
