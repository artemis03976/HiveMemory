"""GatewayService：Gateway 子系统业务入口。"""

from __future__ import annotations

from time import monotonic

from hivememory.core.models import Identity
from hivememory.gateway.context import SessionContext
from hivememory.gateway.pipeline import GatewayState, ShortCircuit, StageTrace
from hivememory.gateway.runtime import GatewayRuntime


class GatewayService:
    """
    Gateway 子系统业务门面。

    调用方只通过 process 进入 Gateway Workflow，不感知 Pipeline 内部装配。
    """

    def __init__(self, runtime: GatewayRuntime) -> None:
        self._runtime = runtime

    async def process(
        self,
        message: str,
        *,
        identity: Identity,
    ) -> GatewayState:
        """
        执行 Gateway 决策流程。

        Phase 3A 先执行 S0 command 短路；未命中时再构造 SessionContext，
        并运行最小空 Pipeline。
        """

        state = GatewayState(
            raw_message=message,
            session_context=SessionContext(identity=identity),
        )
        state = await self._run_command_interceptor(state)
        if state.sealed:
            return state

        context = await self._runtime.context_builder.build(
            message=message,
            identity=identity,
        )
        state.session_context = context
        return await self._runtime.pipeline.run_state(state)

    async def _run_command_interceptor(self, state: GatewayState) -> GatewayState:
        """在 context hydration 前执行 S0，避免 command 命中时加载话题。"""

        stage = self._runtime.command_interceptor
        stage_name = getattr(stage, "stage_name", stage.__class__.__name__)
        trace_count = len(state.stage_trace)
        start = monotonic()
        try:
            state = await stage.process(state)
        except ShortCircuit as short_circuit:
            state = short_circuit.state
            duration_ms = (monotonic() - start) * 1000
            if len(state.stage_trace) == trace_count:
                state.stage_trace.append(
                    StageTrace(
                        stage_name=stage_name,
                        duration_ms=duration_ms,
                        short_circuited=True,
                    )
                )
            return state.seal()

        duration_ms = (monotonic() - start) * 1000
        if len(state.stage_trace) == trace_count:
            state.stage_trace.append(
                StageTrace(stage_name=stage_name, duration_ms=duration_ms)
            )
        return state


__all__ = ["GatewayService"]
