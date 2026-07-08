"""
GatewayService：Gateway 子系统业务入口。
"""

from __future__ import annotations

from time import monotonic

from hivememory.core.models import Identity
from hivememory.gateway.commands import CommandExecutionResult, CommandExecutionStatus
from hivememory.gateway.context import SessionContext
from hivememory.gateway.pipeline import GatewayState, ShortCircuit, StageTrace
from hivememory.gateway.runtime import GatewayRuntime


class GatewayService:
    """
    Gateway 子系统业务门面。

    调用方只通过 process 进入 Gateway Workflow。
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

        先执行 S0 command 短路；未命中时再构造 SessionContext，
        并运行最小空 Pipeline。
        """

        state = GatewayState(
            raw_message=message,
            session_context=SessionContext(identity=identity),
        )
        state = await self._run_entry_interceptor(state, identity=identity)
        if state.sealed:
            return state

        context = await self._runtime.context_builder.build(
            message=message,
            identity=identity,
        )
        state.session_context = context
        return await self._runtime.pipeline.run_state(state)

    async def _run_entry_interceptor(
        self,
        state: GatewayState,
        *,
        identity: Identity,
    ) -> GatewayState:
        """
        在 context hydration 前执行 S0，避免命中时加载话题。
        """

        stage = self._runtime.entry_interceptor
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
            await self._dispatch_system_command_if_needed(state, identity=identity)
            return state.seal()

        duration_ms = (monotonic() - start) * 1000
        if len(state.stage_trace) == trace_count:
            state.stage_trace.append(
                StageTrace(stage_name=stage_name, duration_ms=duration_ms)
            )
        return state

    async def _dispatch_system_command_if_needed(
        self,
        state: GatewayState,
        *,
        identity: Identity,
    ) -> None:
        """
        S0 命中系统指令后，由 Service 编排 dispatcher 执行副作用。
        """

        if state.short_circuit_reason != "system_command":
            return

        dispatcher = self._runtime.command_dispatcher
        if dispatcher is None:
            command_id = (
                state.command_parse_result.command_id
                if state.command_parse_result is not None
                else None
            )
            command_name = (
                state.command_parse_result.name
                if state.command_parse_result is not None
                else None
            )
            state.command_execution_result = CommandExecutionResult(
                command_id=command_id or command_name or "unknown",
                status=CommandExecutionStatus.REJECTED,
                message="系统指令执行器未启用。",
                error_code="command.dispatcher.disabled",
            )
            return

        state.command_execution_result = await dispatcher.execute(
            state.command_parse_result,
            identity=identity,
        )


__all__ = ["GatewayService"]
