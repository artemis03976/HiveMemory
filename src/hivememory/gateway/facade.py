"""Phase 3 GatewayFacade 最小骨架。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.gateway.context import GatewayContextBuilder, SessionContext
from hivememory.gateway.eye import TheEye
from hivememory.gateway.pipeline import (
    GatewayPipeline,
    GatewayStage,
    GatewayState,
    ShortCircuit,
    StageTrace,
)
from hivememory.gateway.commands import SystemCommandDispatcher


@dataclass(frozen=True)
class GatewayFacade:
    """
    Gateway 子系统对外入口骨架。

    Phase 3A 仅提供兼容 gaze 代理和 process 落点；完整 GatewayState Pipeline
    在 Phase 3B/3C 接入。
    """

    eye: TheEye
    command_dispatcher: SystemCommandDispatcher | None = None
    context_builder: GatewayContextBuilder | None = None
    pipeline: GatewayPipeline | None = None
    command_interceptor: GatewayStage | None = None

    async def gaze(
        self,
        query: str,
        topic_snapshots: Sequence[TopicSnapshot] | None = None,
        identity: Identity | None = None,
    ) -> EyeGazeResult:
        """兼容 Phase 1/2 的 TheEye.gaze 调用。"""

        return await self.eye.gaze(
            query=query,
            topic_snapshots=topic_snapshots,
            identity=identity,
        )

    async def process(self, message: str, *, identity: Identity) -> GatewayState:
        """执行 Phase 3 Pipeline，返回封印后的 GatewayState。"""

        if self.context_builder is None or self.pipeline is None:
            raise RuntimeError("GatewayFacade 缺少 context_builder 或 pipeline")

        state = GatewayState(
            raw_message=message,
            session_context=SessionContext(identity=identity),
        )
        if self.command_interceptor is not None:
            state = await self._run_command_interceptor(state)
            if state.sealed:
                return state

        context = await self.context_builder.build(message=message, identity=identity)
        state.session_context = context
        return await self.pipeline.run_state(state)

    async def _run_command_interceptor(self, state: GatewayState) -> GatewayState:
        """在 context hydration 前执行 S0，避免命中 command 时加载话题。"""

        assert self.command_interceptor is not None
        stage_name = getattr(
            self.command_interceptor,
            "stage_name",
            self.command_interceptor.__class__.__name__,
        )
        trace_count = len(state.stage_trace)
        start = monotonic()
        try:
            state = await self.command_interceptor.process(state)
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
                StageTrace(
                    stage_name=stage_name,
                    duration_ms=duration_ms,
                )
            )
        return state


__all__ = ["GatewayFacade"]
