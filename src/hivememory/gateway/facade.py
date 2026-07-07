"""Phase 3 GatewayFacade 最小骨架。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.gateway.context import GatewayContextBuilder
from hivememory.gateway.eye import TheEye
from hivememory.gateway.pipeline import GatewayPipeline, GatewayState
from hivememory.system.gateway.commands import SystemCommandDispatcher


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
        context = await self.context_builder.build(message=message, identity=identity)
        return await self.pipeline.run(message, context)


__all__ = ["GatewayFacade"]
