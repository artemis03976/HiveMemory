"""Phase 3 GatewayFacade 最小骨架。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hivememory.core.models import Identity, TopicSnapshot
from hivememory.core.protocol.models import EyeGazeResult
from hivememory.gateway.context import GatewayContextBuilder
from hivememory.gateway.eye import TheEye
from hivememory.gateway.pipeline import GatewayPipeline
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

    async def process(self, message: str, *, identity: Identity) -> object:
        """Phase 3B GatewayState Pipeline 的预留入口。"""

        _ = (message, identity)
        raise NotImplementedError("GatewayFacade.process 将在 Phase 3B 接入 GatewayState")


__all__ = ["GatewayFacade"]
