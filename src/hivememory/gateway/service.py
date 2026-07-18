"""GatewayService：Gateway 子系统业务入口。"""

from __future__ import annotations

from typing import Any

from hivememory.core.models import Identity
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
    ) -> Any:
        """把一次 Gateway 请求完整委托给 Runtime 持有的 workflow。"""

        return await self._runtime.workflow.run(message, identity=identity)


__all__ = ["GatewayService"]
