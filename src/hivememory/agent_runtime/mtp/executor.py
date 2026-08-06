"""MTP 执行端口及 Koakuma 适配器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.core.protocol.models import MTPExecutionResult

if TYPE_CHECKING:
    from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime


class MTPExecutor(ABC):
    """AgentRuntime 消费的最窄 MTP 执行端口。"""

    @abstractmethod
    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
    ) -> MTPExecutionResult | None:
        pass


class KoakumaMTPExecutor(MTPExecutor):
    """由 KoakumaRuntime 支撑的 MTPExecutor 适配器。"""

    def __init__(self, koakuma: KoakumaRuntime) -> None:
        self._koakuma = koakuma

    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
    ) -> MTPExecutionResult | None:
        return await self._koakuma.intercept_and_execute(
            assistant_text,
            context=context,
        )


__all__ = [
    "KoakumaMTPExecutor",
    "MTPExecutor",
]
