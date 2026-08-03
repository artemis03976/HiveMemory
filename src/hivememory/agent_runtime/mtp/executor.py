"""MTP 执行端口及 Koakuma 适配器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.core.protocol.models import MTPExecutionResult

if TYPE_CHECKING:
    from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime


class MTPExecutor(ABC):
    """Narrow port used by AgentRuntime to execute MTP commands."""

    @abstractmethod
    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
        *,
        cancel_event=None,
    ) -> MTPExecutionResult | None:
        pass


class KoakumaMTPExecutor(MTPExecutor):
    """MTPExecutor adapter backed by KoakumaRuntime."""

    def __init__(self, koakuma: KoakumaRuntime) -> None:
        self._koakuma = koakuma

    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
        *,
        cancel_event=None,
    ) -> MTPExecutionResult | None:
        return await self._koakuma.intercept_and_execute(
            assistant_text,
            context=context,
            cancel_event=cancel_event,
        )


__all__ = [
    "KoakumaMTPExecutor",
    "MTPExecutor",
]
