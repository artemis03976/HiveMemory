from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

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
    ) -> Optional[MTPExecutionResult]:
        pass


class KoakumaMTPExecutor(MTPExecutor):
    """MTPExecutor adapter backed by KoakumaRuntime."""

    def __init__(self, koakuma: "KoakumaRuntime") -> None:
        self._koakuma = koakuma

    def set_cancel_event(self, cancel_event: Optional[asyncio.Event]) -> None:
        self._koakuma.cancel_event = cancel_event

    async def intercept_and_execute(
        self,
        assistant_text: str,
        context: MTPExecutionContext,
    ) -> Optional[MTPExecutionResult]:
        return await self._koakuma.intercept_and_execute(
            assistant_text,
            context=context,
        )


__all__ = [
    "KoakumaMTPExecutor",
    "MTPExecutor",
]
