from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from hivememory.alice.runtime.models import MTPExecutionContext
from hivememory.core.protocol.models import MTPExecutionResult

if TYPE_CHECKING:
    from hivememory.alice.runtime.koakuma import KoakumaRuntime


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
