from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivememory.patchouli.runtime import PatchouliRuntime


class ModelReadinessService:
    """Patchouli model readiness public API."""

    def __init__(self, runtime: "PatchouliRuntime") -> None:
        self._runtime = runtime

    async def warmup_models(self) -> None:
        await self._runtime.warmup_models()

    async def is_models_ready(self) -> bool:
        return self._runtime.is_models_ready()
