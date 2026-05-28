from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.models import (
    Artifacts,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class AgentApplicationService:
    """Agent profile API use-case service.

    Phase 1 only establishes the dependency boundary. Router behavior will be
    migrated into this service in later phases.
    """

    def __init__(
        self,
        global_bus: "GlobalSystemBus",
        config: "HiveMemoryConfig",
    ) -> None:
        self._global_bus = global_bus
        self._config = config

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    async def create_agent_profile(
        self,
        *,
        title: str,
        alias: str,
        summary: str = "",
        content: str = "",
        tags: list[str],
        agent_config: dict[str, Any] | None = None,
    ) -> MemoryAtom:
        atom = MemoryAtom(
            meta=MetaData(source_agent_id="ui", user_id="default"),
            index=IndexLayer(
                title=title,
                summary=summary or self._default_summary(title),
                tags=tags,
                memory_type=MemoryType.AGENT_PROFILE,
                alias=alias,
            ),
            payload=PayloadLayer(
                content=content,
                artifacts=Artifacts(agent_config=agent_config),
            ),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE,
            atom,
        )

    async def list_agent_profiles(self, *, limit: int = 100) -> list[MemoryAtom]:
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            limit=limit,
        )

    @staticmethod
    def _default_summary(title: str) -> str:
        summary = title.strip() or "Agent Profile"
        if len(summary) >= 10:
            return summary
        return f"{summary} agent profile"
