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
        storage: Any | None = None,
    ) -> None:
        self._global_bus = global_bus
        self._config = config
        self._storage_backend = storage

    @property
    def config(self) -> "HiveMemoryConfig":
        return self._config

    def bind_storage(self, storage: Any) -> None:
        self._storage_backend = storage

    @property
    def _storage(self):
        if self._storage_backend is None:
            raise RuntimeError("Storage is not bound to AgentApplicationService")
        return self._storage_backend

    def create_agent_profile(
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
        self._storage.upsert_memory(atom)
        return atom

    def list_agent_profiles(self, *, limit: int = 100) -> list[MemoryAtom]:
        return self._storage.get_all_memories(
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=limit,
        )

    @staticmethod
    def _default_summary(title: str) -> str:
        summary = title.strip() or "Agent Profile"
        if len(summary) >= 10:
            return summary
        return f"{summary} agent profile"
