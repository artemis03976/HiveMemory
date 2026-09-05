from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.models import (
    ActorIdentity,
    Artifacts,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    resolve_default_identity_scope,
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
        user_id: str,
    ) -> MemoryAtom:
        # TODO: 复核 agent_id 的手写来源
        identity_scope = resolve_default_identity_scope(
            ActorIdentity(user_id=user_id, agent_id="ui"),
        )
        atom = MemoryAtom(
            meta=MetaData(
                workspace_identity=identity_scope.workspace_identity,
                source_agent_id=identity_scope.actor_identity.agent_id,
                source_team_id=identity_scope.actor_identity.team_id,
                access_policy=MemoryAccessPolicy.public(),
            ),
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
            identity_scope,
            atom,
        )

    async def list_agent_profiles(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        identity_scope = resolve_default_identity_scope(
            ActorIdentity(user_id=user_id, agent_id="ui"),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            identity_scope=identity_scope,
            limit=limit,
        )

    @staticmethod
    def _default_summary(title: str) -> str:
        summary = title.strip() or "Agent Profile"
        if len(summary) >= 10:
            return summary
        return f"{summary} agent profile"
