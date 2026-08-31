from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.core.models import (
    Artifacts,
    Identity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    IdentityScope,
    resolve_default_identity_scope,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


class MemoryLifecycleUnavailableError(RuntimeError):
    """Raised when lifecycle feedback operations are unavailable."""


class MemoryNotFoundError(ValueError):
    """Raised when the requested memory does not exist."""


class MemoryApplicationService:
    """Memory API use-case service.

    HTTP routers call this service instead of reaching into Patchouli internals.
    This phase preserves the existing storage behavior while narrowing the
    server-facing dependency boundary.
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

    async def create_memory(
        self,
        *,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
        user_id: str,
    ) -> MemoryAtom:
        identity_scope = self._default_identity_scope(user_id)
        return await self.create_memory_scoped(
            identity_scope=identity_scope,
            title=title,
            summary=summary,
            content=content,
            memory_type=memory_type,
            tags=tags,
            alias=alias,
        )

    async def create_memory_scoped(
        self,
        *,
        identity_scope: IdentityScope,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
    ) -> MemoryAtom:
        """内部 seam：使用调用方提供的完整 Workspace scope 创建 Memory。"""
        atom = MemoryAtom(
            meta=MetaData(
                workspace_identity=identity_scope.workspace_identity,
                source_agent_id=identity_scope.actor_identity.agent_id,
                source_team_id=identity_scope.actor_identity.team_id,
                access_policy=MemoryAccessPolicy.public(),
            ),
            index=IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType(memory_type),
                alias=alias,
            ),
            payload=PayloadLayer(
                content=content,
                artifacts=Artifacts(),
            ),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_CREATE,
            identity_scope,
            atom,
        )

    async def list_memories(
        self,
        *,
        query: str | None = None,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        return await self.list_memories_scoped(
            identity_scope=self._default_identity_scope(user_id),
            query=query,
            memory_type=memory_type,
            limit=limit,
        )

    async def list_memories_scoped(
        self,
        *,
        identity_scope: IdentityScope,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryAtom]:
        """内部 seam：在显式 Workspace scope 中列出 Memory。"""
        filters = self._build_filters(memory_type=memory_type)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_LIST,
            identity_scope=identity_scope,
            query=query,
            filters=filters if filters else None,
            limit=limit,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
            refresh_vitality=True,
        )

    async def get_memory(self, memory_id: UUID, *, user_id: str) -> MemoryAtom:
        return await self.get_memory_scoped(
            memory_id,
            identity_scope=self._default_identity_scope(user_id),
        )

    async def get_memory_scoped(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
    ) -> MemoryAtom:
        """内部 seam：在显式 Workspace scope 中读取 Memory。"""
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            memory_id,
            identity_scope=identity_scope,
            refresh_vitality=True,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def update_memory(
        self,
        memory_id: UUID,
        *,
        user_id: str,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom:
        return await self.update_memory_scoped(
            memory_id,
            identity_scope=self._default_identity_scope(user_id),
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )

    async def update_memory_scoped(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom:
        """内部 seam：显式授权 mutation，且不改变原 ownership/provenance。"""
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_UPDATE,
            memory_id,
            identity_scope=identity_scope,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def record_feedback(
        self,
        memory_id: UUID,
        *,
        user_id: str,
        positive: bool,
        source: str,
    ):
        return await self.record_feedback_scoped(
            memory_id,
            identity_scope=self._default_identity_scope(user_id),
            positive=positive,
            source=source,
        )

    async def record_feedback_scoped(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        positive: bool,
        source: str,
    ):
        """内部 seam：在显式 Workspace scope 中记录反馈。"""
        try:
            return await self._global_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK,
                memory_id,
                identity_scope=identity_scope,
                positive=positive,
                source=source,
            )
        except RuntimeError as exc:
            raise MemoryLifecycleUnavailableError(
                "Memory lifecycle engine is unavailable"
            ) from exc
        except ValueError as exc:
            raise MemoryNotFoundError(str(exc)) from exc

    async def delete_memory(self, memory_id: UUID, *, user_id: str) -> bool:
        return await self.delete_memory_scoped(
            memory_id,
            identity_scope=self._default_identity_scope(user_id),
        )

    async def delete_memory_scoped(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
    ) -> bool:
        """内部 seam：在显式 Workspace scope 中删除 Memory。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_DELETE,
            memory_id,
            identity_scope=identity_scope,
        )

    @staticmethod
    def _build_filters(
        *,
        memory_type: str | None,
    ) -> dict[str, str]:
        filters = {}
        if memory_type:
            filters["index.memory_type"] = memory_type
        return filters

    @staticmethod
    def _default_identity_scope(user_id: str) -> IdentityScope:
        """HTTP/System 顶层为当前用户一次性解析默认 Workspace。"""
        return resolve_default_identity_scope(
            Identity(user_id=user_id, agent_id="ui"),
        )
