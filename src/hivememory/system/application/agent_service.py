from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hivememory.core.models import (
    Artifacts,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    IdentityScope,
)
from hivememory.system.contracts.routes import GlobalRoutes

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.system.access import WorkspaceAccessContext


class AgentApplicationService:
    """Agent profile API use-case service.

    身份入口约定（v0.6.2 收敛）：Agent Profile 管理是用户导向的管理用例，
    不是具体 Agent 的执行动作，因此 server 边界为其冻结 ``system`` actor
    的 IdentityScope；``source_agent_id`` 只作 provenance 展示。

    访问上下文约定（A1 计划）：``access`` 为统一认证网关签发的可信
    context，原样透传给 Patchouli 公共路由，行为检查在 application 落实。
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
        identity_scope: IdentityScope,
        title: str,
        alias: str,
        summary: str = "",
        content: str = "",
        tags: list[str],
        agent_config: dict[str, Any] | None = None,
        access: "WorkspaceAccessContext | None" = None,
    ) -> MemoryAtom:
        """在显式 Workspace scope 中创建 Agent Profile（管理用例）。"""
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
            access=access,
        )

    async def list_agent_profiles(
        self,
        *,
        identity_scope: IdentityScope,
        limit: int = 100,
        access: "WorkspaceAccessContext | None" = None,
    ) -> list[MemoryAtom]:
        """在显式 Workspace scope 中列出 Agent Profile（管理用例）。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            identity_scope=identity_scope,
            limit=limit,
            access=access,
        )

    @staticmethod
    def _default_summary(title: str) -> str:
        summary = title.strip() or "Agent Profile"
        if len(summary) >= 10:
            return summary
        return f"{summary} agent profile"
