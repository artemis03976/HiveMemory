"""Agent Profile 能力：Profile 定义读取 + 管理用例薄委托（A2 §1.2，自 ``system/application`` 迁入）。

- ``get_agent_profile``：``profile.read`` 授权在 backing 调用前执行，随后经
  workspace Profile 读取 resolver 按 ``(Workspace, agent_alias)`` 命中或冷读，
  交付边界按源原子 policy 逐次授权（A2 §2.3）；对 actor 只交付 AgentProfile；
- 管理写入/列表保持对库管理路由的薄委托（``management.memory`` 绑定例外，
  检查仍由 Patchouli application 执行）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import InvalidMemoryFieldError
from hivememory.core.models import (
    AgentProfile,
    IdentityScope,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.utils.time import utc_now

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.workspace.access import WorkspaceAccessGuard
    from hivememory.workspace.resolution.profile import ProfileResolver


class AgentApplicationService:
    """Agent profile API use-case service.

    身份入口约定（v0.6.2 收敛）：Agent Profile 管理是用户导向的管理用例，
    不是具体 Agent 的执行动作，因此 server 边界为其冻结 ``system`` actor
    的 IdentityScope；``source_agent_id`` 只作 provenance 展示。

    访问上下文约定（A1 计划）：管理用例的 ``access`` 为统一认证网关签发的
    可信 context，原样透传给 Patchouli 公共管理路由，行为检查在 application
    落实；Profile 定义读取的 ``profile.read`` 在本层、backing 调用前检查。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        access_guard: WorkspaceAccessGuard,
        profile_reader: ProfileResolver,
    ) -> None:
        self._global_bus = global_bus
        self._access_guard = access_guard
        self._profile_reader = profile_reader

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
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        """在显式 Workspace scope 中创建 Agent Profile（管理用例）。"""
        # 只包装调用方提交字段的构造：输入不合法是 422，不是程序错误。
        try:
            index = IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType.AGENT_PROFILE,
                alias=alias,
            )
            payload = PayloadLayer(content=content, agent_config=agent_config)
        except ValidationError as exc:
            raise InvalidMemoryFieldError.from_validation_error(exc) from exc
        # A2-P：创建时点在提交边界取一次 now，created/updated/decay anchor 同值；
        # MVL-2 收敛后统一由 Patchouli 完整写入路径赋值。
        now = utc_now()
        atom = MemoryAtom(
            meta=MetaData(
                workspace_identity=identity_scope.workspace_identity,
                provenance=MemoryProvenance(
                    source_agent_id=identity_scope.actor_identity.agent_id,
                    source_team_id=identity_scope.actor_identity.team_id,
                ),
                access_policy=MemoryAccessPolicy.public(),
                created_at=now,
                updated_at=now,
                lifecycle=MemoryLifecycleState(decay_anchor_at=now),
            ),
            index=index,
            payload=payload,
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
        access: WorkspaceAccessContext | None = None,
    ) -> list[MemoryAtom]:
        """在显式 Workspace scope 中列出 Agent Profile（管理用例）。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            identity_scope=identity_scope,
            limit=limit,
            access=access,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """Profile 定义读取（``profile.read``）：交付 AgentProfile 独立副本。

        未指定 alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；
        自定义 alias 缺失、不可见、类型不符或配置损坏均显式失败，不降级为
        默认配置。读取 Profile 不等于获得其描述的权限。
        """
        scope = self._access_guard.authorize_operation(access, WorkspaceOperation.PROFILE_READ)
        return await self._profile_reader.get(agent_alias, scope=scope, access=access)
