"""Agent Profile 能力：Profile 定义读取 + 管理用例薄委托（A2 §1.2，自 ``system/application`` 迁入）。

- ``get_agent_profile``：``profile.read`` 授权在 backing 调用前执行，随后经
  workspace Profile 读取 resolver 按 ``(Workspace, agent_alias)`` 命中或冷读，
  交付边界按源原子 policy 逐次授权（A2 §2.3）；对 actor 只交付 AgentProfile；
- 管理写入/列表（``management.memory`` 绑定例外）的授权同样在本层、
  路由调用前执行。

身份与访问约定（A1 访问边界返工第 4.5 节）：本层是授权点——方法只接收
访问 context 与目标 workspace，``IdentityScope`` 由操作授权者组装，不接收
调用方传入的 scope。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import InvalidMemoryFieldError
from hivememory.core.models import (
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
    from hivememory.core.models import AgentProfile, WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.profile import ProfileResolver


class AgentApplicationService:
    """Agent profile API use-case service.

    身份入口约定（v0.6.2 收敛）：Agent Profile 管理是用户导向的管理用例，
    不是具体 Agent 的执行动作，因此 server 边界以 ``system`` actor 的声明
    认证；``source_agent_id`` 只作 provenance 展示。

    访问上下文约定（A1 访问边界返工第 4.5 节）：方法的 ``access`` 为统一
    认证网关签发的可信 context，``management.memory`` / ``profile.read``
    授权在本层、路由调用前执行；Patchouli 只接收操作授权者返回的可信
    scope，不再接收 context。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        profile_reader: ProfileResolver,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._profile_reader = profile_reader

    async def create_agent_profile(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        title: str,
        alias: str,
        summary: str = "",
        content: str = "",
        tags: list[str],
        agent_config: dict[str, Any] | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """在目标 Workspace 中创建 Agent Profile（``management.memory`` 管理用例）。"""
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace
        )
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
                workspace_identity=scope.workspace_identity,
                provenance=MemoryProvenance(
                    source_agent_id=scope.actor_identity.agent_id,
                    source_team_id=scope.actor_identity.team_id,
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
            scope,
            atom,
        )

    async def list_agent_profiles(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        limit: int = 100,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """在目标 Workspace 中列出 Agent Profile（``management.memory`` 管理用例）。"""
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            identity_scope=scope,
            limit=limit,
        )

    async def get_agent_profile(
        self,
        agent_alias: str | None,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> AgentProfile:
        """Profile 定义读取（``profile.read``）：交付 AgentProfile 独立副本。

        未指定 alias 或显式选择 ``default`` / ``omni_doll`` 时返回内置 Profile；
        自定义 alias 缺失、不可见、类型不符或配置损坏均显式失败，不降级为
        默认配置。读取 Profile 不等于获得其描述的权限。
        """
        scope = self._authorizer.authorize_operation(
            access, WorkspaceOperation.PROFILE_READ, target_workspace
        )
        return await self._profile_reader.get(agent_alias, scope=scope)
