from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import (
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
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.utils.time import utc_now

if TYPE_CHECKING:
    from hivememory.system.config import HiveMemoryConfig
    from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
    from hivememory.workspace import WorkspaceAccessContext


class MemoryLifecycleUnavailableError(RuntimeError):
    """生命周期反馈操作不可用时抛出。"""


class MemoryNotFoundError(ValueError):
    """请求的记忆不存在时抛出。"""


class MemoryApplicationService:
    """Memory API use-case service.

    HTTP routers call this service instead of reaching into Patchouli internals.

    身份入口约定（v0.6.2 收敛）：所有用例只接受 server 边界一次性冻结的
    ``IdentityScope``，不在服务内解析裸 ``user_id`` 或默认 Agent。管理用例
    （本服务的全部读写）按 owner-management 语义在 Workspace ownership
    hard boundary 内访问该 Workspace 的全部 Memory，不执行 Agent 级
    ``MemoryAccessPolicy`` 可见性过滤；``system`` actor 只标记"没有具体
    Agent 作为操作来源主体"，不承担任何权限绕过语义。

    访问上下文约定（A1 计划第 1.2/3.3 节）：全部用例接收统一认证网关
    签发的 ``WorkspaceAccessContext`` 并**原样透传**给 Patchouli 公共
    路由，最终行为检查在 Patchouli application 落实；本层不解释、不裁剪
    access，也不以 DTO scope 覆盖可信坐标。``access`` 缺省时依赖下游
    冻结的迁移期兼容分支（管理入口 HTTP 链路），A6 切换生产入口后收紧。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        config: HiveMemoryConfig,
    ) -> None:
        self._global_bus = global_bus
        self._config = config

    @property
    def config(self) -> HiveMemoryConfig:
        return self._config

    async def create_memory(
        self,
        *,
        identity_scope: IdentityScope,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        """管理创建入口：在显式 Workspace scope 中创建 Memory。

        ``provenance.source_agent_id`` 记录来源 actor（管理入口为保留
        ``system``），只作 provenance 展示，不参与可见性授权。
        """
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
            index=IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType(memory_type),
                alias=alias,
            ),
            payload=PayloadLayer(
                content=content,
            ),
        )
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_CREATE,
            identity_scope,
            atom,
            access=access,
        )

    async def list_memories(
        self,
        *,
        identity_scope: IdentityScope,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
        access: WorkspaceAccessContext | None = None,
    ) -> list[MemoryAtom]:
        """管理读取入口：在显式 Workspace scope 中列出 Memory。

        按 owner-management 语义返回该 Workspace 的全部 Memory（不含
        Agent Profile），不做 Agent ``MemoryAccessPolicy`` 过滤。
        """
        filters = self._build_filters(memory_type=memory_type)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_LIST,
            identity_scope=identity_scope,
            query=query,
            filters=filters if filters else None,
            limit=limit,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
            refresh_vitality=True,
            access=access,
        )

    async def get_memory(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        """管理读取入口：在显式 Workspace scope 中读取 Memory。"""
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            memory_id,
            identity_scope=identity_scope,
            refresh_vitality=True,
            access=access,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def update_memory(
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
        access: WorkspaceAccessContext | None = None,
    ) -> MemoryAtom:
        """管理更新入口：显式授权 mutation，且不改变原 ownership/provenance。"""
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
            access=access,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def record_feedback(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        positive: bool,
        source: str,
        access: WorkspaceAccessContext | None = None,
    ):
        """管理反馈入口：在显式 Workspace scope 中记录反馈。"""
        try:
            return await self._global_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK,
                memory_id,
                identity_scope=identity_scope,
                positive=positive,
                source=source,
                access=access,
            )
        except WorkspaceDomainError:
            # 访问/领域受控错误必须按原语义传播（A1 第 3.4 节），
            # 不得被通用 RuntimeError 分支包装成"服务不可用"。
            raise
        except RuntimeError as exc:
            raise MemoryLifecycleUnavailableError("Memory lifecycle engine is unavailable") from exc
        except ValueError as exc:
            raise MemoryNotFoundError(str(exc)) from exc

    async def delete_memory(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        access: WorkspaceAccessContext | None = None,
    ) -> bool:
        """管理删除入口：在显式 Workspace scope 中删除 Memory。"""
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_DELETE,
            memory_id,
            identity_scope=identity_scope,
            access=access,
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
