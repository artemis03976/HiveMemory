"""Memory 能力：Actor 可见读取 + 管理用例薄委托（A2 §1.2，自 ``system/application`` 迁入）。

能力层是 in-process 的 workspace server API（宪章 §5.3）：actor 经 HTTP/MTP/
外部 adapter 归一化后调用本模块，本模块作为 client 调用 Patchouli backing。

- Actor 可见读取（``read`` / ``retrieve_by_aliases`` / ``retrieve``）：
  ``resource.read`` / ``resource.search`` 授权在 backing 调用前执行，随后
  经 workspace alias resolver 多级解析并在交付边界逐次授权（A2 §2.2）；
- 管理用例（create/list/get/update/delete/feedback）：``management.memory``
  授权同样在本层、路由调用前执行。

身份与访问约定（A1 访问边界返工第 4.5 节）：本层是授权点——方法只接收
访问 context 与目标 workspace，先用操作授权者的 ``authorize_operation``
取得可信 ``IdentityScope``，再用它构造领域对象与检索请求并调用
Patchouli；context 不向下传递，调用方也不能另行传入 scope 或携带身份
的检索请求。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import (
    InvalidMemoryFieldError,
    PendingUpdateNotAllowedError,
    ResourceNotFoundError,
    WorkspaceDomainError,
)
from hivememory.core.models import (
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
    PendingAtom,
    ReferenceResolution,
    UpdateFocus,
    WriteFocus,
)
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.utils.time import utc_now
from hivememory.utils.uuid import normalize_uuid

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import IdentityScope, WorkspaceIdentity
    from hivememory.core.models.query import QueryFilters
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.alias import AliasResolver


class MemoryLifecycleUnavailableError(RuntimeError):
    """生命周期反馈操作不可用时抛出。"""


class MemoryNotFoundError(ValueError):
    """请求的记忆不存在时抛出。"""


class MemoryApplicationService:
    """Memory 能力入口（类名沿用迁移前名称，A6 收口时评估更名）。

    HTTP routers call this service instead of reaching into Patchouli internals.

    操作授权统一在本层、backing/管理路由调用前执行：``read`` /
    ``retrieve_by_aliases`` → ``resource.read``，``retrieve`` →
    ``resource.search``；管理用例（create/list/get/update/delete/feedback，
    含 Agent Profile 的既有绑定例外）→ ``management.memory``。方法显式
    接收目标 workspace（当前只接受等于 context 驻留 workspace 的目标），
    ``IdentityScope`` 由操作授权者组装，不接收调用方传入的 scope。

    管理用例按 owner-management 语义在 Workspace ownership hard boundary
    内访问该 Workspace 的全部 Memory，不执行 Agent 级
    ``MemoryAccessPolicy`` 可见性过滤；``system`` actor 只标记"没有具体
    Agent 作为操作来源主体"，不承担任何权限绕过语义。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        memory_reader: AliasResolver,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._reader = memory_reader
        # 提交与 L0 回读必须是同一份登记：只经读取视图取得，不另行注入。
        self._intents = memory_reader.intents

    async def submit_write_intent(
        self,
        focus: WriteFocus,
        *,
        process_id: str | None,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> PendingAtom:
        """提交 WRITE（``memory_intent.submit``），ACK 仅表示意图已登记。"""
        scope = self._authorize(access, WorkspaceOperation.MEMORY_INTENT_SUBMIT, target_workspace)
        return self._intents.register_write(
            focus,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
        )

    async def submit_update_intent(
        self,
        base_alias: str,
        instruction: str,
        content: str | None = None,
        *,
        process_id: str | None,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> PendingAtom:
        """提交 UPDATE：授权后解析可读正式基础原子，登记成功才失效基础缓存。"""
        scope = self._authorize(access, WorkspaceOperation.MEMORY_INTENT_SUBMIT, target_workspace)
        result = (await self._reader.resolve_references([base_alias], scope=scope))[0]
        if result.kind == "pending":
            raise PendingUpdateNotAllowedError(details={"alias": base_alias})
        if result.kind != "atom" or result.atom is None:
            raise ResourceNotFoundError(details={"alias": base_alias})
        atom = result.atom
        pending = self._intents.register_update(
            UpdateFocus(
                base_alias=atom.index.alias or base_alias,
                base_uuid=str(atom.id),
                instruction=instruction,
                content=content or None,
            ),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
        )
        self._reader.evict(scope.workspace_identity, atom.id)
        return pending

    async def cancel_intents(
        self,
        aliases: list[str],
        *,
        process_id: str,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> list[str]:
        """撤回本进程提交且尚未认领的意图（``memory_intent.submit``）。

        属于提交的取消语义：只改变 ``process_id`` 对应且仍为 PENDING 的记录，
        返回实际撤回的 alias。
        """
        self._authorize(access, WorkspaceOperation.MEMORY_INTENT_SUBMIT, target_workspace)
        return self._intents.cancel_aliases(aliases, process_id=process_id)

    async def resolve_references(
        self,
        aliases: list[str],
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> list[ReferenceResolution]:
        """读取引用（``resource.read``）：每个请求项均得到一个中立解析结果。"""
        scope = self._authorize(access, WorkspaceOperation.RESOURCE_READ, target_workspace)
        return await self._reader.resolve_references(aliases, scope=scope)

    async def create_memory(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        title: str,
        summary: str,
        content: str,
        memory_type: str,
        tags: list[str],
        alias: str | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理创建入口（``management.memory``）：在目标 Workspace 中创建 Memory。

        ``provenance.source_agent_id`` 记录来源 actor（管理入口为保留
        ``system``），只作 provenance 展示，不参与可见性授权。
        """
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        # 只包装调用方提交字段的构造：输入不合法是 422，不是程序错误。
        try:
            index = IndexLayer(
                title=title,
                summary=summary,
                tags=tags,
                memory_type=MemoryType(memory_type),
                alias=alias,
            )
            payload = PayloadLayer(content=content)
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
            GlobalRoutes.PATCHOULI_MEMORY_CREATE,
            scope,
            atom,
        )

    async def list_memories(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """管理读取入口（``management.memory``）：在目标 Workspace 中列出 Memory。

        按 owner-management 语义返回该 Workspace 的全部 Memory（不含
        Agent Profile），不做 Agent ``MemoryAccessPolicy`` 过滤。
        """
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        filters = self._build_filters(memory_type=memory_type)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_LIST,
            identity_scope=scope,
            query=query,
            filters=filters if filters else None,
            limit=limit,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
            refresh_vitality=True,
        )

    async def get_memory(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理读取入口（``management.memory``）：在目标 Workspace 中读取 Memory。"""
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_GET,
            memory_id,
            identity_scope=scope,
            refresh_vitality=True,
        )
        if atom is None:
            raise MemoryNotFoundError("记忆不存在")
        return atom

    async def update_memory(
        self,
        memory_id: UUID,
        *,
        target_workspace: WorkspaceIdentity,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom:
        """管理更新入口（``management.memory``）：显式授权 mutation，且不改变原 ownership/provenance。"""
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        atom = await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_UPDATE,
            memory_id,
            identity_scope=scope,
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
        target_workspace: WorkspaceIdentity,
        positive: bool,
        source: str,
        access: WorkspaceAccessContext,
    ):
        """管理反馈入口（``management.memory``）：在目标 Workspace 中记录反馈。"""
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        try:
            return await self._global_bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RECORD_FEEDBACK,
                memory_id,
                identity_scope=scope,
                positive=positive,
                source=source,
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
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> bool:
        """管理删除入口（``management.memory``）：在目标 Workspace 中删除 Memory。"""
        scope = self._authorize(access, WorkspaceOperation.MANAGEMENT_MEMORY, target_workspace)
        return await self._global_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_DELETE,
            memory_id,
            identity_scope=scope,
        )

    # ---- Actor 可见读取（operation 授权在 backing 调用前，A2 §2.2） ----

    async def read(
        self,
        memory_id: UUID | str,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> MemoryAtom | None:
        """UUID 点读（``resource.read``）：返回完整原子独立副本。

        未知或对当前 Actor 不可见的资源按 A1 防泄露规则返回 ``None``；非法
        UUID 字符串维持 ``ValueError``（A2 §8.2）。
        """
        scope = self._authorize(access, WorkspaceOperation.RESOURCE_READ, target_workspace)
        return await self._reader.read(normalize_uuid(memory_id), scope=scope)

    async def retrieve_by_aliases(
        self,
        aliases: list[str],
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """alias 批量读取（``resource.read``）：按请求顺序返回实际可读的完整原子。"""
        scope = self._authorize(access, WorkspaceOperation.RESOURCE_READ, target_workspace)
        return await self._reader.resolve_aliases(aliases, scope=scope)

    async def retrieve(
        self,
        *,
        semantic_query: str,
        keywords: list[str] | None = None,
        top_k: int = 5,
        filters: QueryFilters | None = None,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """语义检索（``resource.search``）：保持领域排序，结果协作预热缓存。

        调用方只提交检索参数：检索请求由本层用授权返回的可信 scope 构造，
        ``IdentityScope`` 不进入调用方输入（A1 访问边界返工第 4.5 节），
        不能据请求体重新选择 Workspace。
        """
        scope = self._authorize(access, WorkspaceOperation.RESOURCE_SEARCH, target_workspace)
        request = RetrievalRequest(
            semantic_query=semantic_query,
            keywords=list(keywords or ()),
            identity_scope=scope,
            filters=filters,
            top_k=top_k,
        )
        return await self._reader.search(request, scope=scope)

    # ---- 内部辅助 ----

    def _authorize(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """在 backing/管理路由调用前执行操作授权，返回组装的可信 scope。"""
        return self._authorizer.authorize_operation(access, operation, target_workspace)

    @staticmethod
    def _build_filters(
        *,
        memory_type: str | None,
    ) -> dict[str, str]:
        filters = {}
        if memory_type:
            filters["index.memory_type"] = memory_type
        return filters
