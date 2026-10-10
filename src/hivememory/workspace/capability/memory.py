"""actor API：逐次检查操作权限，读取视图交付与写入意图登记属于 workspace。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.core.access import WorkspaceOperation
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import (
    PendingUpdateNotAllowedError,
    ResourceNotFoundError,
)
from hivememory.core.models import (
    MemoryAtom,
    PendingAtom,
    ReferenceResolution,
    UpdateFocus,
    WriteFocus,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.utils.uuid import normalize_uuid

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import IdentityScope, WorkspaceIdentity
    from hivememory.core.models.query import QueryFilters
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.alias import AliasResolver

logger = logging.getLogger(__name__)


class MemoryApplicationService:
    """actor 资源入口，访问 context 只由本层授权并转为本次操作 scope。"""

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
        """读取引用（``resource.read``），为本次交付的正式原子逐一记录引用。

        逐项结果保留顺序与重复项，引用记录在本请求内按 memory id 去重。
        只处理 atom 或带可读原子的 redirect；缓存命中也记录。记录失败只
        影响日志，不改变读取结果，取消则继续传播给调用方。
        """
        scope = self._authorize(access, WorkspaceOperation.RESOURCE_READ, target_workspace)
        results = await self._reader.resolve_references(aliases, scope=scope)
        cited: set[UUID] = set()
        for result in results:
            atom = result.atom
            if result.kind not in ("atom", "redirect") or atom is None or atom.id in cited:
                continue
            # 去重发生在尝试前：同批失败不重试，以免已提交但应答失败时重复计数。
            cited.add(atom.id)
            try:
                await self._global_bus.request(
                    GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
                    memory_id=atom.id,
                    identity_scope=scope,
                    source="workspace.reference_read",
                )
            except Exception:
                # 生命周期反馈为 best-effort；CancelledError 不属于 Exception，
                # 必须正常传播，不能把已经取消的读取变为成功。
                logger.warning(
                    "Failed to record memory citation for memory_id=%s source=%s",
                    atom.id,
                    "workspace.reference_read",
                    exc_info=True,
                )
        return results

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

    def _authorize(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """在 backing 调用前执行操作授权，返回组装的可信 scope。"""
        return self._authorizer.authorize_operation(access, operation, target_workspace)
