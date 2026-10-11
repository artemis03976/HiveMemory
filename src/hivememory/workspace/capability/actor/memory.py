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
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.contracts.operations import (
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkspaceAccessContext
    from hivememory.core.models import IdentityScope
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.resolution.alias import AliasResolver

logger = logging.getLogger(__name__)


class ActorMemoryService:
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
        request: SubmitWriteIntentRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> PendingAtom:
        """提交 WRITE（``memory.write``），ACK 仅表示意图已登记。"""
        scope = self._authorize(access, WorkspaceOperation.MEMORY_WRITE)
        process_id = self._authorizer.get_thread_process_id(access)
        return self._intents.register_write(
            request.focus,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
        )

    async def submit_update_intent(
        self,
        request: SubmitUpdateIntentRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> PendingAtom:
        """提交 UPDATE（``memory.update``）：授权后解析可读正式基础原子，登记成功才失效基础缓存。"""
        scope = self._authorize(access, WorkspaceOperation.MEMORY_UPDATE)
        # 在冷读之前取得关联；跨越撤销的迟到登记由操作入口同步补偿。
        process_id = self._authorizer.get_thread_process_id(access)
        result = (await self._reader.resolve_references([request.base_alias], scope=scope))[0]
        if result.kind == "pending":
            raise PendingUpdateNotAllowedError(details={"alias": request.base_alias})
        if result.kind != "atom" or result.atom is None:
            raise ResourceNotFoundError(details={"alias": request.base_alias})
        atom = result.atom
        pending = self._intents.register_update(
            UpdateFocus(
                base_alias=atom.index.alias or request.base_alias,
                base_uuid=str(atom.id),
                instruction=request.instruction,
                content=request.content or None,
            ),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
        )
        self._reader.evict(scope.workspace_identity, atom.id)
        return pending

    async def resolve_references(
        self,
        request: ResolveReferencesRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> list[ReferenceResolution]:
        """读取引用（``memory.read``），为本次交付的正式原子逐一记录引用。

        逐项结果保留顺序与重复项，引用记录在本请求内按 memory id 去重。
        只处理 atom 或带可读原子的 redirect；缓存命中也记录。记录失败只
        影响日志，不改变读取结果，取消则继续传播给调用方。
        """
        scope = self._authorize(access, WorkspaceOperation.MEMORY_READ)
        return await self._resolve_and_record_citations(list(request.aliases), scope=scope)

    async def resolve_run_target(
        self,
        request: ResolveRunTargetRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> ReferenceResolution:
        """解析 RUN 的单个执行目标（``memory.run``），交付时记录正式引用。

        RUN 与 READ 独立授权；资源可见性与引用副作用沿用同一读取视图，
        返回中立解析结果，目标是否可执行仍由 Actor 的工具运行时判断。
        """
        scope = self._authorize(access, WorkspaceOperation.MEMORY_RUN)
        return (await self._resolve_and_record_citations([request.alias], scope=scope))[0]

    async def _resolve_and_record_citations(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
    ) -> list[ReferenceResolution]:
        """复用 READ/RUN 的解析与引用记录，权限只由各自的公共入口检查。"""
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

    async def retrieve(
        self,
        request: RetrieveRequest,
        *,
        access: WorkspaceAccessContext,
    ) -> list[MemoryAtom]:
        """语义检索（``memory.search``）：保持领域排序，结果协作预热缓存。

        调用方只提交检索参数：检索请求由本层用授权返回的可信 scope 构造，
        ``IdentityScope`` 不进入调用方输入（A1 访问边界返工第 4.5 节），
        不能据请求体重新选择 Workspace。
        """
        scope = self._authorize(access, WorkspaceOperation.MEMORY_SEARCH)
        retrieval = RetrievalRequest(
            semantic_query=request.semantic_query,
            keywords=list(request.keywords),
            identity_scope=scope,
            filters=request.filters,
            top_k=request.top_k,
        )
        return await self._reader.search(retrieval, scope=scope)

    def _authorize(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
    ) -> IdentityScope:
        """在 backing 调用前执行操作授权，返回组装的可信 scope。"""
        return self._authorizer.authorize_thread_operation(access, operation)
