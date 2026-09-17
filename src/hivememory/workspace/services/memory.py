"""scope-aware 正式 Memory 读取服务（Patchouli 低层 provider adapter）。

职责边界（父计划 WRX-1）：
- 只接受经 admission 签发的 ``WorkspaceAccessContext``，拒绝裸 scope；
- 绑定 Patchouli 低层 ``MidTermMemoryStore`` / ``RetrievalFamiliar`` 实例，
  不经过 Workspace public facade，也不拼接全局总线资源路由（避免递归）；
- 可见性 policy 的单一强制点在低层 store/检索引擎内（``memory_is_readable``），
  本服务只区分 not found / not visible 并做检索结果的防御性复验；
- 返回不可变 ``MemorySnapshot``，不泄漏可变 ``MemoryAtom``。
"""

from __future__ import annotations

import logging
from uuid import UUID

from hivememory.core.errors import (
    ResourceNotFoundError,
    ResourceNotVisibleError,
    ResourceUnavailableError,
)
from hivememory.core.models import IdentityScope, MemoryAtom
from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.engines.retrieval.policy import memory_is_readable
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)
from hivememory.workspace.ports import WorkspaceResourcePort
from hivememory.workspace.projections import MemorySnapshot

logger = logging.getLogger(__name__)


class WorkspaceMemoryService(WorkspaceResourcePort):
    """正式 Memory 读取的资源服务实现。

    point read 走 ``MidTermMemoryStore``（可见性 policy 在 store 的
    ``memory_is_readable`` 单点强制）；语义检索走 ``RetrievalFamiliar.retrieve``
    （检索引擎内按 identity 预过滤），并对结果做防御性 policy 复验。
    """

    def __init__(
        self,
        mid_term: MidTermMemoryStore,
        retrieval,
    ) -> None:
        """
        Args:
            mid_term: Patchouli 中期记忆库（canonical point read 入口）。
            retrieval: Patchouli 检索使魔（语义检索事实来源）。
        """
        self._mid_term = mid_term
        self._retrieval = retrieval

    # ---- WorkspaceResourcePort ----

    async def read_memory(self, access: WorkspaceAccessContext, memory_id: str) -> MemorySnapshot:
        """按 canonical UUID 读取当前授权下的 Memory 快照。

        UUID 点读是精确键获取，可在存储层之外区分"不存在"与"存在但
        不可见"，二者映射为不同稳定错误。
        """
        access = require_access_context(access, operation=WorkspaceOperation.RESOURCE_READ)
        try:
            normalized_id = UUID(str(memory_id))
        except (TypeError, ValueError) as exc:
            raise ResourceNotFoundError(
                message=f"非法 Memory 标识: {memory_id!r}",
                details={"reason": "invalid_memory_id"},
            ) from exc

        atom = await self._read_with_visibility_diagnosis(
            access.identity_scope,
            lambda *, enforce: self._mid_term.get(
                access.identity_scope,
                normalized_id,
                enforce_actor_visibility=enforce,
            ),
        )
        return MemorySnapshot.from_atom(atom)

    async def read_memory_by_alias(
        self, access: WorkspaceAccessContext, alias: str
    ) -> MemorySnapshot:
        """按 Workspace 分区 alias 读取当前授权下的 Memory 快照。

        存储适配器对 alias 读取在查询预过滤层即合并 actor 可见性——
        "不可见"与"不存在"在此路径无法区分，统一映射为 not found
        （与既有 ``RetrievalFamiliar`` alias 语义一致）；需要可见性区分
        的调用方使用 UUID 点读。
        """
        access = require_access_context(access, operation=WorkspaceOperation.RESOURCE_READ)
        normalized_alias = (alias or "").strip()
        if not normalized_alias:
            raise ResourceNotFoundError(
                message="alias 不能为空",
                details={"reason": "empty_alias"},
            )

        atom = await self._read_through_store(
            lambda: self._mid_term.get_by_alias(
                access.identity_scope,
                normalized_alias,
                enforce_actor_visibility=True,
            ),
            not_found_details={"alias": normalized_alias},
        )
        return MemorySnapshot.from_atom(atom)

    async def search_memory(
        self,
        access: WorkspaceAccessContext,
        query: str,
        *,
        top_k: int = 8,
    ) -> tuple[MemorySnapshot, ...]:
        """语义检索当前 Workspace 内对当前 Actor 可见的 Memory 快照。"""
        access = require_access_context(access, operation=WorkspaceOperation.RESOURCE_SEARCH)
        normalized_query = (query or "").strip()
        if not normalized_query:
            # 空查询不是检索失败，返回空结果由调用方决定语义。
            return ()

        request = RetrievalRequest(
            semantic_query=normalized_query,
            identity_scope=access.identity_scope,
            top_k=max(int(top_k), 0),
        )
        try:
            response = await self._retrieval.retrieve(request)
        except (StorageOfflineError, StorageReadError) as exc:
            raise ResourceUnavailableError(
                message="Memory 检索 provider 暂不可用",
                details={"reason": "retrieval_unavailable"},
            ) from exc

        scope = access.identity_scope
        snapshots: list[MemorySnapshot] = []
        for atom in response.memories or []:
            # 防御性复验：检索引擎已按 identity 预过滤，这里保证错误的
            # adapter 返回值不会绕过 ownership/visibility 扩大可见范围。
            if not memory_is_readable(
                atom,
                workspace_identity=scope.workspace_identity,
                actor_identity=scope.actor_identity,
            ):
                logger.warning(
                    "Search returned unauthorized atom; dropped: memory_id=%s",
                    atom.id,
                )
                continue
            snapshots.append(MemorySnapshot.from_atom(atom))
        return tuple(snapshots)

    # ---- 内部辅助 ----

    async def _read_through_store(self, reader, *, not_found_details=None) -> MemoryAtom:
        """单次读取并把存储故障映射为 unavailable、miss 映射为 not found。"""
        try:
            atom = await reader()
        except (StorageOfflineError, StorageReadError) as exc:
            raise ResourceUnavailableError(
                message="Memory provider 暂不可用",
                details={"reason": "storage_unavailable"},
            ) from exc
        if atom is None:
            raise ResourceNotFoundError(details=not_found_details)
        return atom

    async def _read_with_visibility_diagnosis(
        self,
        scope: IdentityScope,
        reader,
    ) -> MemoryAtom:
        """执行 point read，并把 miss 区分为 not found / not visible。

        可见性 policy 仍在 store 内单点强制；这里的第二次 ``enforce=False``
        查询仅用于把"不存在"与"存在但不可见"区分为稳定错误语义，
        仍然拒绝返回内容，不构成授权旁路。
        """
        try:
            atom = await reader(enforce=True)
        except (StorageOfflineError, StorageReadError) as exc:
            raise ResourceUnavailableError(
                message="Memory provider 暂不可用",
                details={"reason": "storage_unavailable"},
            ) from exc

        if atom is not None:
            return atom

        try:
            invisible = await reader(enforce=False)
        except (StorageOfflineError, StorageReadError) as exc:
            raise ResourceUnavailableError(
                message="Memory provider 暂不可用",
                details={"reason": "storage_unavailable"},
            ) from exc
        if invisible is not None:
            raise ResourceNotVisibleError(
                details={
                    "workspace_id": scope.workspace_identity.workspace_id,
                    "actor_id": scope.actor_identity.agent_id,
                }
            )
        raise ResourceNotFoundError(details={"workspace_id": scope.workspace_identity.workspace_id})


__all__ = ["WorkspaceMemoryService"]
