"""Agent Profile 资源服务：Root / CALL 共用的 profile 投影入口。

Root Profile 与 CALL callee Profile 都通过本服务解析（父计划 5.2 节），
区别只在传入的 access context 与调用方 provenance；本服务不区分调用方，
也不把 caller 的读取结果当作 callee 的授权。

内建 default/omni_doll 短路保留在同一入口，以 ``source_kind="builtin"``
显式返回；自定义 alias 的缺失、越权与类型错误映射为稳定资源错误（MTP
adapter 在 WRX-4 按既有契约把可见性/缺失合并翻译为 AliasNotFoundError）。
"""

from __future__ import annotations

import logging

from hivememory.core.errors import ResourceNotFoundError, ResourceUnavailableError
from hivememory.core.models import OMNI_DOLL_PROFILE, AgentProfile, MemoryType
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
    MemoryTypeMismatchError,
    StorageOfflineError,
    StorageReadError,
)
from hivememory.patchouli.memory_library.stores import MidTermMemoryStore
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)
from hivememory.workspace.projections import ProfileSnapshot

logger = logging.getLogger(__name__)

_BUILTIN_PROFILE_ALIASES = frozenset({"default", "omni_doll"})


class ProfileResourceService:
    """Agent Profile 的统一读取服务（低层 ``MidTermMemoryStore`` adapter）。

    与 ``RetrievalFamiliar.get_agent_profile`` 保持相同的领域判定顺序
    （内建短路 → alias 读取 → 类型校验 → 投影解析），但返回携带 source
    身份的不可变快照，供 Alice 侧 resolver（WRX-4）与无 Alice 消费者共用。
    只实现 ``WorkspaceResourcePort`` 的 profile 读取面；协议本身由资源
    门面（WRX-2）组合完整。
    """

    def __init__(self, mid_term: MidTermMemoryStore) -> None:
        self._mid_term = mid_term

    async def read_profile(
        self,
        access: WorkspaceAccessContext,
        agent_alias: str | None,
    ) -> ProfileSnapshot:
        """按当前授权读取 Agent Profile 快照；内建 alias 显式标识返回。"""
        access = require_access_context(access, operation=WorkspaceOperation.PROFILE_READ)
        normalized_alias = (agent_alias or "").strip()
        if not normalized_alias or normalized_alias in _BUILTIN_PROFILE_ALIASES:
            return ProfileSnapshot(
                agent_alias=None,
                profile=OMNI_DOLL_PROFILE.model_copy(deep=True),
                source_kind="builtin",
            )

        scope = access.identity_scope
        try:
            atom = await self._mid_term.get_by_alias(scope, normalized_alias)
        except (StorageOfflineError, StorageReadError) as exc:
            raise ResourceUnavailableError(
                message="Profile provider 暂不可用",
                details={"reason": "storage_unavailable"},
            ) from exc

        if atom is None:
            # alias 读取的存储预过滤层已合并 actor 可见性：不可见与缺失
            # 统一按 not found 拒绝，与既有 ``RetrievalFamiliar`` alias
            # 语义保持一致（MTP adapter 负责翻译为 AliasNotFoundError）。
            raise ResourceNotFoundError(details={"agent_alias": normalized_alias})

        if atom.index.memory_type != MemoryType.AGENT_PROFILE:
            raise MemoryTypeMismatchError(
                message_key="mtp.call.profile_type_mismatch",
                params={"agent_alias": normalized_alias},
            )

        profile = AgentProfile.from_atom(atom)
        if profile is None:
            raise InvalidArgumentError(
                message_key="mtp.call.profile_invalid",
                params={"agent_alias": normalized_alias},
            )

        logger.info("Agent profile %r resolved via ProfileResourceService", normalized_alias)
        return ProfileSnapshot(
            agent_alias=normalized_alias,
            profile=profile.model_copy(deep=True),
            source_kind="atom",
            source_atom_uuid=str(atom.id),
            source_revision=atom.meta.version,
        )


__all__ = ["ProfileResourceService"]
