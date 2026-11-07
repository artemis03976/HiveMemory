from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.bus import PatchouliBus

logger = logging.getLogger(__name__)


class PatchouliService:
    """Patchouli 对外能力门面中服务对话轮次的记忆反馈：引用、检索命中与存储健康。

    访问边界：本门面是授权点以下的资源 owner，公开路由只接收授权点组装的
    ``IdentityScope``，不接收工作 actor，也不做 operation 检查；资源归属仍由
    本层与领域实现独立校验。
    """

    def __init__(self, bus: PatchouliBus) -> None:
        self._local_bus = bus

    async def record_memory_citation(
        self,
        memory_id: str | UUID,
        *,
        identity_scope: IdentityScope,
        source: str = "mtp",
    ) -> Any:
        """记录一次记忆引用事件。"""

        normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
        return await self._local_bus.request(
            PatchouliLocalRoutes.MEMORY_RECORD_CITATION,
            normalized_id,
            belong_to=require_identity_scope(identity_scope).workspace_identity,
            source=source,
        )

    async def record_retrieval_hits(
        self,
        memory_ids: Sequence[str | UUID],
        *,
        identity_scope: IdentityScope,
        source: str,
    ) -> None:
        """为一批检索到的记忆逐条记录命中，同一批按记忆去重。

        命中是生命周期的强化信号，尽力而为：单条记录失败只记日志，不影响其余
        记忆，也不交回调用方；取消照常传播。
        """
        belong_to = require_identity_scope(identity_scope).workspace_identity
        seen: set[UUID] = set()
        for memory_id in memory_ids:
            normalized_id = memory_id if isinstance(memory_id, UUID) else UUID(str(memory_id))
            if normalized_id in seen:
                continue
            seen.add(normalized_id)
            try:
                await self._local_bus.request(
                    PatchouliLocalRoutes.MEMORY_RECORD_HIT,
                    normalized_id,
                    belong_to=belong_to,
                    source=source,
                )
            except Exception:
                logger.warning(
                    "记录检索命中失败: memory_id=%s, source=%s",
                    normalized_id,
                    source,
                    exc_info=True,
                )

    async def storage_available(self) -> bool:
        """存储是否可用；不可用时提示词注入离线提示，阻止 Agent 发出记忆操作。"""
        return await self._local_bus.request(PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH)


__all__ = ["PatchouliService"]
