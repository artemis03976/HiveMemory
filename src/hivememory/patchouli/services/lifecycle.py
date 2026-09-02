"""Patchouli 生命周期使魔。

承接记忆生命力刷新、强化事件、定期整理与复活入口。
"""

from __future__ import annotations

import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple
from uuid import UUID

from hivememory.core.models import (
    MemoryAtom,
    IdentityScope,
    require_identity_scope,
)
from hivememory.utils.uuid import normalize_uuid

if TYPE_CHECKING:
    from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
    from hivememory.patchouli.memory_library.library import MemoryLibrary

logger = logging.getLogger(__name__)


class LifecycleFamiliar:
    """生命周期业务门面。"""

    def __init__(
        self,
        *,
        lifecycle_engine: "MemoryLifecycleEngine",
        memory_library: "MemoryLibrary",
    ) -> None:
        self.lifecycle_engine = lifecycle_engine
        self._memory_library = memory_library
        logger.info("LifecycleFamiliar 初始化完成")

    async def run_gardening_once(self) -> dict[str, Any]:
        """执行一次生命周期垃圾回收，供全局调度器调用。"""
        start = monotonic()
        result: dict[str, Any] = {
            "success": False,
            "archived_count": 0,
            "duration_ms": 0.0,
            "error": None,
        }
        try:
            archived = await self.lifecycle_engine.run_garbage_collection(force=False)
            result["success"] = True
            result["archived_count"] = int(archived or 0)
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Lifecycle gardening failed: %s", exc, exc_info=True)
        finally:
            result["duration_ms"] = (monotonic() - start) * 1000
        return result

    async def refresh_memory_vitality(
        self,
        memories: Iterable[MemoryAtom],
        persist: bool = False,
    ) -> List[Tuple[UUID, float]]:
        """批量刷新记忆生命力。"""
        return await self.lifecycle_engine.refresh_vitality_batch(memories, persist=persist)

    async def record_hit(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        source: str = "system",
    ) -> Any:
        """记录一次命中事件。"""
        return await self.lifecycle_engine.record_hit(
            require_identity_scope(identity_scope),
            normalize_uuid(memory_id),
            source=source,
        )

    async def record_citation(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        source: str = "system",
    ) -> Any:
        """记录一次引用事件。"""
        return await self.lifecycle_engine.record_citation(
            require_identity_scope(identity_scope),
            normalize_uuid(memory_id),
            source=source,
        )

    async def record_feedback(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
        positive: bool,
        source: str = "user",
    ) -> Any:
        """记录用户反馈事件。"""
        return await self.lifecycle_engine.record_feedback(
            require_identity_scope(identity_scope),
            normalize_uuid(memory_id),
            positive=positive,
            source=source,
        )

    async def revive_memory(
        self,
        memory_id: UUID | str,
        *,
        identity_scope: IdentityScope,
    ) -> None:
        """从长期存储复活记忆到中期存储。"""
        await self._memory_library.revive(
            require_identity_scope(identity_scope),
            normalize_uuid(memory_id),
        )


__all__ = ["LifecycleFamiliar"]
