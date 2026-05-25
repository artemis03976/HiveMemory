"""
RuntimeAliasResolver - 统一三级别名解析层。

解析优先级:
  L0: PendingAtomCache (运行时 pending atom)
  L1: KoakumaAtomCache (会话级正式 atom 缓存)
  L2: Storage (冷查询长期存储)

作者: HiveMemory Team
版本: 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional
from uuid import UUID

from hivememory.alice.runtime.cache import PendingAtomCache
from hivememory.alice.runtime.models import PendingAtom
from hivememory.core.models import MemoryAtom
from hivememory.core.mtp.exceptions import (
    BusRouteUnavailableError,
    StorageOfflineError,
    StorageReadError,
)

if TYPE_CHECKING:
    from hivememory.alice.runtime.bus import AliceBus
    from hivememory.alice.runtime.cache import KoakumaAtomCache
    from hivememory.alice.runtime.models import MTPExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    """别名解析结果。"""

    kind: Literal["pending", "atom", "not_found"]
    pending: Optional[PendingAtom] = field(default=None)
    atom: Optional[MemoryAtom] = field(default=None)


class RuntimeAliasResolver:
    """
    统一三级别名解析器。

    作为 Alice runtime 共享基础设施，协调 L0/L1/L2 三级缓存命中路径。
    """

    def __init__(
        self,
        pending_cache: PendingAtomCache,
        atom_cache: "KoakumaAtomCache",
        bus: "AliceBus",
    ) -> None:
        self._pending_cache = pending_cache
        self._atom_cache = atom_cache
        self._bus = bus

    def is_pending(self, alias: str) -> bool:
        """快速检查 alias 是否在 L0 pending cache 中。"""
        return self._pending_cache.has(alias)

    async def resolve(
        self,
        alias: str,
        context: Optional["MTPExecutionContext"] = None,
    ) -> ResolveResult:
        """
        三级解析 alias。

        Returns:
            ResolveResult: kind="pending" / "atom" / "not_found"
        """
        # L0: PendingAtomCache
        pending = self._pending_cache.get(alias)
        if pending is not None:
            logger.debug(f"L0 pending cache hit: alias='{alias}'")
            return ResolveResult(kind="pending", pending=pending)

        # L1: KoakumaAtomCache
        atom = self._atom_cache.get_atom_by_alias(alias)
        if atom is not None:
            logger.debug(f"L1 atom cache hit: alias='{alias}'")
            return ResolveResult(kind="atom", atom=atom)

        # L2: Storage cold lookup
        atom = await self._cold_lookup(alias, context)
        if atom is not None:
            return ResolveResult(kind="atom", atom=atom)

        return ResolveResult(kind="not_found")

    async def _cold_lookup(
        self,
        alias: str,
        context: Optional["MTPExecutionContext"] = None,
    ) -> Optional[MemoryAtom]:
        """L2 冷查询：通过 bus 查询存储层。"""
        from hivememory.alice.runtime.models import MTPExecutionContext
        from hivememory.system.contracts.routes import GlobalRoutes

        try:
            identity = (
                context.identity
                if context is not None
                else MTPExecutionContext().identity
            )
            retrieval_response = await self._bus.request(
                GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
                aliases=[alias],
                identity=identity,
            )
            memories = getattr(retrieval_response, "memories", []) or []
            memory = memories[0] if memories else None
            if memory is None:
                logger.debug(f"L2 cold-lookup miss: alias='{alias}'")
                return None

            uuid_str = str(memory.id)
            UUID(uuid_str)

            self._atom_cache.ingest_atom(memory)
            logger.debug(
                f"L2 cold-lookup hit: alias='{alias}' -> {uuid_str}, cached"
            )
            return memory
        except KeyError as e:
            logger.error(f"L2 cold-lookup route unavailable: alias='{alias}', error={e}")
            raise BusRouteUnavailableError(
                "Memory storage service is not available."
            ) from e
        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(
                f"L2 cold-lookup infrastructure failure: alias='{alias}', error={e}"
            )
            raise StorageReadError(
                "Memory storage encountered an error during alias lookup."
            ) from e

    @property
    def atom_cache(self) -> "KoakumaAtomCache":
        return self._atom_cache

    @property
    def pending_cache(self) -> PendingAtomCache:
        return self._pending_cache


__all__ = ["ResolveResult", "RuntimeAliasResolver"]
