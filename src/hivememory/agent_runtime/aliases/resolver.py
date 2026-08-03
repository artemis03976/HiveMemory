"""
RuntimeAliasResolver - 统一三级别名解析层。

解析优先级:
  L0: PendingAtomRuntime (运行时 pending atom)
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

from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.core.models import MemoryAtom
from hivememory.core.models.pending import (
    PendingAtom,
    PendingAtomResolution,
    PendingAtomSettlement,
    PendingAtomStatus,
)
from hivememory.core.mtp.exceptions import (
    BusRouteUnavailableError,
    StorageOfflineError,
    StorageReadError,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.aliases.cache import KoakumaAtomCache
    from hivememory.agent_runtime.models import MTPExecutionContext
    from hivememory.system.runtime.bus.async_bus import AsyncSystemBus

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    """别名解析结果。"""

    kind: Literal["pending", "redirect", "discarded", "failed", "expired", "atom", "not_found"]
    requested_alias: Optional[str] = field(default=None)
    canonical_alias: Optional[str] = field(default=None)
    canonical_uuid: Optional[str] = field(default=None)
    pending: Optional[PendingAtom] = field(default=None)
    atom: Optional[MemoryAtom] = field(default=None)
    settlement: Optional[PendingAtomSettlement] = field(default=None)


class RuntimeAliasResolver:
    """
    统一三级别名解析器。

    作为 Alice runtime 共享基础设施，协调 L0/L1/L2 三级缓存命中路径。
    """

    def __init__(
        self,
        pending_runtime: PendingAtomRuntime,
        atom_cache: "KoakumaAtomCache",
        bus: "AsyncSystemBus",
    ) -> None:
        self._pending_runtime = pending_runtime
        self._atom_cache = atom_cache
        self._bus = bus

    def is_pending(self, alias: str) -> bool:
        """快速检查 alias 是否在 L0 pending cache 中。"""
        return self._pending_runtime.has(alias)

    async def resolve(
        self,
        alias: str,
        context: Optional["MTPExecutionContext"] = None,
    ) -> ResolveResult:
        """
        三级解析 alias。

        Returns:
            ResolveResult: kind="pending" / "redirect" / "discarded" / "failed" / "atom" / "not_found"
        """
        # L0: PendingAtomRuntime
        pending = self._pending_runtime.get(alias)
        if pending is not None:
            logger.debug(f"L0 pending cache hit: alias='{alias}'")
            return await self._resolve_pending_hit(pending, alias, context)

        # L1: KoakumaAtomCache
        atom = self._atom_cache.get_atom_by_alias(alias)
        if atom is not None:
            logger.debug(f"L1 atom cache hit: alias='{alias}'")
            return ResolveResult(kind="atom", requested_alias=alias, atom=atom)

        # L2: Storage cold lookup
        atom = await self._cold_lookup(alias, context)
        if atom is not None:
            return ResolveResult(kind="atom", requested_alias=alias, atom=atom)

        return ResolveResult(kind="not_found", requested_alias=alias)

    async def _resolve_pending_hit(
        self,
        pending: PendingAtom,
        alias: str,
        context: Optional["MTPExecutionContext"] = None,
    ) -> ResolveResult:
        """解析 L0 pending 命中，包括已结算后的 redirect 状态。"""
        settlement = pending.settlement

        if pending.status.is_in_flight:
            return ResolveResult(
                kind="pending",
                requested_alias=alias,
                pending=pending,
                settlement=settlement,
            )

        if pending.status == PendingAtomStatus.FAILED:
            return ResolveResult(
                kind="failed",
                requested_alias=alias,
                pending=pending,
                settlement=settlement,
            )

        if pending.status == PendingAtomStatus.SETTLED:
            snapshot = self._pending_runtime.snapshot(alias)
            resolution = snapshot.resolution if snapshot else None

            if resolution == PendingAtomResolution.DISCARDED:
                return ResolveResult(
                    kind="discarded",
                    requested_alias=alias,
                    pending=pending,
                    settlement=settlement,
                )

            if settlement is not None and (
                settlement.canonical_alias or settlement.canonical_uuid
            ):
                atom = self._resolve_cached_canonical(settlement)
                if atom is None and settlement.canonical_alias:
                    atom = await self._cold_lookup(settlement.canonical_alias, context)

                return ResolveResult(
                    kind="redirect",
                    requested_alias=alias,
                    canonical_alias=settlement.canonical_alias,
                    canonical_uuid=settlement.canonical_uuid,
                    pending=pending,
                    atom=atom,
                    settlement=settlement,
                )

        if pending.status == PendingAtomStatus.EXPIRED:
            return ResolveResult(
                kind="expired",
                requested_alias=alias,
                pending=pending,
            )

        # CANCELLED → not_found
        return ResolveResult(
            kind="not_found",
            requested_alias=alias,
            pending=pending,
            settlement=settlement,
        )

    def _resolve_cached_canonical(
        self,
        settlement: PendingAtomSettlement,
    ) -> Optional[MemoryAtom]:
        if settlement.canonical_uuid:
            atom = self._atom_cache.get_atom_by_uuid(settlement.canonical_uuid)
            if atom is not None:
                return atom
        if settlement.canonical_alias:
            return self._atom_cache.get_atom_by_alias(settlement.canonical_alias)
        return None

    async def _cold_lookup(
        self,
        alias: str,
        context: Optional["MTPExecutionContext"] = None,
    ) -> Optional[MemoryAtom]:
        """L2 冷查询：通过 bus 查询存储层。"""
        from hivememory.agent_runtime.models import MTPExecutionContext
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
            raise BusRouteUnavailableError(cause=e) from e
        except (StorageOfflineError, StorageReadError):
            raise
        except Exception as e:
            logger.error(
                f"L2 cold-lookup infrastructure failure: alias='{alias}', error={e}"
            )
            raise StorageReadError(cause=e) from e

    @property
    def atom_cache(self) -> "KoakumaAtomCache":
        return self._atom_cache

    @property
    def pending_runtime(self) -> PendingAtomRuntime:
        return self._pending_runtime


__all__ = ["ResolveResult", "RuntimeAliasResolver"]
