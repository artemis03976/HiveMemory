"""Patchouli 的 PendingAtom 功能事件结算。"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Protocol

from hivememory.core.models import PendingAtomSettlement
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents

logger = logging.getLogger(__name__)


class _PendingAtomEventBus(Protocol):
    async def publish(self, event: str, *args: Any, **kwargs: Any) -> None: ...


class PendingAtomSettler:
    """按 pending alias 集中发布一次 best-effort 终态功能事件。"""

    def __init__(
        self,
        bus: _PendingAtomEventBus,
        *,
        terminal_retention: int = 1024,
    ) -> None:
        if terminal_retention < 1:
            raise ValueError("terminal_retention must be at least 1")
        self._bus = bus
        self._terminal_retention = terminal_retention
        self._terminals: OrderedDict[str, str] = OrderedDict()
        self._lock = asyncio.Lock()

    async def settled(self, settlement: PendingAtomSettlement) -> None:
        """发布结算事件；发布失败时降级为失败事件。"""

        pending_alias = settlement.pending_alias
        if not await self._reserve(pending_alias, "settled"):
            return
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_SETTLED,
                settlement=settlement,
            )
        except asyncio.CancelledError:
            logger.warning(
                "PendingAtom SETTLED event publish cancelled: pending_alias=%s",
                pending_alias,
                exc_info=True,
            )
        except Exception:
            logger.warning(
                "PendingAtom SETTLED event publish failed: pending_alias=%s",
                pending_alias,
                exc_info=True,
            )
            await self._fallback_failed(pending_alias)

    async def failed(self, pending_alias: str) -> None:
        """为指定 pending alias 发布一次 best-effort 失败事件。"""

        if not await self._reserve(pending_alias, "failed"):
            return
        await self._publish_alias_event(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            terminal="FAILED",
            pending_alias=pending_alias,
        )

    async def cancelled(self, pending_alias: str) -> None:
        """为指定 pending alias 发布一次 best-effort 取消事件。"""

        if not await self._reserve(pending_alias, "cancelled"):
            return
        await self._publish_alias_event(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            terminal="CANCELLED",
            pending_alias=pending_alias,
        )

    async def _reserve(self, pending_alias: str, terminal: str) -> bool:
        if not pending_alias.strip():
            logger.warning("PendingAtom %s event skipped for blank alias", terminal.upper())
            return False

        async with self._lock:
            existing = self._terminals.get(pending_alias)
            if existing is not None:
                if existing != terminal:
                    logger.warning(
                        "PendingAtom terminal event ignored after %s: "
                        "pending_alias=%s, requested=%s",
                        existing,
                        pending_alias,
                        terminal,
                    )
                return False
            self._remember(pending_alias, terminal)
            return True

    async def _fallback_failed(self, pending_alias: str) -> None:
        async with self._lock:
            if self._terminals.get(pending_alias) != "settled":
                return
            self._terminals[pending_alias] = "failed"
            self._terminals.move_to_end(pending_alias)
        await self._publish_alias_event(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            terminal="FAILED",
            pending_alias=pending_alias,
        )

    async def _publish_alias_event(
        self,
        event: str,
        *,
        terminal: str,
        pending_alias: str,
    ) -> None:
        try:
            await self._bus.publish(event, pending_alias=pending_alias)
        except asyncio.CancelledError:
            logger.warning(
                "PendingAtom %s event publish cancelled: pending_alias=%s",
                terminal,
                pending_alias,
                exc_info=True,
            )
        except Exception:
            logger.warning(
                "PendingAtom %s event publish failed: pending_alias=%s",
                terminal,
                pending_alias,
                exc_info=True,
            )

    def _remember(self, pending_alias: str, terminal: str) -> None:
        self._terminals[pending_alias] = terminal
        while len(self._terminals) > self._terminal_retention:
            self._terminals.popitem(last=False)


__all__ = ["PendingAtomSettler"]
