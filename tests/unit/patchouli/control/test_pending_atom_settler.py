from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call

import pytest

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.control.pending_atom_settler import PendingAtomSettler


def _settlement(alias: str = "draft_memory") -> PendingAtomSettlement:
    return PendingAtomSettlement(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        resolution=PendingAtomResolution.CREATED,
        canonical_alias="memory_alias",
        canonical_uuid="memory-uuid",
    )


@pytest.mark.asyncio
async def test_settled_publishes_canonical_payload_once() -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)
    settlement = _settlement()

    await settler.settled(settlement)
    await settler.settled(settlement)

    bus.publish.assert_awaited_once_with(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        settlement=settlement,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "event"),
    [
        ("failed", PatchouliLocalEvents.PENDING_ATOM_FAILED),
        ("cancelled", PatchouliLocalEvents.PENDING_ATOM_CANCELLED),
    ],
)
async def test_alias_terminals_publish_canonical_payload_once(method: str, event: str) -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)

    await getattr(settler, method)("draft_memory")
    await getattr(settler, method)("draft_memory")

    bus.publish.assert_awaited_once_with(event, pending_alias="draft_memory")


@pytest.mark.asyncio
async def test_first_terminal_event_wins_across_concurrent_callers() -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)

    await asyncio.gather(
        settler.failed("draft_memory"),
        settler.cancelled("draft_memory"),
    )

    assert bus.publish.await_count == 1
    assert bus.publish.await_args in (
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, pending_alias="draft_memory"),
        call(PatchouliLocalEvents.PENDING_ATOM_CANCELLED, pending_alias="draft_memory"),
    )


@pytest.mark.asyncio
async def test_settlement_publish_failure_falls_back_to_failed() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(
        side_effect=[ConnectionError("settlement unavailable"), None]
    )
    settler = PendingAtomSettler(bus)
    settlement = _settlement()

    await settler.settled(settlement)

    assert bus.publish.await_args_list == [
        call(PatchouliLocalEvents.PENDING_ATOM_SETTLED, settlement=settlement),
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, pending_alias="draft_memory"),
    ]
    await settler.failed("draft_memory")
    assert bus.publish.await_count == 2


@pytest.mark.asyncio
async def test_publish_failures_are_isolated() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=ConnectionError("bus unavailable"))
    settler = PendingAtomSettler(bus)

    await settler.failed("failed_alias")
    await settler.cancelled("cancelled_alias")
    await settler.settled(_settlement("settled_alias"))

    assert bus.publish.await_count == 4


@pytest.mark.asyncio
async def test_bounded_retention_allows_evicted_alias_to_be_published_again() -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus, terminal_retention=1)

    await settler.failed("first")
    await settler.failed("second")
    await settler.failed("first")

    assert bus.publish.await_args_list == [
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, pending_alias="first"),
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, pending_alias="second"),
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, pending_alias="first"),
    ]
