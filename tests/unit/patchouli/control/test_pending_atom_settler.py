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
    ("method", "kwargs", "event", "extra"),
    [
        (
            "failed",
            {"reason": "model_provider_failed"},
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            {"reason": "model_provider_failed"},
        ),
        ("cancelled", {}, PatchouliLocalEvents.PENDING_ATOM_CANCELLED, {}),
    ],
)
async def test_alias_terminals_publish_canonical_payload_once(
    method: str, kwargs: dict, event: str, extra: dict
) -> None:
    """失败事件带上生成任务的原因码，取消事件只有 alias 与 intent_id。"""
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)

    await getattr(settler, method)("draft_memory", "intent_draft_memory", **kwargs)
    await getattr(settler, method)("draft_memory", "intent_draft_memory", **kwargs)

    bus.publish.assert_awaited_once_with(
        event, pending_alias="draft_memory", intent_id="intent_draft_memory", **extra
    )


@pytest.mark.asyncio
async def test_first_terminal_event_wins_across_concurrent_callers() -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)

    await asyncio.gather(
        settler.failed("draft_memory", "intent_draft_memory"),
        settler.cancelled("draft_memory", "intent_draft_memory"),
    )

    payload = {"pending_alias": "draft_memory", "intent_id": "intent_draft_memory"}
    assert bus.publish.await_count == 1
    assert bus.publish.await_args in (
        call(PatchouliLocalEvents.PENDING_ATOM_FAILED, **payload, reason=None),
        call(PatchouliLocalEvents.PENDING_ATOM_CANCELLED, **payload),
    )


@pytest.mark.asyncio
async def test_settlement_publish_failure_falls_back_to_failed() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=[ConnectionError("settlement unavailable"), None])
    settler = PendingAtomSettler(bus)
    settlement = _settlement()

    await settler.settled(settlement)

    assert bus.publish.await_args_list == [
        call(PatchouliLocalEvents.PENDING_ATOM_SETTLED, settlement=settlement),
        call(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_memory",
            intent_id="intent_draft_memory",
            reason=None,
        ),
    ]
    await settler.failed("draft_memory", "intent_draft_memory")
    assert bus.publish.await_count == 2


@pytest.mark.asyncio
async def test_publish_failures_are_isolated() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=ConnectionError("bus unavailable"))
    settler = PendingAtomSettler(bus)

    await settler.failed("failed_alias", "intent_failed_alias")
    await settler.cancelled("cancelled_alias", "intent_cancelled_alias")
    await settler.settled(_settlement("settled_alias"))

    assert bus.publish.await_count == 4


@pytest.mark.asyncio
async def test_bounded_retention_allows_evicted_alias_to_be_published_again() -> None:
    bus = AsyncMock()
    settler = PendingAtomSettler(bus, terminal_retention=1)

    await settler.failed("first", "intent_first")
    await settler.failed("second", "intent_second")
    await settler.failed("first", "intent_first")

    def failed(alias: str):
        return call(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias=alias,
            intent_id=f"intent_{alias}",
            reason=None,
        )

    assert bus.publish.await_args_list == [failed("first"), failed("second"), failed("first")]


@pytest.mark.asyncio
async def test_merge_redirect_publishes_after_terminal_without_reserving_another_terminal():
    """改指不被首次结算的保留拦住，也不抢占其他 alias 的首次终态。"""
    bus = AsyncMock()
    settler = PendingAtomSettler(bus)
    created = _settlement()
    merged = created.model_copy(
        update={"resolution": PendingAtomResolution.MERGED, "canonical_uuid": "retained"}
    )

    await settler.settled(created)
    await settler.merged(merged, absorbed_memory_uuid="memory-uuid")
    other = _settlement("other")
    await settler.merged(other, absorbed_memory_uuid="absorbed")
    await settler.settled(other)

    assert bus.publish.await_args_list == [
        call(PatchouliLocalEvents.PENDING_ATOM_SETTLED, settlement=created),
        call(
            PatchouliLocalEvents.PENDING_ATOM_MERGED,
            settlement=merged,
            absorbed_memory_uuid="memory-uuid",
        ),
        call(
            PatchouliLocalEvents.PENDING_ATOM_MERGED,
            settlement=other,
            absorbed_memory_uuid="absorbed",
        ),
        call(PatchouliLocalEvents.PENDING_ATOM_SETTLED, settlement=other),
    ]


@pytest.mark.asyncio
async def test_merge_publish_failure_only_logs_warning(caplog):
    """正式写入已成功，改指事件发布失败不能再发 FAILED。"""
    bus = AsyncMock()
    bus.publish.side_effect = ConnectionError("bus unavailable")
    settler = PendingAtomSettler(bus)
    settlement = _settlement().model_copy(update={"resolution": PendingAtomResolution.MERGED})

    await settler.merged(settlement, absorbed_memory_uuid="absorbed")

    assert bus.publish.await_args_list == [
        call(
            PatchouliLocalEvents.PENDING_ATOM_MERGED,
            settlement=settlement,
            absorbed_memory_uuid="absorbed",
        )
    ]
    assert "MERGED event publish failed" in caplog.text
