"""InMemoryWorkStore 的 Q1 contract tests。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from hivememory.infrastructure.work_queue import InMemoryWorkStore
from hivememory.system.runtime.work_queue import (
    DuplicateWorkItemError,
    QueuePolicy,
    WorkErrorSnapshot,
    WorkItem,
    WorkQueueCapacityError,
    WorkState,
    WorkStateConflictError,
    encode_canonical_json,
)
from tests.unit.infrastructure.work_queue.work_store_contract import WorkStoreContract


class TestInMemoryWorkStoreContract(WorkStoreContract):
    store_factory = staticmethod(InMemoryWorkStore)


def _item(work_id: str, *, key: str | None = None, lane: str = "lane") -> WorkItem:
    return WorkItem(
        work_id=work_id,
        lane=lane,
        kind="test.work.v1",
        schema_version=1,
        payload=encode_canonical_json(work_id),
        ordering_key=key,
    )


@pytest.mark.asyncio
async def test_enqueue_claim_and_terminal_transition() -> None:
    store = InMemoryWorkStore()
    store.configure_lane("lane", QueuePolicy(capacity=2, max_concurrency=1))

    queued = await store.enqueue(_item("work-1"))
    claimed = await store.claim_ready("lane", limit=1)
    terminal = await store.mark_succeeded("work-1", result_ref="artifact-1")

    assert queued.state == WorkState.QUEUED
    assert claimed[0].state == WorkState.RUNNING
    assert claimed[0].attempt_count == 1
    assert terminal is not None
    assert terminal.state == WorkState.SUCCEEDED
    assert terminal.result_ref == "artifact-1"


@pytest.mark.asyncio
async def test_capacity_rejects_new_work_without_dropping_existing_item() -> None:
    store = InMemoryWorkStore()
    store.configure_lane("lane", QueuePolicy(capacity=1, max_concurrency=1))
    await store.enqueue(_item("work-1"))

    with pytest.raises(WorkQueueCapacityError):
        await store.enqueue(_item("work-2"))

    assert (await store.get("work-1")).state == WorkState.QUEUED  # type: ignore[union-attr]
    assert await store.get("work-2") is None


@pytest.mark.asyncio
async def test_duplicate_work_id_is_rejected() -> None:
    store = InMemoryWorkStore()
    store.configure_lane("lane", QueuePolicy(capacity=2, max_concurrency=1))
    await store.enqueue(_item("work-1"))

    with pytest.raises(DuplicateWorkItemError):
        await store.enqueue(_item("work-1"))


@pytest.mark.asyncio
async def test_same_key_retry_keeps_original_fifo_position() -> None:
    store = InMemoryWorkStore()
    store.configure_lane(
        "lane",
        QueuePolicy(capacity=4, max_concurrency=2, ordered_by_key=True),
    )
    await store.enqueue(_item("first", key="topic-1"))
    await store.enqueue(_item("second", key="topic-1"))

    first_claim = await store.claim_ready("lane", limit=2)
    assert [record.work_id for record in first_claim] == ["first"]

    retrying = await store.schedule_retry(
        "first",
        available_at=datetime.now(UTC),
        error=WorkErrorSnapshot(error_class="TransientError"),
    )
    assert retrying.state == WorkState.RETRY_WAIT
    retry_claim = await store.claim_ready("lane", limit=2)
    assert [record.work_id for record in retry_claim] == ["first"]
    assert retry_claim[0].attempt_count == 2

    await store.mark_succeeded("first")
    second_claim = await store.claim_ready("lane", limit=2)
    assert [record.work_id for record in second_claim] == ["second"]


@pytest.mark.asyncio
async def test_different_ordering_keys_can_be_claimed_together() -> None:
    store = InMemoryWorkStore()
    store.configure_lane(
        "lane",
        QueuePolicy(capacity=4, max_concurrency=2, ordered_by_key=True),
    )
    await store.enqueue(_item("first", key="topic-1"))
    await store.enqueue(_item("second", key="topic-1"))
    await store.enqueue(_item("other", key="topic-2"))

    claimed = await store.claim_ready("lane", limit=3)

    assert [record.work_id for record in claimed] == ["first", "other"]


@pytest.mark.asyncio
async def test_concurrent_claim_does_not_return_duplicate_work() -> None:
    store = InMemoryWorkStore()
    store.configure_lane("lane", QueuePolicy(capacity=2, max_concurrency=2))
    await store.enqueue(_item("work-1"))

    results = await asyncio.gather(
        store.claim_ready("lane", limit=1),
        store.claim_ready("lane", limit=1),
    )

    assert sum(len(result) for result in results) == 1


@pytest.mark.asyncio
async def test_terminal_retention_is_bounded() -> None:
    store = InMemoryWorkStore()
    store.configure_lane(
        "lane",
        QueuePolicy(capacity=2, max_concurrency=1, terminal_retention=1),
    )
    for work_id in ("work-1", "work-2"):
        await store.enqueue(_item(work_id))
        await store.claim_ready("lane", limit=1)
        await store.mark_succeeded(work_id)

    assert await store.get("work-1") is None
    assert (await store.get("work-2")).state == WorkState.SUCCEEDED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_illegal_terminal_transition_is_rejected() -> None:
    store = InMemoryWorkStore()
    store.configure_lane("lane", QueuePolicy(capacity=1, max_concurrency=1))
    await store.enqueue(_item("work-1"))

    with pytest.raises(WorkStateConflictError):
        await store.mark_succeeded("work-1")
