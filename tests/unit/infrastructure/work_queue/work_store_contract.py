"""Reusable contract tests for WorkStorePort adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from hivememory.system.runtime.work_queue import (
    QueuePolicy,
    WorkErrorSnapshot,
    WorkItem,
    WorkRecord,
    WorkState,
    WorkStateConflictError,
    WorkStorePort,
    encode_canonical_json,
)

type WorkStoreFactory = Callable[[], WorkStorePort]


def _item(work_id: str) -> WorkItem:
    return WorkItem(
        work_id=work_id,
        lane="lane",
        kind="test.work.v1",
        schema_version=1,
        payload=encode_canonical_json(work_id),
    )


class WorkStoreContract:
    """Contract shared by every WorkStorePort adapter implementation."""

    store_factory: WorkStoreFactory

    def _store(self) -> WorkStorePort:
        store = self.store_factory()
        store.configure_lane("lane", QueuePolicy(capacity=8, max_concurrency=4))
        return store

    async def _running(self, store: WorkStorePort, work_id: str) -> WorkRecord:
        await store.enqueue(_item(work_id))
        claimed = await store.claim_ready("lane", limit=1)
        assert len(claimed) == 1
        return claimed[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("transition", "expected_state"),
        [
            ("succeeded", WorkState.SUCCEEDED),
            ("retry", WorkState.RETRY_WAIT),
            ("failed", WorkState.FAILED),
            ("dead_lettered", WorkState.DEAD_LETTER),
        ],
    )
    async def test_successful_transition_returns_committed_record(
        self,
        transition: str,
        expected_state: WorkState,
    ) -> None:
        store = self._store()
        running = await self._running(store, f"work-{transition}")
        error = WorkErrorSnapshot(error_class="ContractError", message="safe-reason")
        retry_at = datetime.now(UTC) + timedelta(seconds=5)

        if transition == "succeeded":
            transitioned = await store.mark_succeeded(running.work_id, result_ref="artifact-1")
        elif transition == "retry":
            transitioned = await store.schedule_retry(
                running.work_id,
                available_at=retry_at,
                error=error,
            )
        elif transition == "failed":
            transitioned = await store.mark_failed(running.work_id, error)
        else:
            transitioned = await store.mark_dead_lettered(running.work_id, error)

        assert transitioned == await store.get(running.work_id)
        assert transitioned.state == expected_state
        assert transitioned.attempt_count == running.attempt_count
        if expected_state == WorkState.SUCCEEDED:
            assert transitioned.result_ref == "artifact-1"
            assert transitioned.finished_at is not None
            assert transitioned.last_error is None
        elif expected_state == WorkState.RETRY_WAIT:
            assert transitioned.available_at == retry_at
            assert transitioned.finished_at is None
            assert transitioned.last_error == error
        else:
            assert transitioned.finished_at is not None
            assert transitioned.last_error == error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transition",
        ["succeeded", "retry", "failed", "dead_lettered"],
    )
    async def test_illegal_transition_is_rejected(self, transition: str) -> None:
        store = self._store()
        queued = await store.enqueue(_item(f"queued-{transition}"))
        error = WorkErrorSnapshot(error_class="ContractError")

        with pytest.raises(WorkStateConflictError):
            if transition == "succeeded":
                await store.mark_succeeded(queued.work_id)
            elif transition == "retry":
                await store.schedule_retry(
                    queued.work_id,
                    available_at=datetime.now(UTC),
                    error=error,
                )
            elif transition == "failed":
                await store.mark_failed(queued.work_id, error)
            else:
                await store.mark_dead_lettered(queued.work_id, error)

        assert await store.get(queued.work_id) == queued

    @pytest.mark.asyncio
    async def test_concurrent_cancel_returns_one_authoritative_terminal_record(self) -> None:
        store = self._store()
        queued = await store.enqueue(_item("cancel-concurrently"))

        first, second = await asyncio.gather(
            store.cancel(queued.work_id),
            store.cancel(queued.work_id),
        )
        committed = await store.get(queued.work_id)

        assert first is not None and second is not None
        assert committed is not None
        assert first.state == second.state == committed.state == WorkState.CANCELLED
        assert first.finished_at == second.finished_at == committed.finished_at
        assert first.state == WorkState.CANCELLED
        assert first.finished_at is not None

    @pytest.mark.asyncio
    async def test_terminal_repeats_report_the_committed_final_state(self) -> None:
        store = self._store()
        running = await self._running(store, "already-succeeded")
        succeeded = await store.mark_succeeded(running.work_id)

        assert await store.cancel(succeeded.work_id) is None
        with pytest.raises(WorkStateConflictError):
            await store.mark_succeeded(succeeded.work_id)

        cancelled_queued = await store.enqueue(_item("already-cancelled"))
        cancelled = await store.cancel(cancelled_queued.work_id)
        assert cancelled is not None
        assert await store.cancel(cancelled.work_id) == cancelled
