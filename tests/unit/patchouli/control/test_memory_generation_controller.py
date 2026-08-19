import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.controller import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from tests.helpers.memory import make_memory_creation_context


def _task_handle(
    *,
    task_id="j1",
    topic_id="t1",
    source=MemoryGenerationSource.ARCHIVE,
):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=source,
    )


def _spec(
    *,
    topic_id="t1",
    label="t1",
    source=MemoryGenerationSource.ARCHIVE,
    intent_id=None,
    pending_alias=None,
):
    return MemoryGenerationTaskSpec(
        topic_id=topic_id,
        label=label,
        source=source,
        request=GenerationRequest(
            context=GenerationContext(),
            creation_context=make_memory_creation_context(),
        ),
        intent_id=intent_id,
        pending_alias=pending_alias,
    )


def _runtime_event_types_for_task(recorder, memory_task):
    return [event.event_type for event in recorder.events if event.task_id == memory_task.task_id]


def _assert_runtime_event_task_payload(event, memory_task):
    assert event.task_id == memory_task.task_id
    assert event.topic_id == memory_task.topic_id
    assert event.status == memory_task.status.value
    assert event.data["task_id"] == memory_task.task_id
    assert event.data["topic_id"] == memory_task.topic_id
    assert event.data["source"] == memory_task.source.value
    assert event.data["status"] == memory_task.status.value
    assert "pending_alias" in event.data
    assert "cancel_requested" in event.data
    assert "cancelled" in event.data


class TestMemoryGenerationTaskController:
    @pytest.mark.asyncio
    async def test_active_intent_reuses_existing_generation_task(self):
        blocker = asyncio.Event()
        recorder = RecordingRuntimeEventSink()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock(request=AsyncMock(side_effect=request), publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus, runtime_events=recorder)
        await controller.start()
        spec = _spec(
            source=MemoryGenerationSource.WRITE,
            intent_id="intent_stable",
            pending_alias="draft_stable",
        )

        first = await controller.submit_generation(spec)
        second = await controller.submit_generation(spec)

        assert first.task_id == "active:intent_stable"
        assert second.task_id == first.task_id
        created_events = [
            event
            for event in recorder.events
            if event.event_type == RuntimeEventType.MEMORY_TASK_CREATED
        ]
        assert len(created_events) == 1

        blocker.set()
        await controller.wait_task(first.task_id)
        assert bus.request.await_count == 1

    @pytest.mark.asyncio
    async def test_active_intent_rejects_conflicting_spec(self):
        bus = Mock(request=AsyncMock(return_value=[]), publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()
        first_spec = _spec(
            label="first",
            source=MemoryGenerationSource.WRITE,
            intent_id="intent_conflict",
            pending_alias="draft-conflict",
        )
        conflicting_spec = _spec(
            label="second",
            source=MemoryGenerationSource.WRITE,
            intent_id="intent_conflict",
            pending_alias="draft-conflict",
        )

        first = await controller.submit_generation(first_spec)
        with pytest.raises(ValueError, match="different spec"):
            await controller.submit_generation(conflicting_spec)
        assert await controller.submit_generation_many([conflicting_spec]) == []
        bus.publish.assert_not_awaited()

        await controller.wait_task(first.task_id)

    @pytest.mark.asyncio
    async def test_submit_generation_many_isolates_one_admission_failure(self):
        bus = Mock(request=AsyncMock(return_value=[]), publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()
        try:
            result = await controller.submit_generation_many(
                [
                    _spec(label="first"),
                    _spec(label="", pending_alias="draft-failed"),
                    _spec(label="third"),
                ]
            )

            assert [task.label for task in result] == ["first", "third"]
            bus.publish.assert_awaited_once_with(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias="draft-failed",
            )
            completed = await controller.wait_all(timeout=1)
            assert len(completed) == 2
            assert all(
                task.status == MemoryGenerationTaskStatus.COMPLETED
                for task in completed
            )
        finally:
            await controller.stop()

    @pytest.mark.asyncio
    async def test_submit_generation_many_keeps_unknown_admission_pending(self):
        queue = Mock()
        queue.start = AsyncMock()
        queue.started = True
        queue.submit = AsyncMock(side_effect=ConnectionError("response lost"))
        bus = Mock(publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus, memory_queue=queue)

        result = await controller.submit_generation_many(
            [
                _spec(
                    label="unknown",
                    intent_id="intent-unknown",
                    pending_alias="draft-unknown",
                ),
            ]
        )

        assert result == []
        bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_generation_returns_before_background_generation_completes(self):
        blocker = asyncio.Event()
        bus = Mock()

        async def request(route, spec):
            assert route == PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC
            await blocker.wait()
            return []

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())

        assert isinstance(memory_task, MemoryGenerationTask)
        waiter = asyncio.create_task(controller.wait_task(memory_task.task_id))
        await asyncio.sleep(0)
        assert not waiter.done()
        blocker.set()
        await waiter

    @pytest.mark.asyncio
    async def test_completed_task_does_not_publish_local_status_event(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=[])
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await controller.wait_task(memory_task.task_id)
        completed = await controller.get_task(memory_task.task_id)

        assert completed.status == MemoryGenerationTaskStatus.COMPLETED
        bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_task_marked_failed(self):
        bus = Mock()
        bus.request = AsyncMock(side_effect=RuntimeError("generation error"))
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await controller.wait_task(memory_task.task_id)
        failed = await controller.get_task(memory_task.task_id)

        assert failed.status == MemoryGenerationTaskStatus.FAILED
        assert "generation error" in failed.error

    @pytest.mark.asyncio
    async def test_active_success_publishes_settlement_and_runtime_events(self):
        recorder = RecordingRuntimeEventSink()
        settlement = PendingAtomSettlement(
            pending_alias="draft_matrix",
            intent_id="intent_draft_matrix",
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_matrix",
            canonical_uuid="uuid_matrix",
        )
        results = [
            MemoryGenerationResult(
                canonical_alias="fact_matrix",
                settlement=settlement,
            )
        ]
        bus = Mock()
        bus.request = AsyncMock(return_value=results)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus, runtime_events=recorder)
        await controller.start()

        memory_task = await controller.submit_generation(
            _spec(
                topic_id="topic_active",
                label="draft_matrix",
                source=MemoryGenerationSource.WRITE,
                pending_alias="draft_matrix",
            )
        )
        await controller.wait_task(memory_task.task_id)
        completed = await controller.get_task(memory_task.task_id)

        assert completed.status == MemoryGenerationTaskStatus.COMPLETED
        assert completed.canonical_alias == "fact_matrix"
        bus.publish.assert_any_await(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )
        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_COMPLETED,
        ]
        _assert_runtime_event_task_payload(recorder.events[-1], completed)

    @pytest.mark.asyncio
    async def test_cancel_task_via_registry_cancels_background_task(self):
        started = asyncio.Event()
        released = asyncio.Event()

        async def request(route, spec):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await asyncio.wait_for(started.wait(), timeout=1)

        assert await controller.cancel_task(memory_task.task_id) is True
        await asyncio.wait_for(released.wait(), timeout=1)
        result = await controller.wait_task(memory_task.task_id)
        assert result.status == MemoryGenerationTaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_task_before_background_coroutine_starts_marks_cancelled(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=[])
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(
            _spec(
                source=MemoryGenerationSource.WRITE,
                pending_alias="draft_early_cancel",
            )
        )

        assert await controller.cancel_task(memory_task.task_id) is True
        await controller.wait_task(memory_task.task_id)
        cancelled = await controller.get_task(memory_task.task_id)

        assert cancelled.status == MemoryGenerationTaskStatus.CANCELLED
        assert cancelled.finished_at is not None
        assert bus.request.await_count == 0
        bus.publish.assert_any_await(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            pending_alias="draft_early_cancel",
        )

    @pytest.mark.asyncio
    async def test_cancel_many_uses_shutdown_timeout_reason(self):
        recorder = RecordingRuntimeEventSink()
        started = asyncio.Event()
        released = asyncio.Event()

        async def request(route, spec):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(
            bus=bus,
            runtime_events=recorder,
        )
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await asyncio.wait_for(started.wait(), timeout=1)

        assert (
            await controller.cancel_many(
                [memory_task.task_id],
                reason="shutdown_timeout",
            )
            == 1
        )
        await asyncio.wait_for(released.wait(), timeout=1)
        await controller.wait_task(memory_task.task_id)

        task_events = [event for event in recorder.events if event.task_id == memory_task.task_id]
        assert task_events[-2].event_type == RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED
        assert task_events[-2].reason == "shutdown_timeout"
        assert task_events[-2].data["reason"] == "shutdown_timeout"
        assert task_events[-1].event_type == RuntimeEventType.MEMORY_TASK_CANCELLED
        assert task_events[-1].reason == "shutdown_timeout"
        assert task_events[-1].data["reason"] == "shutdown_timeout"

    @pytest.mark.asyncio
    async def test_terminal_snapshot_is_stable_across_repeated_waits(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=[])
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()
        memory_task = await controller.submit_generation(_spec())
        first = await controller.wait_task(memory_task.task_id)
        second = await controller.wait_task(memory_task.task_id)

        assert first == second

    @pytest.mark.asyncio
    async def test_running_status_is_published_before_work_completes(self):
        recorder = RecordingRuntimeEventSink()
        started = asyncio.Event()
        release = asyncio.Event()

        async def request(route, spec):
            started.set()
            await release.wait()
            return []

        bus = Mock(request=AsyncMock(side_effect=request), publish=AsyncMock())
        controller = MemoryGenerationTaskController(
            bus=bus,
            runtime_events=recorder,
        )
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await asyncio.wait_for(started.wait(), timeout=1)
        for _ in range(10):
            task_event_types = _runtime_event_types_for_task(recorder, memory_task)
            if RuntimeEventType.MEMORY_TASK_STATUS in task_event_types:
                break
            await asyncio.sleep(0)

        assert RuntimeEventType.MEMORY_TASK_STATUS in task_event_types
        assert (await controller.get_task(memory_task.task_id)).status == (
            MemoryGenerationTaskStatus.RUNNING
        )

        release.set()
        await controller.wait_task(memory_task.task_id)

    @pytest.mark.asyncio
    async def test_wait_task_waits_for_background_completion(self):
        blocker = asyncio.Event()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        waiter = asyncio.create_task(controller.wait_task(memory_task.task_id))

        await asyncio.sleep(0)
        assert not waiter.done()
        blocker.set()
        result = await waiter

        assert result is not None
        assert result.status == MemoryGenerationTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_task_timeout_does_not_cancel_background_task(self):
        blocker = asyncio.Event()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=0.01)

        assert result is not None
        assert result.status in {
            MemoryGenerationTaskStatus.PENDING,
            MemoryGenerationTaskStatus.RUNNING,
        }

        blocker.set()
        completed = await controller.wait_task(memory_task.task_id)
        assert completed.status == MemoryGenerationTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_task_returns_not_found_for_missing_task(self):
        bus = Mock()
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        result = await controller.wait_task("missing-task")

        assert result is None

    @pytest.mark.asyncio
    async def test_wait_many_returns_ordered_snapshots_and_missing_entries(self):
        blocker = asyncio.Event()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        results = await controller.wait_many(
            [memory_task.task_id, "missing-task"],
            timeout=0.01,
        )

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].task_id == memory_task.task_id
        assert results[0].status in {
            MemoryGenerationTaskStatus.PENDING,
            MemoryGenerationTaskStatus.RUNNING,
        }
        assert results[1] is None

        blocker.set()
        await controller.wait_task(memory_task.task_id)

    @pytest.mark.asyncio
    async def test_wait_all_waits_for_current_running_tasks(self):
        blocker = asyncio.Event()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        first = await controller.submit_generation(_spec(label="first"))
        second = await controller.submit_generation(_spec(label="second"))
        waiter = asyncio.create_task(controller.wait_all())

        await asyncio.sleep(0)
        assert not waiter.done()
        blocker.set()
        completed = await waiter

        assert all(
            task.status == MemoryGenerationTaskStatus.COMPLETED
            for task in completed
        )
        assert {task.task_id for task in completed} == {
            first.task_id,
            second.task_id,
        }
