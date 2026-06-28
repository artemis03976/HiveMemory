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
from hivememory.patchouli.control.memory_generation_tasks import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    MemoryGenerationTaskWaitSummary,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink


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
    pending_alias=None,
):
    return MemoryGenerationTaskSpec(
        topic_id=topic_id,
        label=label,
        source=source,
        request=GenerationRequest(context=GenerationContext()),
        source_intent=source.value,
        pending_alias=pending_alias,
    )


def _memory_task_statuses(bus):
    return [
        call.kwargs["status"]
        for call in bus.publish.await_args_list
        if call.args and call.args[0] == PatchouliLocalEvents.MEMORY_TASK_ITEM_STATUS
    ]


def _runtime_event_types_for_task(recorder, memory_task):
    return [
        event.event_type
        for event in recorder.events
        if event.task_id == memory_task.task_id
    ]


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

        memory_task = await controller.submit_generation(_spec())

        assert isinstance(memory_task, MemoryGenerationTask)
        assert memory_task._bg_task is not None
        assert not memory_task._bg_task.done()
        blocker.set()
        await memory_task._bg_task

    @pytest.mark.asyncio
    async def test_completed_task_publishes_terminal_status(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=[])
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        memory_task = await controller.submit_generation(_spec())
        await memory_task._bg_task

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert _memory_task_statuses(bus) == ["running", "completed"]

    @pytest.mark.asyncio
    async def test_failed_task_marked_failed(self):
        bus = Mock()
        bus.request = AsyncMock(side_effect=RuntimeError("generation error"))
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        memory_task = await controller.submit_generation(_spec())
        await memory_task._bg_task

        assert memory_task.status == MemoryGenerationTaskStatus.FAILED
        assert "generation error" in memory_task.error
        assert _memory_task_statuses(bus) == ["running", "failed"]

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
                pending_alias="draft_matrix",
                intent_id="intent_draft_matrix",
                canonical_alias="fact_matrix",
                settlement=settlement,
            )
        ]
        bus = Mock()
        bus.request = AsyncMock(return_value=results)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus, runtime_events=recorder)

        memory_task = await controller.submit_generation(
            _spec(
                topic_id="topic_active",
                label="draft_matrix",
                source=MemoryGenerationSource.WRITE,
                pending_alias="draft_matrix",
            )
        )
        await memory_task._bg_task

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task.canonical_alias == "fact_matrix"
        bus.publish.assert_any_await(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )
        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_COMPLETED,
        ]
        _assert_runtime_event_task_payload(recorder.events[-1], memory_task)

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

        memory_task = await controller.submit_generation(_spec())
        await asyncio.wait_for(started.wait(), timeout=1)

        assert await controller.cancel_task(memory_task.task_id) is True
        await asyncio.wait_for(released.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.CANCELLED
        assert _memory_task_statuses(bus) == ["running", "cancelled"]

    @pytest.mark.asyncio
    async def test_cancel_task_before_background_coroutine_starts_marks_cancelled(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=[])
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        memory_task = await controller.submit_generation(
            _spec(
                source=MemoryGenerationSource.WRITE,
                pending_alias="draft_early_cancel",
            )
        )

        assert await controller.cancel_task(memory_task.task_id) is True
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        await asyncio.sleep(0)

        assert memory_task.status == MemoryGenerationTaskStatus.CANCELLED
        assert memory_task.finished_at is not None
        assert bus.request.await_count == 0
        assert _memory_task_statuses(bus) == ["cancelled"]
        bus.publish.assert_any_await(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            pending_alias="draft_early_cancel",
        )

    @pytest.mark.asyncio
    async def test_finish_task_is_idempotent(self):
        bus = Mock()
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        memory_task = _task_handle()

        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)
        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.FAILED)

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert _memory_task_statuses(bus) == ["completed"]

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

        memory_task = await controller.submit_generation(_spec())
        waiter = asyncio.create_task(controller.wait_task(memory_task.task_id))

        await asyncio.sleep(0)
        assert not waiter.done()
        blocker.set()
        result = await waiter

        assert result.found is True
        assert result.timed_out is False
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

        memory_task = await controller.submit_generation(_spec())
        result = await controller.wait_task(memory_task.task_id, timeout=0.01)

        assert result.found is True
        assert result.timed_out is True
        assert memory_task._bg_task is not None
        assert not memory_task._bg_task.done()

        blocker.set()
        await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_task_returns_not_found_for_missing_task(self):
        bus = Mock()
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        result = await controller.wait_task("missing-task")

        assert result.found is False
        assert result.task_id == "missing-task"

    @pytest.mark.asyncio
    async def test_wait_many_summarizes_mixed_results(self):
        blocker = asyncio.Event()

        async def request(route, spec):
            await blocker.wait()
            return []

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)

        memory_task = await controller.submit_generation(_spec())
        summary = await controller.wait_many(
            [memory_task.task_id, "missing-task"],
            timeout=0.01,
        )

        assert isinstance(summary, MemoryGenerationTaskWaitSummary)
        assert summary.requested == 2
        assert summary.found == 1
        assert summary.missing == 1
        assert summary.timed_out == 1
        assert summary.running == 1

        blocker.set()
        await memory_task._bg_task

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

        first = await controller.submit_generation(_spec(label="first"))
        second = await controller.submit_generation(_spec(label="second"))
        waiter = asyncio.create_task(controller.wait_all())

        await asyncio.sleep(0)
        assert not waiter.done()
        blocker.set()
        summary = await waiter

        assert summary.requested == 2
        assert summary.completed == 2
        assert {result.task_id for result in summary.results} == {
            first.task_id,
            second.task_id,
        }
