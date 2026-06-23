"""Phase 2: MemoryGenerationTask runtime 单元测试"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.core.models.pending import (
    Identity,
    PendingAtomMaterializeTask,
    WriteFocus,
    UpdateFocus,
)
from hivememory.engines.generation.models import GenerationContext, MemoryGenerationResult
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)
from hivememory.patchouli.services.librarian import LibrarianCore
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink


def _make_core(mock_generation=None, mock_storage=None, bus=None, runtime_events=None):
    from hivememory.patchouli.control.memory_generation_tasks import MemoryGenerationTaskController
    gen = mock_generation if mock_generation is not None else MagicMock()
    if mock_generation is None:
        gen.process = AsyncMock(return_value=[])
    storage = mock_storage if mock_storage is not None else MagicMock()
    if mock_storage is None:
        storage.get_memory = AsyncMock(return_value=MagicMock())
    perception_layer = MagicMock()
    perception_layer.get_topic_context.return_value = {
        "state_summary": "",
        "blocks": [],
    }
    task_controller = MemoryGenerationTaskController(
        storage=storage,
        generation_engine=gen,
        bus=bus or AsyncMock(),
        runtime_events=runtime_events,
    )
    core = LibrarianCore(
        storage=storage,
        bus=bus or AsyncMock(),
        task_controller=task_controller,
        perception_layer=perception_layer,
    )
    return core, gen


def _write_task(alias="draft_001"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        identity=Identity(user_id="test_user"),
        focus=WriteFocus(content="test content"),
    )


def _task_handle(task_id="j1", topic_id="t1"):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.ARCHIVE,
    )


async def _single_memory_task(core, task=None, topic_id="t1"):
    memory_tasks = await core.run_active_generation(
        [task or _write_task()],
        topic_id=topic_id,
    )
    assert len(memory_tasks) == 1
    return memory_tasks[0]


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


class TestMemoryGenerationTaskRegistry:
    def test_register_and_get(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        assert reg.get("j1") is memory_task

    def test_cancel_existing(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        assert reg.cancel("j1") is True
        assert memory_task.cancelled is True

    def test_cancel_missing_returns_false(self):
        reg = MemoryGenerationTaskRegistry()
        assert reg.cancel("nonexistent") is False

    def test_close_sets_status_and_finished_at(self):
        reg = MemoryGenerationTaskRegistry()
        memory_task = _task_handle()
        reg.register(memory_task)
        reg.close("j1", MemoryGenerationTaskStatus.COMPLETED)
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task.finished_at is not None

    def test_evicts_old_completed_tasks(self):
        reg = MemoryGenerationTaskRegistry(max_completed=2)
        for i in range(3):
            j = _task_handle(task_id=f"j{i}", topic_id="t")
            reg.register(j)
            reg.close(f"j{i}", MemoryGenerationTaskStatus.COMPLETED)
        # Only 2 completed tasks retained
        assert len(reg.list_all()) <= 2

    def test_list_all(self):
        reg = MemoryGenerationTaskRegistry()
        j1 = _task_handle(task_id="j1", topic_id="t")
        j2 = _task_handle(task_id="j2", topic_id="t")
        reg.register(j1)
        reg.register(j2)
        assert len(reg.list_all()) == 2


class TestMemoryGenerationTaskPayload:
    def test_memory_task_to_payload_contains_public_fields(self):
        memory_task = _task_handle()
        memory_task.request_cancel()

        payload = memory_task_to_payload(memory_task)

        assert payload["task_id"] == "j1"
        assert payload["topic_id"] == "t1"
        assert payload["source"] == "ARCHIVE"
        assert payload["status"] == "pending"
        assert payload["cancel_requested"] is True
        assert payload["cancelled"] is False
        assert payload["reason"] == "user_requested"
        assert payload["created_at"] == memory_task.created_at.isoformat()

    def test_memory_task_to_payload_accepts_explicit_reason(self):
        memory_task = _task_handle()

        payload = memory_task_to_payload(memory_task, reason="system")

        assert payload["reason"] == "system"


class TestRunActiveGenerationReturnsTasks:
    @pytest.mark.asyncio
    async def test_returns_single_memory_generation_task_per_pending_atom(self):
        core, _ = _make_core()
        result = await _single_memory_task(core)
        assert isinstance(result, MemoryGenerationTask)
        assert result.topic_id == "t1"

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_empty_list(self):
        core, _ = _make_core()
        memory_tasks = await core.run_active_generation([], topic_id="t1")
        assert memory_tasks == []

    @pytest.mark.asyncio
    async def test_returns_before_generation_completes(self):
        """run_active_generation 必须立即返回，不等待后台 task。"""
        blocker = asyncio.Event()

        gen = MagicMock()

        async def slow_process(_):
            await blocker.wait()
            return []

        gen.process = slow_process
        core, _ = _make_core(mock_generation=gen)

        memory_task = await _single_memory_task(core)
        # Task was returned while bg_task is still pending
        assert memory_task._bg_task is not None
        assert not memory_task._bg_task.done()
        blocker.set()
        await memory_task._bg_task  # cleanup


class TestTaskLifecycleAfterCompletion:
    @pytest.mark.asyncio
    async def test_completed_after_bg_task_finishes(self):
        core, _ = _make_core()
        memory_task = await _single_memory_task(core)
        if memory_task._bg_task:
            await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_completed_task_publishes_terminal_status(self):
        bus = AsyncMock()
        core, _ = _make_core(bus=bus)
        memory_task = await _single_memory_task(core)
        if memory_task._bg_task:
            await memory_task._bg_task

        assert _memory_task_statuses(bus) == ["running", "completed"]

    @pytest.mark.asyncio
    async def test_completed_task_publishes_runtime_events(self):
        recorder = RecordingRuntimeEventSink()
        core, _ = _make_core(runtime_events=recorder)
        memory_task = await _single_memory_task(core)
        if memory_task._bg_task:
            await memory_task._bg_task

        event_types = [event.event_type for event in recorder.events]
        assert RuntimeEventType.MEMORY_TASK_CREATED in event_types
        assert RuntimeEventType.MEMORY_TASK_STATUS in event_types
        assert RuntimeEventType.MEMORY_TASK_COMPLETED in event_types
        assert recorder.events[0].task_id == memory_task.task_id
        assert recorder.events[0].data["task_id"] == memory_task.task_id
        assert recorder.events[0].data["topic_id"] == memory_task.topic_id
        assert recorder.events[0].data["cancel_requested"] is False
        assert recorder.events[0].data["cancelled"] is False

    @pytest.mark.asyncio
    async def test_task_metadata_updated(self):
        core, _ = _make_core()
        task = _write_task("draft_abc")
        memory_task = await _single_memory_task(core, task)
        if memory_task._bg_task:
            await memory_task._bg_task
        assert memory_task.label == "draft_abc"
        assert memory_task.pending_alias == "draft_abc"
        assert memory_task.source == MemoryGenerationSource.WRITE

    @pytest.mark.asyncio
    async def test_failed_task_marked_failed(self):
        gen = MagicMock()
        gen.process.side_effect = RuntimeError("generation error")
        core, _ = _make_core(mock_generation=gen)
        memory_task = await _single_memory_task(core)
        if memory_task._bg_task:
            await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.FAILED
        assert "generation error" in memory_task.error

    @pytest.mark.asyncio
    async def test_failed_task_publishes_terminal_status(self):
        bus = AsyncMock()
        gen = MagicMock()
        gen.process.side_effect = RuntimeError("generation error")
        core, _ = _make_core(mock_generation=gen, bus=bus)
        memory_task = await _single_memory_task(core)
        if memory_task._bg_task:
            await memory_task._bg_task

        assert _memory_task_statuses(bus) == ["running", "failed"]


class TestMemoryTaskRuntimeEventMatrix:
    @pytest.mark.asyncio
    async def test_archive_success_pushes_created_status_completed(self):
        recorder = RecordingRuntimeEventSink()
        core, _ = _make_core(runtime_events=recorder)
        controller = core._memory_task_controller

        memory_task = await controller.run_archive_generation(
            topic_id="topic_archive",
            gen_context=GenerationContext(),
        )
        if memory_task._bg_task:
            await memory_task._bg_task

        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_COMPLETED,
        ]
        assert memory_task.source == MemoryGenerationSource.ARCHIVE
        _assert_runtime_event_task_payload(recorder.events[-1], memory_task)
        assert recorder.events[-1].data["pending_alias"] is None
        assert recorder.events[-1].data["cancelled"] is False

    @pytest.mark.asyncio
    async def test_archive_failure_pushes_created_status_failed(self):
        recorder = RecordingRuntimeEventSink()
        gen = MagicMock()
        gen.process.side_effect = RuntimeError("archive boom")
        core, _ = _make_core(mock_generation=gen, runtime_events=recorder)
        controller = core._memory_task_controller

        memory_task = await controller.run_archive_generation(
            topic_id="topic_archive",
            gen_context=GenerationContext(),
        )
        if memory_task._bg_task:
            await memory_task._bg_task

        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_FAILED,
        ]
        assert memory_task.status == MemoryGenerationTaskStatus.FAILED
        assert "archive boom" in memory_task.error
        _assert_runtime_event_task_payload(recorder.events[-1], memory_task)
        assert recorder.events[-1].severity == "error"
        assert recorder.events[-1].data["error"] == "archive boom"

    @pytest.mark.asyncio
    async def test_active_success_pushes_created_status_completed(self):
        recorder = RecordingRuntimeEventSink()
        task = _write_task("draft_matrix")
        settlement = PendingAtomSettlement(
            pending_alias=task.pending_alias,
            intent_id=task.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_matrix",
            canonical_uuid="uuid_matrix",
        )
        results = [
            MemoryGenerationResult(
                pending_alias=task.pending_alias,
                intent_id=task.intent_id,
                canonical_alias="fact_matrix",
                settlement=settlement,
            )
        ]
        gen = MagicMock()
        gen.process = AsyncMock(return_value=results)
        core, _ = _make_core(mock_generation=gen, runtime_events=recorder)

        memory_task = await _single_memory_task(core, task, topic_id="topic_active")
        if memory_task._bg_task:
            await memory_task._bg_task

        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_COMPLETED,
        ]
        assert memory_task.source == MemoryGenerationSource.WRITE
        assert memory_task.canonical_alias == "fact_matrix"
        _assert_runtime_event_task_payload(recorder.events[-1], memory_task)
        assert recorder.events[-1].data["pending_alias"] == "draft_matrix"
        assert recorder.events[-1].data["canonical_alias"] == "fact_matrix"

    @pytest.mark.asyncio
    async def test_active_cancel_pushes_cancel_requested_and_cancelled(self):
        recorder = RecordingRuntimeEventSink()
        started = asyncio.Event()
        released = asyncio.Event()

        async def blocking_process(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        gen = MagicMock()
        gen.process = blocking_process
        core, _ = _make_core(mock_generation=gen, runtime_events=recorder)

        memory_task = await _single_memory_task(core, topic_id="topic_active")
        assert memory_task._bg_task is not None
        await asyncio.wait_for(started.wait(), timeout=1)

        assert core.cancel_task(memory_task.task_id) is True
        await asyncio.wait_for(released.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        await asyncio.sleep(0)

        assert _runtime_event_types_for_task(recorder, memory_task) == [
            RuntimeEventType.MEMORY_TASK_CREATED,
            RuntimeEventType.MEMORY_TASK_STATUS,
            RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED,
            RuntimeEventType.MEMORY_TASK_CANCELLED,
        ]
        assert memory_task.status == MemoryGenerationTaskStatus.CANCELLED
        _assert_runtime_event_task_payload(recorder.events[-1], memory_task)
        assert recorder.events[-2].reason == "user_requested"
        assert recorder.events[-2].data["cancel_requested"] is True
        assert recorder.events[-1].data["cancelled"] is True


class TestTaskCancellation:
    @pytest.mark.asyncio
    async def test_cancel_individual_tasks(self):
        blocker = asyncio.Event()
        ran_tasks = []

        async def blocking_process(_):
            ran_tasks.append(1)
            await blocker.wait()
            return []

        gen = MagicMock()
        gen.process = blocking_process
        core, _ = _make_core(mock_generation=gen)

        # Each pending atom gets its own memory task handle.
        task1 = _write_task("draft_001")
        task2 = _write_task("draft_002")
        memory_tasks = await core.run_active_generation([task1, task2], topic_id="t1")
        assert len(memory_tasks) == 2

        # Cancel immediately
        memory_tasks[0].request_cancel()
        memory_tasks[1].request_cancel()
        blocker.set()
        for memory_task in memory_tasks:
            if memory_task._bg_task:
                await memory_task._bg_task

        # Cancellation applies per returned memory task handle.
        assert len(ran_tasks) <= 2

    @pytest.mark.asyncio
    async def test_cancel_task_via_registry(self):
        recorder = RecordingRuntimeEventSink()
        core, _ = _make_core(runtime_events=recorder)
        memory_task = await _single_memory_task(core)
        ok = core.cancel_task(memory_task.task_id)
        assert ok is True
        assert memory_task.cancelled is True
        assert RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED in [
            event.event_type for event in recorder.events
        ]

    @pytest.mark.asyncio
    async def test_cancel_task_via_registry_cancels_background_task(self):
        started = asyncio.Event()
        released = asyncio.Event()

        async def blocking_process(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        gen = MagicMock()
        gen.process = blocking_process
        core, _ = _make_core(mock_generation=gen)

        memory_task = await _single_memory_task(core)
        assert memory_task._bg_task is not None
        await asyncio.wait_for(started.wait(), timeout=1)

        ok = core.cancel_task(memory_task.task_id)
        assert ok is True
        await asyncio.wait_for(released.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        await asyncio.sleep(0)
        assert memory_task.status == MemoryGenerationTaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_task_publishes_terminal_status(self):
        bus = AsyncMock()
        started = asyncio.Event()
        released = asyncio.Event()

        async def blocking_process(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        gen = MagicMock()
        gen.process = blocking_process
        core, _ = _make_core(mock_generation=gen, bus=bus)

        memory_task = await _single_memory_task(core)
        assert memory_task._bg_task is not None
        await asyncio.wait_for(started.wait(), timeout=1)

        assert core.cancel_task(memory_task.task_id) is True
        await asyncio.wait_for(released.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        await asyncio.sleep(0)

        assert _memory_task_statuses(bus) == ["running", "cancelled"]

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_returns_false(self):
        core, _ = _make_core()
        assert core.cancel_task("nonexistent-id") is False

    @pytest.mark.asyncio
    async def test_finish_task_is_idempotent(self):
        bus = AsyncMock()
        core, _ = _make_core(bus=bus)
        memory_task = _task_handle()
        controller = core._memory_task_controller

        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)
        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.FAILED)

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert _memory_task_statuses(bus) == ["completed"]

    @pytest.mark.asyncio
    async def test_finish_task_closes_registry_when_status_publish_fails(self):
        bus = AsyncMock()
        bus.publish.side_effect = RuntimeError("publish failed")
        core, _ = _make_core(bus=bus)
        memory_task = _task_handle()
        controller = core._memory_task_controller
        controller._task_registry.register(memory_task)

        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task.finished_at is not None
        assert memory_task._terminal_status_published is True

    @pytest.mark.asyncio
    async def test_finish_task_swallow_cancelled_error_during_status_publish(self):
        core, _ = _make_core()
        memory_task = _task_handle()
        controller = core._memory_task_controller
        controller._task_registry.register(memory_task)

        async def cancelled_publish(_):
            raise asyncio.CancelledError()

        controller._publish_memory_task_status = cancelled_publish

        await controller._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)

        assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
        assert memory_task.finished_at is not None
        assert memory_task._terminal_status_published is True


class TestTaskQueryApi:
    @pytest.mark.asyncio
    async def test_get_task_by_id(self):
        core, _ = _make_core()
        memory_task = await _single_memory_task(core)
        found = core.get_task(memory_task.task_id)
        assert found is memory_task

    @pytest.mark.asyncio
    async def test_list_tasks_includes_task(self):
        core, _ = _make_core()
        memory_task = await _single_memory_task(core)
        all_tasks = core.list_tasks()
        assert memory_task in all_tasks

    def test_get_nonexistent_returns_none(self):
        core, _ = _make_core()
        assert core.get_task("does-not-exist") is None
