import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.core.models.pending import Identity, PendingAtomMaterializeTask, WriteFocus
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    MemoryGenerationResult,
)
from hivememory.engines.perception.models import LogicalBlock, TopicMaterializeTask
from hivememory.core.models import TurnRecord
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation_coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation_tasks import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink


def _task_handle(task_id="j1", topic_id="t1"):
    return MemoryGenerationTask(
        task_id=task_id,
        topic_id=topic_id,
        label=topic_id,
        source=MemoryGenerationSource.ARCHIVE,
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


def _write_task(alias="draft_001"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        identity=Identity(user_id="test_user"),
        focus=WriteFocus(content="test content"),
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
        assert MemoryGenerationTaskRegistry().cancel("missing") is False

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
            task = _task_handle(task_id=f"j{i}", topic_id="t")
            reg.register(task)
            reg.close(f"j{i}", MemoryGenerationTaskStatus.COMPLETED)
        assert len(reg.list_all()) <= 2


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

    def test_memory_task_to_payload_accepts_explicit_reason(self):
        assert memory_task_to_payload(_task_handle(), reason="system")["reason"] == "system"


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

        assert controller.cancel_task(memory_task.task_id) is True
        await asyncio.wait_for(released.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await memory_task._bg_task
        assert memory_task.status == MemoryGenerationTaskStatus.CANCELLED
        assert _memory_task_statuses(bus) == ["running", "cancelled"]

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


class TestMemoryGenerationCoordinator:
    @pytest.mark.asyncio
    async def test_submit_settlement_builds_archive_spec(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=_task_handle())
        coordinator = MemoryGenerationCoordinator(bus=bus)
        payload = TopicMaterializeTask(
            topic_id="t1",
            topic_title="title",
            topic_summary="summary",
            blocks=[
                LogicalBlock(
                    turn=TurnRecord(user_query="q", assistant_final_text="a")
                )
            ],
            state_summary="state",
        )

        task = await coordinator.submit_settlement(payload)

        assert isinstance(task, MemoryGenerationTask)
        route, spec = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION
        assert spec.topic_id == "t1"
        assert spec.source == MemoryGenerationSource.ARCHIVE
        assert spec.request.context.state_summary == "state"
        assert spec.interaction_input.topic_title == "title"

    @pytest.mark.asyncio
    async def test_submit_active_builds_write_specs_from_topic_context(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [
            LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))
        ]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_task_handle(source=MemoryGenerationSource.WRITE)] if False else [_task_handle()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active([_write_task("draft_1")], "t1")

        assert len(result) == 1
        specs = bus.request.await_args_list[-1].args[1]
        assert len(specs) == 1
        spec = specs[0]
        assert spec.source == MemoryGenerationSource.WRITE
        assert spec.pending_alias == "draft_1"
        assert spec.request.is_write is True
        assert spec.request.identity.user_id == "test_user"
