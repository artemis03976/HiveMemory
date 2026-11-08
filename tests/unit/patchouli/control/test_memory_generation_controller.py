"""生成控制器的接纳、终态发布与后续合并调度测试。"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import MemoryMergeAbortedError
from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    MergeFocus,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmissionStatus
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
from tests.helpers.memory import make_memory_identity_scope


def _task_handle(
    *,
    task_id="j1",
    topic_id="t1",
    source=MemoryGenerationSource.SETTLE,
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
    source=MemoryGenerationSource.SETTLE,
    intent_id=None,
    pending_alias=None,
    request=None,
):
    return MemoryGenerationTaskSpec(
        belong_to=make_memory_identity_scope().workspace_identity,
        from_actor=make_memory_identity_scope().actor_identity,
        topic_id=topic_id,
        label=label,
        source=source,
        request=request
        or GenerationRequest(
            context=GenerationContext(),
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
        controller = MemoryGenerationTaskController(
            bus=bus, runtime_events=RuntimeEventPublisher(recorder)
        )
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
        conflict = await controller.admit_generation(conflicting_spec)
        # 既有任务仍拥有该 intent 的结算权：冲突按结果未知处理，不发布失败。
        assert conflict.status is MemoryGenerationAdmissionStatus.UNKNOWN
        bus.publish.assert_not_awaited()

        await controller.wait_task(first.task_id)

    @pytest.mark.asyncio
    async def test_admit_generation_settles_deterministic_rejection_as_failed(self):
        """确定性拒绝结算关联的 PendingAtom 为失败，不伪造 FAILED 任务。"""
        bus = Mock(request=AsyncMock(return_value=[]), publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()
        try:
            admission = await controller.admit_generation(
                _spec(label="", intent_id="intent-failed", pending_alias="draft-failed")
            )

            assert admission.status is MemoryGenerationAdmissionStatus.REJECTED
            assert admission.task is None
            assert await controller.list_tasks() == []
            bus.publish.assert_awaited_once_with(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias="draft-failed",
                intent_id="intent-failed",
            )
        finally:
            await controller.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure", [ConnectionError("response lost"), ValueError("after admission")]
    )
    async def test_admit_generation_keeps_unknown_admission_pending(self, failure):
        queue = Mock()
        queue.start = AsyncMock()
        queue.started = True
        queue.submit = AsyncMock(side_effect=failure)
        bus = Mock(publish=AsyncMock())
        controller = MemoryGenerationTaskController(bus=bus, memory_queue=queue)

        admission = await controller.admit_generation(
            _spec(
                label="unknown",
                intent_id="intent-unknown",
                pending_alias="draft-unknown",
            )
        )

        assert admission.status is MemoryGenerationAdmissionStatus.UNKNOWN
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
        assert failed.error == "记忆生成失败，请检查服务状态"

    @pytest.mark.asyncio
    async def test_failed_task_events_carry_the_work_record_reason_code(self):
        """工作记录、work.failed 与 memory.task.failed 带同一个原因码，任务事件载荷带异常类名。"""
        recorder = RecordingRuntimeEventSink()
        bus = Mock()
        bus.request = AsyncMock(
            side_effect=MemoryMergeAbortedError(
                details={"reason": "version_changed", "memory_id": "m1"}
            )
        )
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(
            bus=bus, runtime_events=RuntimeEventPublisher(recorder)
        )
        await controller.start()

        memory_task = await controller.submit_generation(_spec())
        await controller.wait_task(memory_task.task_id)
        record = await controller.queue.get(controller.queue.work_id_for(memory_task.task_id))

        reason = "merge_aborted.version_changed"
        assert record.last_error.message == reason
        (work_failed,) = [
            e for e in recorder.events if e.event_type == RuntimeEventType.WORK_FAILED
        ]
        (task_failed,) = [
            e for e in recorder.events if e.event_type == RuntimeEventType.MEMORY_TASK_FAILED
        ]
        assert work_failed.reason == reason
        assert (
            task_failed.reason,
            task_failed.data["reason"],
            task_failed.data["error_class"],
        ) == (
            reason,
            reason,
            "MemoryMergeAbortedError",
        )

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
        controller = MemoryGenerationTaskController(
            bus=bus, runtime_events=RuntimeEventPublisher(recorder)
        )
        await controller.start()

        memory_task = await controller.submit_generation(
            _spec(
                topic_id="topic_active",
                label="draft_matrix",
                source=MemoryGenerationSource.WRITE,
                intent_id="intent_draft_matrix",
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
    async def test_topic_write_memories_reach_later_write_and_settle_of_that_topic(self):
        """主动写入结算到的记忆记入任务话题：之后同一话题的 WRITE 执行时带上、不清除，
        SETTLE 执行时带上并清除；其他话题与 UPDATE 不读取这个话题的记录。"""

        def written(label, memory_id, resolution=PendingAtomResolution.CREATED):
            return [
                MemoryGenerationResult(
                    canonical_alias=f"fact_{label}",
                    settlement=PendingAtomSettlement(
                        pending_alias=label,
                        intent_id=f"intent_{label}",
                        resolution=resolution,
                        canonical_alias=f"fact_{label}",
                        canonical_uuid=memory_id,
                    ),
                )
            ]

        replies = {
            "w1": written("w1", "m1"),
            "u1": written("u1", "m2", PendingAtomResolution.UPDATED),
            "wb": written("wb", "mb"),
            "w2": written("w2", "m3"),
        }
        seen = []

        async def request(route, spec):
            seen.append((spec.label, spec.request.topic_write_memory_ids))
            return replies.get(spec.label, [])

        bus = Mock()
        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        controller = MemoryGenerationTaskController(bus=bus)
        await controller.start()

        write, update, settle = (
            MemoryGenerationSource.WRITE,
            MemoryGenerationSource.UPDATE,
            MemoryGenerationSource.SETTLE,
        )
        for topic_id, label, source in [
            ("topic_a", "w1", write),
            ("topic_a", "u1", update),
            ("topic_b", "wb", write),
            ("topic_a", "w2", write),
            ("topic_a", "settle_a", settle),
            ("topic_a", "settle_a_again", settle),
            ("topic_b", "settle_b", settle),
        ]:
            intent = (f"intent_{label}", label) if source != settle else (None, None)
            memory_task = await controller.submit_generation(
                _spec(
                    topic_id=topic_id,
                    label=label,
                    source=source,
                    intent_id=intent[0],
                    pending_alias=intent[1],
                )
            )
            await controller.wait_task(memory_task.task_id)

        assert seen == [
            ("w1", frozenset()),
            ("u1", frozenset()),
            ("wb", frozenset()),
            ("w2", frozenset({"m1", "m2"})),
            ("settle_a", frozenset({"m1", "m2", "m3"})),
            ("settle_a_again", frozenset()),
            ("settle_b", frozenset({"mb"})),
        ]

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
                intent_id="intent_early_cancel",
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
            intent_id="intent_early_cancel",
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
            runtime_events=RuntimeEventPublisher(recorder),
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
            runtime_events=RuntimeEventPublisher(recorder),
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

        assert all(task.status == MemoryGenerationTaskStatus.COMPLETED for task in completed)
        assert {task.task_id for task in completed} == {
            first.task_id,
            second.task_id,
        }


def _merge_spec():
    """合并规范只在 focus 中保留改指坐标，任务本身不持有写入意图。"""
    return _spec(
        source=MemoryGenerationSource.MERGE,
        request=GenerationRequest(
            merge_focus=MergeFocus(
                retained_memory_id="retained",
                absorbed_memory_id="absorbed",
                intent_id="original",
                pending_alias="draft",
            )
        ),
    )


@pytest.mark.asyncio
async def test_successful_nomination_settles_first_then_submits_and_redirects():
    """提名沿用原任务归属与话题，首次结算先于合并，成功后发布独立改指事件。"""
    events = []
    executed = []
    created = PendingAtomSettlement(
        pending_alias="draft",
        intent_id="original",
        resolution=PendingAtomResolution.CREATED,
        canonical_uuid="absorbed",
    )
    merged = created.model_copy(
        update={
            "resolution": PendingAtomResolution.MERGED,
            "canonical_uuid": "retained",
            "canonical_alias": "fact_retained",
        }
    )

    async def execute(route, spec):
        executed.append(spec)
        if spec.source == MemoryGenerationSource.WRITE:
            return [
                MemoryGenerationResult(
                    settlement=created, merge_nomination=_merge_spec().request.merge_focus
                )
            ]
        assert events == [(PatchouliLocalEvents.PENDING_ATOM_SETTLED, {"settlement": created})]
        return [MemoryGenerationResult(settlement=merged, absorbed_memory_uuid="absorbed")]

    async def publish(event, **payload):
        events.append((event, payload))

    controller = MemoryGenerationTaskController(
        bus=Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock(side_effect=publish))
    )
    await controller.start()
    try:
        original = _spec(
            source=MemoryGenerationSource.WRITE, pending_alias="draft", intent_id="original"
        )
        task = await controller.submit_generation(original)
        completed = await controller.wait_task(task.task_id)
        tasks = await controller.list_tasks()
        merge_task = next(item for item in tasks if item.source == MemoryGenerationSource.MERGE)
        merged_task = await controller.wait_task(merge_task.task_id)

        assert (completed.status, merged_task.status, merged_task.canonical_alias) == (
            MemoryGenerationTaskStatus.COMPLETED,
            MemoryGenerationTaskStatus.COMPLETED,
            "fact_retained",
        )
        spec = executed[1]
        assert (spec.belong_to, spec.from_actor, spec.topic_id) == (
            original.belong_to,
            original.from_actor,
            original.topic_id,
        )
        assert (
            spec.intent_id,
            spec.pending_alias,
            spec.interaction_input,
            spec.request.context.turns,
        ) == (None, None, None, [])
        assert merge_task.task_id != task.task_id
        assert events[-1] == (
            PatchouliLocalEvents.PENDING_ATOM_MERGED,
            {"settlement": merged, "absorbed_memory_uuid": "absorbed"},
        )
    finally:
        await controller.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [WorkQueueStoppedError("stopped"), WorkQueueCapacityError("generation", 1)]
)
async def test_merge_submission_rejection_keeps_original_success(error, monkeypatch, caplog):
    created = PendingAtomSettlement(
        pending_alias="draft",
        intent_id="original",
        resolution=PendingAtomResolution.CREATED,
        canonical_uuid="absorbed",
    )
    bus = Mock(
        request=AsyncMock(
            return_value=[
                MemoryGenerationResult(
                    settlement=created, merge_nomination=_merge_spec().request.merge_focus
                )
            ]
        ),
        publish=AsyncMock(),
    )
    controller = MemoryGenerationTaskController(bus=bus)
    submit = controller.queue.submit

    async def submit_or_reject(task_id, spec):
        if spec.source == MemoryGenerationSource.MERGE:
            raise error
        return await submit(task_id, spec)

    monkeypatch.setattr(controller.queue, "submit", submit_or_reject)
    await controller.start()
    try:
        task = await controller.submit_generation(
            _spec(source=MemoryGenerationSource.WRITE, pending_alias="draft", intent_id="original")
        )
        completed = await controller.wait_task(task.task_id)
        assert completed.status == MemoryGenerationTaskStatus.COMPLETED
        assert [item.task_id for item in await controller.list_tasks()] == [task.task_id]
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED, settlement=created
        )
        assert "Memory merge submission rejected" in caplog.text
    finally:
        await controller.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["failure", "cancel"])
async def test_merge_failure_or_cancel_never_settles_original_intent(cancel):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(route, spec):
        entered.set()
        await release.wait()
        raise RuntimeError("merge failed")

    bus = Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock())
    controller = MemoryGenerationTaskController(bus=bus)
    await controller.start()
    try:
        task = await controller.submit_generation(_merge_spec())
        async with asyncio.timeout(5):
            await entered.wait()
        if cancel:
            assert await controller.cancel_task(task.task_id)
        else:
            release.set()
        completed = await controller.wait_task(task.task_id)
        assert completed.status == (
            MemoryGenerationTaskStatus.CANCELLED if cancel else MemoryGenerationTaskStatus.FAILED
        )
        bus.publish.assert_not_awaited()
    finally:
        release.set()
        await controller.stop()


@pytest.mark.asyncio
async def test_shutdown_wait_excludes_pending_merge():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(route, spec):
        entered.set()
        await release.wait()
        return []

    controller = MemoryGenerationTaskController(
        bus=Mock(request=AsyncMock(side_effect=execute), publish=AsyncMock())
    )
    await controller.start()
    try:
        task = await controller.submit_generation(_merge_spec())
        async with asyncio.timeout(5):
            await entered.wait()
            tasks = await controller.wait_all(include_merges=False)
        assert tasks == []
        assert (
            await controller.get_task(task.task_id)
        ).status == MemoryGenerationTaskStatus.RUNNING
    finally:
        release.set()
        await controller.stop()
