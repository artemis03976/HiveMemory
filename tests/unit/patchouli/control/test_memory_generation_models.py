from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from hivememory.engines.generation.models import GenerationContext, GenerationRequest
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)
from hivememory.system.runtime.work_queue import (
    TaskOutcome,
    WorkItem,
    WorkRecord,
    WorkState,
)
from tests.helpers.memory import make_memory_creation_context


def test_memory_generation_source_derives_artifact_semantics():
    assert MemoryGenerationSource.ARCHIVE.creation_artifact_intent == "ARCHIVE"
    assert MemoryGenerationSource.WRITE.creation_artifact_intent == "WRITE"
    assert MemoryGenerationSource.UPDATE.creation_artifact_intent == "SYSTEM"
    assert MemoryGenerationSource.UPDATE.version_update_source == "UPDATE"


def _task_snapshot(**updates):
    values = {
        "task_id": "j1",
        "topic_id": "t1",
        "label": "t1",
        "source": MemoryGenerationSource.ARCHIVE,
    }
    values.update(updates)
    return MemoryGenerationTask(**values)


def _spec():
    return MemoryGenerationTaskSpec(
        topic_id="t1",
        label="task",
        source=MemoryGenerationSource.WRITE,
        request=GenerationRequest(
            context=GenerationContext(),
            creation_context=make_memory_creation_context(),
        ),
    )


def _outcome(
    state: WorkState,
    *,
    result: tuple[MemoryGenerationResult, ...] = (),
    error: str | None = None,
    cancel_reason: str | None = None,
) -> TaskOutcome[tuple[MemoryGenerationResult, ...]]:
    now = datetime.now(UTC)
    item = WorkItem(
        work_id="memory_generation:j1",
        lane="patchouli.memory_generation",
        kind="patchouli.memory_generation",
        schema_version=1,
        payload=b"{}",
    )
    record = WorkRecord(
        item=item,
        state=state,
        attempt_count=1,
        enqueued_at=now,
        available_at=now,
        started_at=now if state != WorkState.QUEUED else None,
        finished_at=(
            now
            if state
            in {
                WorkState.SUCCEEDED,
                WorkState.FAILED,
                WorkState.DEAD_LETTER,
                WorkState.CANCELLED,
            }
            else None
        ),
    )
    return TaskOutcome(
        record=record,
        result=result,
        error=error,
        cancel_reason=cancel_reason,
    )


@pytest.mark.parametrize(
    ("work_state", "task_status"),
    [
        (WorkState.QUEUED, MemoryGenerationTaskStatus.PENDING),
        (WorkState.RETRY_WAIT, MemoryGenerationTaskStatus.PENDING),
        (WorkState.RUNNING, MemoryGenerationTaskStatus.RUNNING),
        (WorkState.SUCCEEDED, MemoryGenerationTaskStatus.COMPLETED),
        (WorkState.CANCELLED, MemoryGenerationTaskStatus.CANCELLED),
        (WorkState.FAILED, MemoryGenerationTaskStatus.FAILED),
        (WorkState.DEAD_LETTER, MemoryGenerationTaskStatus.FAILED),
    ],
)
def test_memory_generation_task_maps_work_state_to_domain_status(
    work_state: WorkState,
    task_status: MemoryGenerationTaskStatus,
):
    created = MemoryGenerationTask.from_spec(
        "j1",
        _spec(),
        created_at=datetime.now(UTC),
    )

    snapshot = MemoryGenerationTask.from_outcome(
        created,
        _outcome(work_state),
        expose_terminal=True,
    )

    assert snapshot.status == task_status


def test_memory_generation_task_hides_queue_terminal_before_finalize():
    created = MemoryGenerationTask.from_spec(
        "j1",
        _spec(),
        created_at=datetime.now(UTC),
    )

    snapshot = MemoryGenerationTask.from_outcome(
        created,
        _outcome(WorkState.SUCCEEDED),
        expose_terminal=False,
    )

    assert snapshot.status == MemoryGenerationTaskStatus.RUNNING
    assert snapshot.finished_at is None


def test_memory_generation_task_projects_result_and_cancel_metadata():
    spec = MemoryGenerationTaskSpec(
        topic_id="t1",
        label="task",
        source=MemoryGenerationSource.WRITE,
        request=GenerationRequest(
            context=GenerationContext(),
            creation_context=make_memory_creation_context(),
        ),
        pending_alias="draft-target",
    )
    created = MemoryGenerationTask.from_spec(
        "j1",
        spec,
        created_at=datetime.now(UTC),
    )
    result = MemoryGenerationResult(
        canonical_alias="fact-target",
    )

    completed = MemoryGenerationTask.from_outcome(
        created,
        _outcome(WorkState.SUCCEEDED, result=(result,)),
        expose_terminal=True,
    )
    cancelled = MemoryGenerationTask.from_outcome(
        created,
        _outcome(WorkState.CANCELLED, cancel_reason="user_requested"),
        expose_terminal=True,
    )

    assert completed.canonical_alias == "fact-target"
    assert cancelled.cancel_requested is True
    assert cancelled.cancel_reason == "user_requested"


def test_memory_generation_task_uses_failed_outcome_error_fallback():
    created = MemoryGenerationTask.from_spec(
        "j1",
        _spec(),
        created_at=datetime.now(UTC),
    )

    snapshot = MemoryGenerationTask.from_outcome(
        created,
        _outcome(WorkState.FAILED),
        expose_terminal=True,
    )

    assert snapshot.error == "memory generation work failed"


def test_memory_generation_task_is_a_read_only_snapshot():
    snapshot = _task_snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.status = MemoryGenerationTaskStatus.COMPLETED


def test_memory_task_to_payload_contains_public_fields():
    snapshot = _task_snapshot(
        cancel_requested=True,
        cancel_reason="user_requested",
    )

    payload = memory_task_to_payload(snapshot)

    assert payload["task_id"] == "j1"
    assert payload["topic_id"] == "t1"
    assert payload["source"] == "ARCHIVE"
    assert payload["status"] == "pending"
    assert payload["cancel_requested"] is True
    assert payload["cancelled"] is False
    assert payload["reason"] == "user_requested"


def test_memory_task_to_payload_accepts_explicit_reason():
    assert memory_task_to_payload(_task_snapshot(), reason="system")["reason"] == "system"
