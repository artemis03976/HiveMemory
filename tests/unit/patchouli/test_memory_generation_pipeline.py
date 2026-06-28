from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    TurnRecord,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.perception.models import LogicalBlock, TopicMaterializeTask
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation_coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation_tasks import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskStatus,
)


class _TopicData:
    topic_title = "topic title"
    topic_summary = "topic summary"
    state_summary = "state summary"

    def __init__(self) -> None:
        self._blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    user_query="question",
                    assistant_final_text="answer",
                )
            )
        ]

    def recent_blocks(self, limit: int):
        return self._blocks[:limit]


def _memory_atom(memory_id=None) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id or uuid4(),
        meta=MetaData(source_agent_id="agent-1", user_id="u1"),
        index=IndexLayer(
            title="memory title",
            summary="summary text",
            tags=["tag"],
            memory_type=MemoryType.FACT,
            alias="memory_alias",
        ),
        payload=PayloadLayer(content="content"),
    )


def _write_task(alias="draft_write") -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        focus=WriteFocus(content="remember this"),
    )


def _update_task(base_uuid: str, alias="draft_update") -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        identity=Identity(user_id="u1", agent_id="omni_doll"),
        focus=UpdateFocus(
            instruction="merge this",
            content="new content",
            base_uuid=base_uuid,
            base_alias="memory_alias",
        ),
    )


def _settlement_result(alias="draft_write") -> list[MemoryGenerationResult]:
    settlement = PendingAtomSettlement(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        resolution=PendingAtomResolution.CREATED,
        canonical_alias="memory_alias",
        canonical_uuid=str(uuid4()),
    )
    return [
        MemoryGenerationResult(
            pending_alias=alias,
            intent_id=f"intent_{alias}",
            canonical_alias="memory_alias",
            settlement=settlement,
        )
    ]


def _wire_generation_pipeline(bus: PatchouliBus) -> MemoryGenerationCoordinator:
    controller = MemoryGenerationTaskController(bus=bus)
    coordinator = MemoryGenerationCoordinator(bus=bus)
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
        controller.submit_generation,
    )
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
        controller.submit_generation_many,
    )
    return coordinator


async def _capture_event(target: list, **kwargs) -> None:
    target.append(kwargs)


@pytest.mark.asyncio
async def test_passive_settlement_routes_archive_spec_through_task_controller():
    bus = PatchouliBus()
    coordinator = _wire_generation_pipeline(bus)
    execute_spec = AsyncMock(return_value=[])
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)

    memory_task = await coordinator.submit_settlement(
        TopicMaterializeTask(
            topic_id="topic_1",
            topic_title="topic title",
            topic_summary="topic summary",
            blocks=[
                LogicalBlock(
                    turn=TurnRecord(
                        user_query="question",
                        assistant_final_text="answer",
                    )
                )
            ],
            state_summary="state summary",
        )
    )
    await memory_task._bg_task

    assert memory_task.status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.ARCHIVE
    assert spec.source.creation_artifact_intent == "ARCHIVE"
    assert spec.source.provenance_intent == "ARCHIVE"
    assert spec.interaction_input.topic_id == "topic_1"
    assert spec.request.context.state_summary == "state summary"


@pytest.mark.asyncio
async def test_active_write_routes_to_generation_and_publishes_settlement():
    bus = PatchouliBus()
    coordinator = _wire_generation_pipeline(bus)
    published = []
    execute_spec = AsyncMock(return_value=_settlement_result("draft_write"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    memory_tasks = await coordinator.submit_active([_write_task("draft_write")], "topic_1")
    await memory_tasks[0]._bg_task

    assert memory_tasks[0].status == MemoryGenerationTaskStatus.COMPLETED
    assert memory_tasks[0].canonical_alias == "memory_alias"
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.WRITE
    assert spec.pending_alias == "draft_write"
    assert spec.intent_id == "intent_draft_write"
    assert spec.request.is_write is True
    assert published[0]["settlement"].pending_alias == "draft_write"


@pytest.mark.asyncio
async def test_active_update_fetches_existing_memory_before_generation():
    bus = PatchouliBus()
    coordinator = _wire_generation_pipeline(bus)
    existing = _memory_atom()
    memory_get = AsyncMock(return_value=existing)
    execute_spec = AsyncMock(return_value=_settlement_result("draft_update"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.register(PatchouliLocalRoutes.MEMORY_GET, memory_get)

    memory_tasks = await coordinator.submit_active(
        [_update_task(str(existing.id), "draft_update")],
        "topic_1",
    )
    await memory_tasks[0]._bg_task

    memory_get.assert_awaited_once_with(existing.id)
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.UPDATE
    assert spec.pending_alias == "draft_update"
    assert spec.request.is_update is True
    assert spec.request.existing_memory is existing


@pytest.mark.asyncio
async def test_active_batch_skips_missing_update_and_runs_valid_write():
    bus = PatchouliBus()
    coordinator = _wire_generation_pipeline(bus)
    failed = []
    execute_spec = AsyncMock(return_value=_settlement_result("draft_write"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.register(PatchouliLocalRoutes.MEMORY_GET, AsyncMock(return_value=None))
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_FAILED,
        lambda **kwargs: _capture_event(failed, **kwargs),
    )

    memory_tasks = await coordinator.submit_active(
        [
            _write_task("draft_write"),
            _update_task(str(uuid4()), "draft_update"),
        ],
        "topic_1",
    )
    await memory_tasks[0]._bg_task

    assert len(memory_tasks) == 1
    assert memory_tasks[0].status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.WRITE
    assert spec.pending_alias == "draft_write"
    assert failed == [{"pending_alias": "draft_update"}]
