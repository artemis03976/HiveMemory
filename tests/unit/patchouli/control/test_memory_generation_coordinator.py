from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    TurnRecord,
)
from hivememory.core.models.pending import (
    PendingAtomMaterializeTask,
    UpdateFocus,
    WriteFocus,
)
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)


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


def _write_task(alias="draft_001"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        identity=Identity(user_id="test_user"),
        focus=WriteFocus(content="test content"),
    )


def _update_task(base_uuid: str, alias="draft_update"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        identity=Identity(user_id="test_user"),
        focus=UpdateFocus(
            instruction="merge update",
            content="new content",
            base_uuid=base_uuid,
            base_alias="memory_alias",
        ),
    )


def _topic_block():
    return LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))


def _memory_atom(memory_id) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id,
        meta=MetaData(source_agent_id="agent-1", user_id="test_user"),
        index=IndexLayer(
            title="memory title",
            summary="summary text",
            tags=[],
            memory_type=MemoryType.FACT,
            alias="memory_alias",
        ),
        payload=PayloadLayer(content="content"),
    )


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
            blocks=[_topic_block()],
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
    async def test_submit_settlement_skips_empty_generation_context(self):
        bus = Mock()
        bus.request = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)
        payload = TopicMaterializeTask(
            topic_id="empty",
            topic_title="empty",
            blocks=[],
            state_summary="state",
        )

        task = await coordinator.submit_settlement(payload)

        assert task is None
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_active_empty_tasks_returns_without_bus_request(self):
        bus = Mock()
        bus.request = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active([], "t1")

        assert result == []
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_active_builds_write_specs_from_topic_context(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_task_handle(source=MemoryGenerationSource.WRITE)]
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
        assert spec.interaction_input.topic_title == "title"

    @pytest.mark.asyncio
    async def test_submit_active_update_fetches_existing_memory(self):
        memory_id = uuid4()
        existing = _memory_atom(memory_id)
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return existing
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_task_handle(source=MemoryGenerationSource.UPDATE)]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active(
            [_update_task(str(memory_id), "draft_update")],
            "t1",
        )

        assert len(result) == 1
        memory_get_call = bus.request.await_args_list[1]
        assert memory_get_call.args == (PatchouliLocalRoutes.MEMORY_GET, memory_id)
        spec = bus.request.await_args_list[-1].args[1][0]
        assert spec.source == MemoryGenerationSource.UPDATE
        assert spec.request.is_update is True
        assert spec.request.existing_memory is existing

    @pytest.mark.asyncio
    async def test_submit_active_update_target_missing_marks_pending_failed(self):
        memory_id = uuid4()
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return None
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active(
            [_update_task(str(memory_id), "draft_update")],
            "t1",
        )

        assert result == []
        requested_routes = [call.args[0] for call in bus.request.await_args_list]
        assert PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY not in requested_routes
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_update",
        )

    @pytest.mark.asyncio
    async def test_submit_active_skips_failed_update_but_submits_write(self):
        missing_id = uuid4()
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return None
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_task_handle(source=MemoryGenerationSource.WRITE)]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active(
            [
                _write_task("draft_write"),
                _update_task(str(missing_id), "draft_update"),
            ],
            "t1",
        )

        assert len(result) == 1
        specs = bus.request.await_args_list[-1].args[1]
        assert len(specs) == 1
        assert specs[0].source == MemoryGenerationSource.WRITE
        assert specs[0].pending_alias == "draft_write"
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_update",
        )

    @pytest.mark.asyncio
    async def test_submit_active_isolates_unexpected_spec_build_failure(self):
        memory_id = uuid4()
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                raise RuntimeError("memory store unavailable")
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_task_handle(source=MemoryGenerationSource.WRITE)]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active(
            [
                _write_task("draft_write"),
                _update_task(str(memory_id), "draft_update"),
            ],
            "t1",
        )

        assert len(result) == 1
        specs = bus.request.await_args_list[-1].args[1]
        assert [spec.pending_alias for spec in specs] == ["draft_write"]
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_update",
        )

    @pytest.mark.asyncio
    async def test_submit_active_invalid_update_uuid_marks_pending_failed(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus)

        result = await coordinator.submit_active(
            [_update_task("not-a-uuid", "draft_update")],
            "t1",
        )

        assert result == []
        requested_routes = [call.args[0] for call in bus.request.await_args_list]
        assert PatchouliLocalRoutes.MEMORY_GET not in requested_routes
        assert PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY not in requested_routes
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_update",
        )
