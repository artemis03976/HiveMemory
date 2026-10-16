from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    TopicData,
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
from hivememory.patchouli.contracts.memory_tasks import (
    MemoryGenerationAdmission,
    MemoryGenerationAdmissionStatus,
)
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationSource,
    MemoryGenerationTask,
)
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _identity_scope():
    return make_identity_scope(user_id="test_user", agent_id="test_agent")


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
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
    )


def _write_task(alias="draft_001"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="WRITE",
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
        focus=WriteFocus(content="test content"),
    )


def _update_task(base_uuid: str, alias="draft_update"):
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
        focus=UpdateFocus(
            instruction="merge update",
            content="new content",
            base_uuid=base_uuid,
            base_alias="memory_alias",
        ),
    )


def _topic_block(query: str = "q"):
    return LogicalBlock(turn=TurnRecord(user_query=query, assistant_final_text="a"))


def _accepted(task_id="j1", source=MemoryGenerationSource.WRITE):
    return MemoryGenerationAdmission.accepted(_task_handle(task_id=task_id, source=source))


def _statuses(admissions):
    return [admission.status for admission in admissions]


ACCEPTED = MemoryGenerationAdmissionStatus.ACCEPTED
REJECTED = MemoryGenerationAdmissionStatus.REJECTED
UNKNOWN = MemoryGenerationAdmissionStatus.UNKNOWN


def _memory_atom(memory_id) -> MemoryAtom:
    return MemoryAtom(
        id=memory_id,
        meta=make_memory_metadata(source_agent_id="agent-1", user_id="test_user"),
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
    async def test_submit_active_delegates_specs_to_batch_entry(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                specs = args[0]
                assert [spec.pending_alias for spec in specs] == [
                    "draft_ok",
                    "draft_rejected",
                ]
                return [_accepted("accepted"), MemoryGenerationAdmission.rejected()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_write_task("draft_ok"), _write_task("draft_rejected")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [ACCEPTED, REJECTED]
        assert result[0].task.task_id == "accepted"
        # 批量入口逐项结算；协调层不重复发布失败事件。
        bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_active_keeps_ambiguous_admission_pending(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                raise ConnectionError("admission response lost")
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_write_task("draft_unknown")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [UNKNOWN]
        bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_settlement_builds_settle_spec(self):
        bus = Mock()
        bus.request = AsyncMock(return_value=_task_handle())
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)
        payload = TopicMaterializeTask(
            topic_id="t1",
            topic_title="title",
            topic_summary="summary",
            blocks=[_topic_block()],
            state_summary="state",
            belong_to=_identity_scope().workspace_identity,
        )

        task = await coordinator.submit_settlement(payload)

        assert isinstance(task, MemoryGenerationTask)
        route, spec = bus.request.await_args.args
        assert route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION
        assert spec.topic_id == "t1"
        assert spec.source == MemoryGenerationSource.SETTLE
        # 结算材料不携带访问者：manual/idle/LRU/shutdown 都以归属对应的 system 发起，
        # 最后访问者或驱逐触发者不能成为结算发起者。
        assert spec.belong_to == _identity_scope().workspace_identity
        assert spec.from_actor == ActorIdentity(
            user_id="test_user", agent_id=SYSTEM_AGENT_ID, team_id=None
        )
        assert spec.request.context.state_summary == "state"
        assert spec.interaction_input.topic_title == "title"

    @pytest.mark.asyncio
    async def test_submit_settlement_skips_empty_generation_context(self):
        bus = Mock()
        bus.request = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)
        payload = TopicMaterializeTask(
            topic_id="empty",
            topic_title="empty",
            blocks=[],
            state_summary="state",
            belong_to=_identity_scope().workspace_identity,
        )

        task = await coordinator.submit_settlement(payload)

        assert task is None
        bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_active_empty_tasks_returns_without_bus_request(self):
        bus = Mock()
        bus.request = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

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

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_write_task("draft_1")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [ACCEPTED]
        spec = bus.request.await_args_list[-1].args[1][0]
        assert spec.source == MemoryGenerationSource.WRITE
        assert spec.pending_alias == "draft_1"
        assert spec.request.is_write is True
        assert spec.from_actor.user_id == "test_user"
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

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return existing
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted(source=MemoryGenerationSource.UPDATE)]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_update_task(str(memory_id), "draft_update")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [ACCEPTED]
        memory_get_call = bus.request.await_args_list[1]
        assert memory_get_call.args == (PatchouliLocalRoutes.MEMORY_GET, memory_id)
        assert memory_get_call.kwargs == {
            "belong_to": _identity_scope().workspace_identity,
            "from_actor": _identity_scope().actor_identity,
        }
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

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return None
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_update_task(str(memory_id), "draft_update")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [REJECTED]
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

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                return None
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [
                _write_task("draft_write"),
                _update_task(str(missing_id), "draft_update"),
            ],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [ACCEPTED, REJECTED]
        spec = bus.request.await_args_list[-1].args[1][0]
        assert spec.source == MemoryGenerationSource.WRITE
        assert spec.pending_alias == "draft_write"
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

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_GET:
                raise RuntimeError("memory store unavailable")
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [
                _write_task("draft_write"),
                _update_task(str(memory_id), "draft_update"),
            ],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [ACCEPTED, UNKNOWN]
        spec = bus.request.await_args_list[-1].args[1][0]
        assert spec.pending_alias == "draft_write"
        bus.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_active_invalid_update_uuid_marks_pending_failed(self):
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_update_task("not-a-uuid", "draft_update")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [REJECTED]
        requested_routes = [call.args[0] for call in bus.request.await_args_list]
        assert PatchouliLocalRoutes.MEMORY_GET not in requested_routes
        assert PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY not in requested_routes
        bus.publish.assert_awaited_once_with(
            PatchouliLocalEvents.PENDING_ATOM_FAILED,
            pending_alias="draft_update",
        )

    @pytest.mark.asyncio
    async def test_submit_active_uses_configured_number_of_recent_blocks(self):
        """材料只取话题最近 recent_block_limit 个 block，并带上 state_summary。"""
        topic = TopicData(
            topic_id="t1",
            workspace_identity=_identity_scope().workspace_identity,
            topic_title="title",
            state_summary="state",
            blocks=tuple(_topic_block(f"q{index}") for index in range(5)),
            last_update=0.0,
        )
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=2)

        await coordinator.submit_active(
            [_write_task("draft_1")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        spec = bus.request.await_args_list[-1].args[1][0]
        assert [turn.user_query for turn in spec.request.context.turns] == ["q3", "q4"]
        assert spec.request.context.state_summary == "state"
        assert [block.turn.user_query for block in spec.interaction_input.blocks] == ["q3", "q4"]

    @pytest.mark.asyncio
    async def test_submit_active_rejects_all_intents_when_topic_read_fails(self):
        """读取材料失败时没有意图进入队列：逐项结算为失败并返回拒绝。"""
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                raise RuntimeError("topic store unavailable")
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_write_task("draft_a"), _write_task("draft_b")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [REJECTED, REJECTED]
        failed = [call.kwargs["pending_alias"] for call in bus.publish.await_args_list]
        assert failed == ["draft_a", "draft_b"]

    @pytest.mark.asyncio
    async def test_submit_active_treats_invalid_batch_response_as_unknown(self):
        """批量响应数量与输入不符时无法判断接纳情况，逐项按结果未知处理。"""
        topic_data = Mock()
        topic_data.recent_blocks.return_value = [_topic_block()]
        topic_data.state_summary = "state"
        topic_data.topic_title = "title"
        topic_data.topic_summary = "summary"
        bus = Mock()

        async def request(route, *args, **kwargs):
            if route == PatchouliLocalRoutes.TOPIC_GET:
                return topic_data
            if route == PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY:
                return [_accepted()]
            raise AssertionError(route)

        bus.request = AsyncMock(side_effect=request)
        bus.publish = AsyncMock()
        coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)

        result = await coordinator.submit_active(
            [_write_task("draft_a"), _write_task("draft_b")],
            "t1",
            belong_to=_identity_scope().workspace_identity,
        )

        assert _statuses(result) == [UNKNOWN, UNKNOWN]
        bus.publish.assert_not_awaited()

    def test_negative_recent_block_limit_is_rejected(self):
        with pytest.raises(ValueError, match="recent_block_limit"):
            MemoryGenerationCoordinator(bus=Mock(), recent_block_limit=-1)
