from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from hivememory.core.models import (
    Identity,
    IndexLayer,
    LogicalBlock,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    TurnRecord,
    WorkspaceMemoryKey,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.generation.models import DuplicateDecision, GenerationOutcome
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.memory_generation.controller import (
    MemoryGenerationTaskController,
)
from hivememory.patchouli.control.memory_generation.coordinator import (
    MemoryGenerationCoordinator,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskStatus,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.ports import MidTermStoragePort
from hivememory.patchouli.memory_library.stores import (
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from tests.helpers.memory import make_memory_creation_context, make_memory_metadata
from tests.helpers.workspace import make_access_context


def _creation_context():
    return make_memory_creation_context(user_id="u1", agent_id="omni_doll")


def _access_context():
    return make_access_context(user_id="u1", agent_id="omni_doll")


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
        meta=make_memory_metadata(source_agent_id="agent-1", user_id="u1"),
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
        identity_scope=_creation_context(),
        focus=WriteFocus(content="remember this"),
    )


def _update_task(base_uuid: str, alias="draft_update") -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        identity_scope=_creation_context(),
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
            canonical_alias="memory_alias",
            settlement=settlement,
        )
    ]


def _wire_generation_pipeline(
    bus: PatchouliBus,
) -> tuple[MemoryGenerationCoordinator, MemoryGenerationTaskController]:
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
    return coordinator, controller


async def _capture_event(target: list, **kwargs) -> None:
    target.append(kwargs)


@pytest.mark.asyncio
async def test_passive_settlement_routes_archive_spec_through_task_controller():
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
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
            identity_scope=_creation_context(),
        )
    )
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.ARCHIVE
    assert spec.source.creation_artifact_intent == "ARCHIVE"
    assert spec.interaction_input.topic_id == "topic_1"
    assert spec.request.context.state_summary == "state summary"


@pytest.mark.asyncio
async def test_active_write_routes_to_generation_and_publishes_settlement():
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    published = []
    execute_spec = AsyncMock(return_value=_settlement_result("draft_write"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    memory_tasks = await coordinator.submit_active(
        [_write_task("draft_write")],
        "topic_1",
        access_context=_access_context(),
    )
    await controller.wait_task(memory_tasks[0].task_id)
    completed = await controller.get_task(memory_tasks[0].task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.WRITE
    assert spec.pending_alias == "draft_write"
    assert spec.intent_id == "intent_draft_write"
    assert spec.request.is_write is True
    assert published[0]["settlement"].pending_alias == "draft_write"


@pytest.mark.asyncio
async def test_active_update_fetches_existing_memory_before_generation():
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    existing = _memory_atom()
    memory_get = AsyncMock(return_value=existing)
    execute_spec = AsyncMock(return_value=_settlement_result("draft_update"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.register(PatchouliLocalRoutes.MEMORY_GET, memory_get)

    memory_tasks = await coordinator.submit_active(
        [_update_task(str(existing.id), "draft_update")],
        "topic_1",
        access_context=_access_context(),
    )
    await controller.wait_task(memory_tasks[0].task_id)

    memory_get.assert_awaited_once_with(
        existing.id,
        access_context=_access_context(),
    )
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.UPDATE
    assert spec.pending_alias == "draft_update"
    assert spec.request.is_update is True
    assert spec.request.existing_memory is not existing
    assert (
        spec.request.existing_memory.model_dump(mode="json")
        == existing.model_dump(mode="json")
    )


@pytest.mark.asyncio
async def test_active_batch_skips_missing_update_and_runs_valid_write():
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
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
        access_context=_access_context(),
    )
    await controller.wait_task(memory_tasks[0].task_id)
    completed = await controller.get_task(memory_tasks[0].task_id)

    assert len(memory_tasks) == 1
    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.WRITE
    assert spec.pending_alias == "draft_write"
    assert failed == [{"pending_alias": "draft_update"}]


# ========== 真实 Familiar + stub engine + 真实 mid_term（数据面补档） ==========
# 将 GENERATION_EXECUTE_SPEC 从 AsyncMock 升级为真实 MemoryGenerationFamiliar，
# 验证「coordinator → controller → familiar → 真实 mid_term 落库」完整数据面：
# 控制流断言保留在旧用例，此处聚焦「生成结果真实落库 + settlement 发布」。


class _InMemoryMidTermPort(MidTermStoragePort):
    """真实 MidTermMemoryStore 的内存后端，验证生成结果真实写入中期存储。"""

    def __init__(self) -> None:
        self.memories: dict[tuple[str, str, UUID], MemoryAtom] = {}

    @staticmethod
    def _key(memory: MemoryAtom) -> tuple[str, str, UUID]:
        workspace = memory.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory.id

    @staticmethod
    def _scope_key(scope, memory_id: UUID) -> tuple[str, str, UUID]:
        workspace = scope.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, memory_id

    async def upsert(self, memory: MemoryAtom) -> None:
        self.memories[self._key(memory)] = memory

    async def get(self, scope, memory_id: UUID) -> MemoryAtom | None:
        return self.memories.get(self._scope_key(scope, memory_id))

    async def get_by_alias(self, scope, alias: str) -> MemoryAtom | None:
        for memory in self.memories.values():
            if (
                memory.workspace_identity == scope.workspace_identity
                and memory.index.alias == alias
            ):
                return memory
        return None

    async def get_for_mutation(self, access_context, memory_id: UUID) -> MemoryAtom | None:
        return await self.get(access_context, memory_id)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        workspace = key.workspace_identity
        return self.memories.get(
            (workspace.owner_user_id, workspace.workspace_id, key.memory_id)
        )

    async def update_access_info(self, access_context, memory_id: UUID) -> None:
        memory = await self.get(access_context, memory_id)
        if memory is not None:
            memory.meta.access_count += 1

    async def delete(self, access_context, memory_id: UUID) -> bool:
        return self.memories.pop(self._scope_key(access_context, memory_id), None) is not None

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        workspace = key.workspace_identity
        storage_key = (workspace.owner_user_id, workspace.workspace_id, key.memory_id)
        return self.memories.pop(storage_key, None) is not None

    async def batch_delete(self, access_context, ids: list[UUID]) -> int:
        return sum(1 for mid in ids if await self.delete(access_context, mid))

    async def search(
        self,
        scope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ]

    async def scroll(self, scope, filters=None, limit: int = 100) -> list[MemoryAtom]:
        return [
            memory
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ][:limit]

    async def count(self, scope, filters=None) -> int:
        return len(await self.scroll(scope, filters=filters))

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return list(self.memories.values())[:limit]


class _StubGenerationEngine:
    """真实接口的 stub：process 返回预设 GenerationOutcome，不触达 LLM。"""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self.requests: list = []

    async def process(self, request, *, identity_scope=None):
        self.requests.append(request)
        return self._outcomes


@pytest.fixture
def memory_library(tmp_path) -> MemoryLibrary:
    """真实 MemoryLibrary：真实 MidTermMemoryStore + 内存后端，长期存储用临时目录。"""
    mid_term = MidTermMemoryStore(primary=_InMemoryMidTermPort())
    short_term = ShortTermMemoryStore()
    long_term = LongTermMemoryStore(
        FileBasedStorageAdapter(
            archive_dir=str(tmp_path / "archive"),
            compress=False,
        )
    )
    return MemoryLibrary(
        short_term=short_term,
        mid_term=mid_term,
        long_term=long_term,
    )


def _wire_generation_familiar(
    bus: PatchouliBus,
    memory_library: MemoryLibrary,
    stub_engine: _StubGenerationEngine,
) -> MemoryGenerationFamiliar:
    familiar = MemoryGenerationFamiliar(
        generation_engine=stub_engine,  # type: ignore[arg-type]
        memory_library=memory_library,
    )
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, familiar.execute)
    return familiar


@pytest.mark.asyncio
async def test_passive_settlement_lands_in_real_mid_term(memory_library):
    """被动 SETTLEMENT：真实 Familiar 把生成结果写入真实 mid_term。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()

    atom = _memory_atom()
    stub = _StubGenerationEngine(
        [GenerationOutcome(atom=atom, duplicate_decision=DuplicateDecision.CREATE)]
    )
    _wire_generation_familiar(bus, memory_library, stub)

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
            identity_scope=_creation_context(),
        )
    )
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：stub engine 被真实 familiar 调用，且结果真实落库 mid_term
    assert stub.requests, "真实 familiar 应调用 stub generation engine"
    stored = await memory_library.mid_term.get(_creation_context(), atom.id)
    assert stored is not None, "生成结果应写入真实 mid_term"
    assert stored.payload.content == "content"


@pytest.mark.asyncio
async def test_active_write_lands_in_real_mid_term(memory_library):
    """主动 WRITE：真实 Familiar 执行后落库 mid_term 并发布 settlement。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    published = []

    atom = _memory_atom()
    stub = _StubGenerationEngine(
        [GenerationOutcome(atom=atom, duplicate_decision=DuplicateDecision.CREATE)]
    )
    _wire_generation_familiar(bus, memory_library, stub)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    memory_tasks = await coordinator.submit_active(
        [_write_task("draft_write")],
        "topic_1",
        access_context=_access_context(),
    )
    await controller.wait_task(memory_tasks[0].task_id)
    completed = await controller.get_task(memory_tasks[0].task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：真实落库 mid_term
    stored = await memory_library.mid_term.get(_access_context(), atom.id)
    assert stored is not None, "WRITE 生成结果应写入真实 mid_term"
    # 控制面：settlement 由熟悉发布，pending atom 链路闭环
    assert published, "应发布 PENDING_ATOM_SETTLED 事件"
    assert published[0]["settlement"].pending_alias == "draft_write"
    assert published[0]["settlement"].canonical_alias == "memory_alias"
