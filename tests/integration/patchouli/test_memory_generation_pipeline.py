from collections.abc import Mapping
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from hivememory.core.models import (
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
from hivememory.engines.artifacts.engine import ArtifactEngine
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
from hivememory.patchouli.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.ports import MidTermStoragePort
from hivememory.patchouli.memory_library.stores import (
    ArtifactStore,
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.system.config.patchouli import ArtifactConfig
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope


def _identity_scope():
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


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
        identity_scope=_identity_scope(),
        focus=WriteFocus(content="remember this"),
    )


def _update_task(base_uuid: str, alias="draft_update") -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        identity_scope=_identity_scope(),
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
async def test_passive_settlement_routes_settle_spec_through_task_controller():
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
            identity_scope=_identity_scope(),
        )
    )
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.SETTLE
    assert spec.source.creation_artifact_intent == "SYSTEM"
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
        identity_scope=_identity_scope(),
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
        identity_scope=_identity_scope(),
    )
    await controller.wait_task(memory_tasks[0].task_id)

    memory_get.assert_awaited_once_with(
        existing.id,
        identity_scope=_identity_scope(),
    )
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.UPDATE
    assert spec.pending_alias == "draft_update"
    assert spec.request.is_update is True
    assert spec.request.existing_memory is not existing
    assert spec.request.existing_memory.model_dump(mode="json") == existing.model_dump(mode="json")


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
        identity_scope=_identity_scope(),
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

    @staticmethod
    def _key_of(key: WorkspaceMemoryKey) -> tuple[str, str, UUID]:
        workspace = key.workspace_identity
        return workspace.owner_user_id, workspace.workspace_id, key.memory_id

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        self.memories[self._key(memory)] = memory

    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        memory = self.memories.get(self._key_of(key))
        if memory is None:
            return None
        for dotted_path, value in patch.items():
            parts = dotted_path.split(".")
            target: Any = memory
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        return memory

    async def get(
        self,
        scope,
        memory_id: UUID,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return self.memories.get(self._scope_key(scope, memory_id))

    async def get_by_alias(
        self,
        scope,
        alias: str,
        *,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        for memory in self.memories.values():
            if (
                memory.workspace_identity == scope.workspace_identity
                and memory.index.alias == alias
            ):
                return memory
        return None

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return self.memories.get(self._key_of(key))

    async def delete(self, identity_scope, memory_id: UUID) -> bool:
        return self.memories.pop(self._scope_key(identity_scope, memory_id), None) is not None

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        return self.memories.pop(self._key_of(key), None) is not None

    async def search(
        self,
        scope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        enforce_actor_visibility: bool = True,
    ):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ]

    async def scroll(
        self,
        scope,
        filters=None,
        limit: int = 100,
        *,
        enforce_actor_visibility: bool = True,
    ) -> list[MemoryAtom]:
        return [
            memory
            for memory in self.memories.values()
            if memory.workspace_identity == scope.workspace_identity
        ][:limit]

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return list(self.memories.values())[:limit]


class _StubGenerationEngine:
    """真实接口的 stub：process 返回预设 GenerationOutcome，不触达 LLM。

    MVL-2 引擎纯化后 Familiar 会把提交边界时点作为 ``now`` 传入；stub 与真实
    引擎保持同签名，时间戳断言由落库结果承载。
    """

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self.requests: list = []

    async def process(self, request, *, identity_scope=None, now: datetime | None = None):
        self.requests.append(request)
        return self._outcomes


@pytest.fixture
def artifact_engine(tmp_path) -> ArtifactEngine:
    """真实 ArtifactEngine：版本记录写入 tmp_path 下的文件系统 Artifact 仓库。

    MVL-2 起版本记录是内容提交成功的前置条件（NoOp memory builder 会被
    Familiar 以 RuntimeError 拒绝），数据面用例必须装配会真实写版本的 builder。
    """
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    return ArtifactEngine.from_store(store, ArtifactConfig(enabled=True))


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
    artifact_engine: ArtifactEngine,
) -> MemoryGenerationFamiliar:
    familiar = MemoryGenerationFamiliar(
        generation_engine=stub_engine,  # type: ignore[arg-type]
        memory_library=memory_library,
        artifact_engine=artifact_engine,
    )
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, familiar.execute)
    return familiar


@pytest.mark.asyncio
async def test_passive_settlement_lands_in_real_mid_term(memory_library, artifact_engine):
    """被动 SETTLEMENT：真实 Familiar 把生成结果写入真实 mid_term。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()

    atom = _memory_atom()
    stub = _StubGenerationEngine(
        [GenerationOutcome(atom=atom, duplicate_decision=DuplicateDecision.CREATE)]
    )
    _wire_generation_familiar(bus, memory_library, stub, artifact_engine)

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
            identity_scope=_identity_scope(),
        )
    )
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：stub engine 被真实 familiar 调用，且结果真实落库 mid_term
    assert stub.requests, "真实 familiar 应调用 stub generation engine"
    stored = await memory_library.mid_term.get(_identity_scope(), atom.id)
    assert stored is not None, "生成结果应写入真实 mid_term"
    assert stored.payload.content == "content"


@pytest.mark.asyncio
async def test_active_write_lands_in_real_mid_term(memory_library, artifact_engine):
    """主动 WRITE：真实 Familiar 执行后落库 mid_term 并发布 settlement。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    published = []

    atom = _memory_atom()
    stub = _StubGenerationEngine(
        [GenerationOutcome(atom=atom, duplicate_decision=DuplicateDecision.CREATE)]
    )
    _wire_generation_familiar(bus, memory_library, stub, artifact_engine)
    bus.register(PatchouliLocalRoutes.TOPIC_GET, AsyncMock(return_value=_TopicData()))
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    memory_tasks = await coordinator.submit_active(
        [_write_task("draft_write")],
        "topic_1",
        identity_scope=_identity_scope(),
    )
    await controller.wait_task(memory_tasks[0].task_id)
    completed = await controller.get_task(memory_tasks[0].task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：真实落库 mid_term
    stored = await memory_library.mid_term.get(_identity_scope(), atom.id)
    assert stored is not None, "WRITE 生成结果应写入真实 mid_term"
    # 控制面：settlement 由熟悉发布，pending atom 链路闭环
    assert published, "应发布 PENDING_ATOM_SETTLED 事件"
    assert published[0]["settlement"].pending_alias == "draft_write"
    assert published[0]["settlement"].canonical_alias == "memory_alias"
