import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from hivememory.config.patchouli import (
    ArtifactConfig,
    DeduplicatorConfig,
    ExtractorConfig,
    LibrarianConfig,
    MemoryPerceptionConfig,
    PerceptionEngineConfig,
)
from hivememory.core.models import (
    ArtifactType,
    CitedSegment,
    IndexLayer,
    InteractionArtifact,
    MemoryAtom,
    MemoryType,
    MemoryVersionArtifact,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    TopicData,
    TopicFold,
    TurnEvent,
    WorkspaceMemoryKey,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.librarian import Librarian
from hivememory.engines.generation.models import (
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationOutcome,
)
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.infrastructure.llm.base import LLMRequestParams
from hivememory.memory_library.adapters.artifact import (
    FilesystemArtifactStorageAdapter,
)
from hivememory.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.memory_library.agent_profiles import AgentProfileStore
from hivememory.memory_library.library import MemoryLibrary
from hivememory.memory_library.ports import MidTermStoragePort
from hivememory.memory_library.stores import (
    ArtifactStore,
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
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
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.topic_pool import restore_topic_pool, save_topic_pool
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.generation_bus import register_topic_routes
from tests.helpers.llm import ScriptedLLMService
from tests.helpers.memory import make_memory_metadata
from tests.helpers.sessions import make_interaction_payload, make_logical_block
from tests.helpers.workspace import make_identity_scope


def _identity_scope():
    return make_identity_scope(user_id="u1", agent_id="omni_doll")


def _topic_store() -> ShortTermMemoryStore:
    """生成读取材料与写入记录的话题池：``topic_1`` 接收过一轮交互。"""
    store = ShortTermMemoryStore()
    store.put(
        TopicData(
            topic_id="topic_1",
            workspace_identity=_identity_scope().workspace_identity,
            topic_title="topic title",
            topic_summary="topic summary",
            interactions=(make_logical_block("question", "answer"),),
            last_update=1.0,
        )
    )
    return store


def _settlement_task(*, retires_topic: bool = True) -> TopicMaterializeTask:
    """折叠过一次的话题结算：工作视图只剩最近一轮，状态摘要来自折叠。"""
    folded = make_logical_block("earlier question", "earlier answer")
    return TopicMaterializeTask(
        topic_id="topic_1",
        topic_title="topic title",
        topic_summary="topic summary",
        interactions=(folded, make_logical_block("question", "answer")),
        folds=(
            TopicFold(folded_block_ids=(folded.block_id,), summary="state summary", folded_at=1.0),
        ),
        belong_to=_identity_scope().workspace_identity,
        retires_topic=retires_topic,
    )


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
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
        focus=WriteFocus(content="remember this"),
    )


def _update_task(base_uuid: str, alias="draft_update") -> PendingAtomMaterializeTask:
    return PendingAtomMaterializeTask(
        pending_alias=alias,
        intent_id=f"intent_{alias}",
        source_verb="UPDATE",
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
        focus=UpdateFocus(
            instruction="merge this",
            content="new content",
            base_uuid=base_uuid,
            base_alias="memory_alias",
            base_version=1,
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
            action=GenerationAction.CREATE,
            canonical_alias="memory_alias",
            settlement=settlement,
        )
    ]


def _wire_generation_pipeline(
    bus: PatchouliBus, topics: ShortTermMemoryStore | None = None
) -> tuple[MemoryGenerationCoordinator, MemoryGenerationTaskController]:
    controller = MemoryGenerationTaskController(bus=bus)
    coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)
    register_topic_routes(bus, topics if topics is not None else _topic_store())
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
        controller.submit_generation,
    )
    bus.register(
        PatchouliLocalRoutes.MEMORY_TASK_ADMIT_GENERATION,
        controller.admit_generation,
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

    memory_task = await coordinator.submit_settlement(_settlement_task())
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.SETTLE
    assert spec.source.creation_artifact_intent == "SYSTEM"
    assert spec.interaction_input.topic_id == "topic_1"
    assert spec.request.context.fold_summary == "state summary"


@pytest.mark.asyncio
async def test_active_write_routes_to_generation_and_publishes_settlement():
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    published = []
    execute_spec = AsyncMock(return_value=_settlement_result("draft_write"))
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, execute_spec)
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    admission = await coordinator.submit_active(
        _write_task("draft_write"),
        "topic_1",
        belong_to=_identity_scope().workspace_identity,
    )
    await controller.wait_task(admission.task.task_id)
    completed = await controller.get_task(admission.task.task_id)

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
    bus.register(PatchouliLocalRoutes.MEMORY_GET, memory_get)

    admission = await coordinator.submit_active(
        _update_task(str(existing.id), "draft_update"),
        "topic_1",
        belong_to=_identity_scope().workspace_identity,
    )
    await controller.wait_task(admission.task.task_id)

    memory_get.assert_awaited_once_with(
        existing.id,
        belong_to=_identity_scope().workspace_identity,
        from_actor=_identity_scope().actor_identity,
    )
    spec = execute_spec.await_args.args[0]
    assert spec.source == MemoryGenerationSource.UPDATE
    assert spec.pending_alias == "draft_update"
    assert spec.request.is_update is True
    assert spec.request.existing_memory is not existing
    assert spec.request.existing_memory.model_dump(mode="json") == existing.model_dump(mode="json")


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
        workspace = scope
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
        from_actor,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return self.memories.get(self._scope_key(scope, memory_id))

    async def list_alias_holders(
        self,
        workspace_identity,
        alias: str,
        *,
        limit: int,
    ) -> list[UUID]:
        holders = [
            memory.id
            for memory in self.memories.values()
            if memory.workspace_identity == workspace_identity and memory.index.alias == alias
        ]
        return holders[:limit]

    async def get_by_alias(
        self,
        scope,
        alias: str,
        *,
        from_actor,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        for memory in self.memories.values():
            if memory.workspace_identity == scope and memory.index.alias == alias:
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
        from_actor,
        enforce_actor_visibility: bool = True,
    ):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope
        ]

    async def scroll(
        self,
        scope,
        filters=None,
        limit: int = 100,
        *,
        cursor=None,
        from_actor,
        enforce_actor_visibility: bool = True,
    ) -> tuple[list[MemoryAtom], str | None]:
        in_scope = [m for m in self.memories.values() if m.workspace_identity == scope]
        return in_scope[:limit], None

    async def count(
        self,
        scope,
        filters=None,
        *,
        from_actor,
        enforce_actor_visibility: bool = True,
    ) -> int:
        return sum(1 for m in self.memories.values() if m.workspace_identity == scope)

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return list(self.memories.values())[:limit]


class _StubGenerationEngine:
    """真实接口的 stub：process 返回预设 GenerationOutcome，不触达 LLM。

    Familiar 会把提交边界时点作为 ``now`` 传入；stub 与真实
    引擎保持同签名，时间戳断言由落库结果承载。
    """

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self.requests: list = []

    async def process(self, request, *, belong_to, from_actor, now: datetime | None = None):
        self.requests.append(request)
        return self._outcomes


@pytest.fixture
def artifact_engine(tmp_path) -> ArtifactEngine:
    """真实 ArtifactEngine：版本记录写入 tmp_path 下的文件系统 Artifact 仓库。

    版本记录是内容提交成功的前置条件（NoOp memory builder 会被
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
        agent_profiles=AgentProfileStore(tmp_path / "agent_profiles"),
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
    stub = _StubGenerationEngine([GenerationOutcome(atom=atom, action=GenerationAction.CREATE)])
    _wire_generation_familiar(bus, memory_library, stub, artifact_engine)

    memory_task = await coordinator.submit_settlement(_settlement_task())
    await controller.wait_task(memory_task.task_id)
    completed = await controller.get_task(memory_task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：stub engine 被真实 familiar 调用，且结果真实落库 mid_term
    assert stub.requests, "真实 familiar 应调用 stub generation engine"
    stored = await memory_library.mid_term.get(
        _identity_scope().workspace_identity, atom.id, from_actor=_identity_scope().actor_identity
    )
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
    stub = _StubGenerationEngine([GenerationOutcome(atom=atom, action=GenerationAction.CREATE)])
    _wire_generation_familiar(bus, memory_library, stub, artifact_engine)
    bus.subscribe(
        PatchouliLocalEvents.PENDING_ATOM_SETTLED,
        lambda **kwargs: _capture_event(published, **kwargs),
    )

    admission = await coordinator.submit_active(
        _write_task("draft_write"),
        "topic_1",
        belong_to=_identity_scope().workspace_identity,
    )
    await controller.wait_task(admission.task.task_id)
    completed = await controller.get_task(admission.task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    assert completed.canonical_alias == "memory_alias"
    # 数据面：真实落库 mid_term
    stored = await memory_library.mid_term.get(
        _identity_scope().workspace_identity, atom.id, from_actor=_identity_scope().actor_identity
    )
    assert stored is not None, "WRITE 生成结果应写入真实 mid_term"
    # 控制面：settlement 由熟悉发布，pending atom 链路闭环
    assert published, "应发布 PENDING_ATOM_SETTLED 事件"
    assert published[0]["settlement"].pending_alias == "draft_write"
    assert published[0]["settlement"].canonical_alias == "memory_alias"


@pytest.mark.asyncio
async def test_settlement_saved_at_shutdown_lands_after_restart(
    memory_library, artifact_engine, tmp_path
):
    """关闭时结算产生的任务不调用提取、写进保存文件；新进程读回后由真实数据面落库。"""
    path = tmp_path / "pending_generation.json"
    atom = _memory_atom()
    stub = _StubGenerationEngine([GenerationOutcome(atom=atom, action=GenerationAction.CREATE)])

    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    _wire_generation_familiar(bus, memory_library, stub, artifact_engine)
    await controller.start()
    await controller.stop_dispatch()
    await coordinator.submit_settlement(_settlement_task())
    saved = await controller.save_unfinished(path)
    await controller.stop()

    assert (saved.saved, stub.requests) == (1, [])

    restarted_bus = PatchouliBus()
    _, restarted = _wire_generation_pipeline(restarted_bus)
    _wire_generation_familiar(restarted_bus, memory_library, stub, artifact_engine)
    await restarted.start()
    try:
        restored = await restarted.restore_saved(path)
        [task] = await restarted.wait_all()

        assert (restored.resubmitted, task.status) == (1, MemoryGenerationTaskStatus.COMPLETED)
        [request] = stub.requests
        assert request.context.fold_summary == "state summary"
        stored = await memory_library.mid_term.get(
            _identity_scope().workspace_identity,
            atom.id,
            from_actor=_identity_scope().actor_identity,
        )
        assert stored is not None
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_restored_topic_carries_on_and_its_next_settlement_drops_restated_memory(
    memory_library, artifact_engine, tmp_path
):
    """关闭后用同一工作目录重新启动：话题池先于保存的生成工作读回，关闭时沉淀所建的记忆记回
    读回的话题；新交互接着写进同一个话题，它再次沉淀时复述那条记忆的草稿被丢弃。"""
    pool_path = tmp_path / "topic_pool.json"
    work_path = tmp_path / "pending_generation.json"
    scope = _identity_scope()
    extractor = AsyncMock()
    extractor.extract.side_effect = [
        [_draft("deploy is manual", "deploys are now run by hand")],
        [_draft("deploy is manual", "deploys are now run by hand, restated")],
    ]
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )

    # 第一次运行：一轮交互之后正常关闭。关闭时的沉淀只入队，随生成工作与话题池一起保存。
    controller, perception, store = _wire_process(memory_library, engine, artifact_engine)
    await controller.start()
    topic_id = await perception.apply_interaction(
        make_interaction_payload("how do we deploy now", "deploys are run by hand"),
        belong_to=scope.workspace_identity,
        from_actor=scope.actor_identity,
        interaction_id="i-1",
    )
    await controller.stop_dispatch()
    await perception.flush_all_for_shutdown()
    await controller.save_unfinished(work_path)
    save_topic_pool(pool_path, store.list_all())
    await controller.stop()

    # 第二次运行：按启动顺序先读回话题池，再读回保存的生成工作。
    controller, perception, _ = _wire_process(memory_library, engine, artifact_engine)
    restore_topic_pool(pool_path, perception.restore_topics)
    await controller.start()
    try:
        await controller.restore_saved(work_path)
        await controller.wait_all()
        await perception.apply_interaction(
            make_interaction_payload("and staging?", "staging is run by hand too"),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            target_topic_id=topic_id,
            interaction_id="i-2",
        )
        await perception.manual_settle_topic(scope.workspace_identity, topic_id)
        await controller.wait_all()
    finally:
        await controller.stop()

    assert extractor.extract.await_count == 2
    assert [memory.payload.content for memory in await _stored(memory_library)] == [
        "deploys are now run by hand"
    ]


def _wire_process(
    memory_library: MemoryLibrary, engine: MemoryGenerationEngine, artifact_engine: ArtifactEngine
) -> tuple[MemoryGenerationTaskController, PerceptionFamiliar, ShortTermMemoryStore]:
    """一次运行的 Patchouli：真实感知、生成控制面与数据面；中期库跨运行共用，话题池各自为空。"""
    bus = PatchouliBus()
    store = ShortTermMemoryStore()
    coordinator, controller = _wire_generation_pipeline(bus, store)
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, coordinator.submit_settlement)
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)
    perception = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            config=PerceptionEngineConfig(fold_token_threshold=999999),
            relay_controller=Mock(),
        ),
        store=store,
        working_set=TopicWorkingSet(),
        bus=bus,
        config=MemoryPerceptionConfig(),
    )
    return controller, perception, store


def _draft(title: str, content: str) -> ExtractedMemoryDraft:
    return ExtractedMemoryDraft(
        title=title,
        summary="summary of the drafted memory",
        tags=["tag"],
        memory_type="FACT",
        content=content,
        confidence_score=0.9,
        has_value=True,
        alias_suffix="drafted",
    )


@pytest.mark.asyncio
async def test_passive_settlement_records_the_cited_segment_of_the_captured_turn(
    memory_library, tmp_path
):
    """被动结算的材料按响应切段并印出标签；模型引用的段换算为轮次标识加事件区间，写进
    新记忆的版本记录，被引用的轮次就在同一份版本记录指向的交互快照中。"""
    store = ArtifactStore(FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts")))
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    block = make_logical_block(
        turn_events=[
            TurnEvent(kind="user_message", sequence=0, content="以后怎么部署"),
            TurnEvent(kind="thought", sequence=1, content="先看脚本", response_index=1),
            TurnEvent(
                kind="tool_call",
                sequence=2,
                content="",
                response_index=1,
                action_id="toolu-1",
                tool_name="Bash",
                tool_args={"command": "cat deploy.sh"},
            ),
            TurnEvent(
                kind="assistant_message", sequence=3, content="改为手动部署。", response_index=2
            ),
        ]
    )
    extractor = AsyncMock()
    extractor.extract.return_value = [
        _draft("部署改为手动", "部署改为手动").model_copy(update={"source_segments": ["1.2"]})
    ]
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    _wire_generation_familiar(
        bus, memory_library, engine, ArtifactEngine.from_store(store, ArtifactConfig(enabled=True))
    )
    scope = _identity_scope()

    task = await coordinator.submit_settlement(
        TopicMaterializeTask(
            topic_id="topic_1",
            topic_title="topic title",
            interactions=(block,),
            belong_to=scope.workspace_identity,
            retires_topic=True,
        )
    )
    await controller.wait_task(task.task_id)

    assert "[Segment 1.2]\n[Assistant]: 改为手动部署。" in extractor.extract.call_args.args[0]
    [hit] = await memory_library.mid_term.search(
        scope.workspace_identity,
        query="部署",
        top_k=1,
        mode="dense",
        from_actor=scope.actor_identity,
    )
    refs = {ref.artifact_type: ref for ref in hit["memory"].payload.artifacts.refs}
    version = MemoryVersionArtifact.model_validate(
        await store.get(scope.workspace_identity, refs[ArtifactType.MEMORY_VERSION])
    )
    snapshot = InteractionArtifact.model_validate(
        await store.get(scope.workspace_identity, refs[ArtifactType.INTERACTION])
    )
    assert version.cited_segments == [
        CitedSegment(turn_id=block.turn.turn_id, first_sequence=3, last_sequence=3)
    ]
    assert [turn.turn_id for turn in snapshot.turns] == [block.turn.turn_id]


@pytest.mark.asyncio
async def test_passive_draft_restating_this_topics_write_does_not_rewrite_it(
    memory_library, artifact_engine
):
    """同一话题先写入、后结算：复述写入的被动草稿被丢弃，不产生合并任务，库中只有写入的那条。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    extractor = AsyncMock()
    extractor.extract_write.return_value = _draft(
        "agent title", "what the agent wrote"
    ).model_dump_json()
    extractor.extract.return_value = [_draft("restated title", "the agent's write, restated")]
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)
    scope = _identity_scope()

    admission = await coordinator.submit_active(
        _write_task("draft_write"), "topic_1", belong_to=scope.workspace_identity
    )
    await controller.wait_task(admission.task.task_id)
    settle_task = await coordinator.submit_settlement(_settlement_task())
    await controller.wait_task(settle_task.task_id)

    assert [task.source for task in await controller.list_tasks()] == [
        MemoryGenerationSource.WRITE,
        MemoryGenerationSource.SETTLE,
    ]
    stored = await memory_library.mid_term.search(
        scope.workspace_identity,
        query="any",
        top_k=10,
        mode="dense",
        from_actor=scope.actor_identity,
    )
    assert [hit["memory"].payload.content for hit in stored] == ["what the agent wrote"]


@pytest.mark.asyncio
async def test_second_settlement_drops_draft_restating_the_first_settlements_memory(
    memory_library, artifact_engine
):
    """话题空闲时沉淀、之后再次沉淀：材料仍是全部交互，复述第一次沉淀所建记忆的草稿被丢弃，
    不产生合并任务，库中只有第一次的那条。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    extractor = AsyncMock()
    extractor.extract.side_effect = [
        [_draft("deploy is manual", "deploys are now run by hand")],
        [_draft("deploy is manual", "deploys are now run by hand, restated")],
    ]
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)
    scope = _identity_scope()

    for retires_topic in (False, True):
        settle_task = await coordinator.submit_settlement(
            _settlement_task(retires_topic=retires_topic)
        )
        await controller.wait_task(settle_task.task_id)

    assert [task.source for task in await controller.list_tasks()] == [
        MemoryGenerationSource.SETTLE,
        MemoryGenerationSource.SETTLE,
    ]
    stored = await memory_library.mid_term.search(
        scope.workspace_identity,
        query="any",
        top_k=10,
        mode="dense",
        from_actor=scope.actor_identity,
    )
    assert [hit["memory"].payload.content for hit in stored] == ["deploys are now run by hand"]


@pytest.mark.asyncio
async def test_librarian_settlement_lists_this_topics_write_and_creates_what_is_missing(
    memory_library, artifact_engine
):
    """同一话题先写入、后由馆长结算：馆长列出了写入、检索了库，写下的记忆按新建入库，
    写入的那条保持原样。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    extractor = AsyncMock()
    extractor.extract_write.return_value = _draft(
        "agent title", "what the agent wrote"
    ).model_dump_json()
    new_fact = _draft("deploy is manual", "deploys are now run by hand")
    llm = ScriptedLLMService(
        json.dumps({"tool": "list_topic_writes", "args": {}}),
        json.dumps({"tool": "search_memories", "args": {"query": "deploy"}}),
        json.dumps({"tool": "write_memory", "args": new_fact.model_dump(exclude={"has_value"})}),
        json.dumps({"tool": "finish", "args": {}}),
    )
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
        librarian=Librarian(
            mid_term=memory_library.mid_term,
            llm_service=llm,
            llm_params=LLMRequestParams(model="test-model"),
            config=LibrarianConfig(),
        ),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)
    scope = _identity_scope()

    admission = await coordinator.submit_active(
        _write_task("draft_write"), "topic_1", belong_to=scope.workspace_identity
    )
    await controller.wait_task(admission.task.task_id)
    settle_task = await coordinator.submit_settlement(_settlement_task())
    settled = await controller.wait_task(settle_task.task_id)

    assert settled.status == MemoryGenerationTaskStatus.COMPLETED
    # 前两条是系统提示词与话题材料，其后每个工具结果跟在模型的动作之后。
    replies = [json.loads(m["content"]) for m in llm.calls[-1].messages[2:] if m["role"] == "user"]
    written_alias = replies[0]["result"][0]["alias"]
    assert written_alias in [hit["alias"] for hit in replies[1]["result"]]
    stored = await memory_library.mid_term.search(
        scope.workspace_identity,
        query="any",
        top_k=10,
        mode="dense",
        from_actor=scope.actor_identity,
    )
    assert sorted(hit["memory"].payload.content for hit in stored) == [
        "deploys are now run by hand",
        "what the agent wrote",
    ]


def _edit_set_json(*edits: dict, changelog: str = "revised") -> str:
    return json.dumps({"edits": list(edits), "changelog": changelog}, ensure_ascii=False)


async def _stored(memory_library: MemoryLibrary) -> list[MemoryAtom]:
    scope = _identity_scope()
    hits = await memory_library.mid_term.search(
        scope.workspace_identity,
        query="any",
        top_k=10,
        mode="dense",
        from_actor=scope.actor_identity,
    )
    return [hit["memory"] for hit in hits]


@pytest.mark.asyncio
async def test_librarian_settlement_revises_an_existing_memory_instead_of_adding_one(
    memory_library, artifact_engine
):
    """馆长读到一条被话题改变的记忆并修订它：库中是这条记忆的新版本，没有多出一条重复。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    existing = _memory_atom()
    existing.payload.content = "deploys run from CI"
    await memory_library.mid_term.upsert(existing, recompute_vectors=True)
    llm = ScriptedLLMService(
        json.dumps({"tool": "search_memories", "args": {"query": "deploy"}}),
        json.dumps({"tool": "read_memory", "args": {"alias": "memory_alias"}}),
        json.dumps(
            {
                "tool": "revise_memory",
                "args": {
                    "alias": "memory_alias",
                    "edits": [
                        {"op": "replace", "field": "content", "old": "from CI", "new": "by hand"}
                    ],
                    "changelog": "deploys moved off CI",
                },
            }
        ),
        json.dumps({"tool": "finish", "args": {}}),
    )
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=AsyncMock(),
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
        librarian=Librarian(
            mid_term=memory_library.mid_term,
            llm_service=llm,
            llm_params=LLMRequestParams(model="test-model"),
            config=LibrarianConfig(),
        ),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)

    settle_task = await coordinator.submit_settlement(_settlement_task())
    settled = await controller.wait_task(settle_task.task_id)

    assert settled.status == MemoryGenerationTaskStatus.COMPLETED
    [stored] = await _stored(memory_library)
    assert (stored.id, stored.payload.content, stored.meta.version) == (
        existing.id,
        "deploys run by hand",
        2,
    )


class _TitleSearchPort(_InMemoryMidTermPort):
    """检索只按标题中的字样命中，并记下每次写入是否重算向量。"""

    def __init__(self) -> None:
        super().__init__()
        self.recomputed: list[bool] = []

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        self.recomputed.append(recompute_vectors)
        await super().upsert(memory, recompute_vectors=recompute_vectors)

    async def search(self, scope, query, top_k, filters=None, mode="dense", **kwargs):
        return [
            {"memory": memory, "score": 1.0}
            for memory in self.memories.values()
            if memory.workspace_identity == scope and query in memory.index.title
        ]


@pytest.mark.asyncio
async def test_active_update_of_a_value_in_the_title_is_found_by_the_new_title(
    tmp_path, artifact_engine
):
    """主动 UPDATE 改了标题中的一个值：标题随修订更新、向量重算，检索按新标题命中。"""
    port = _TitleSearchPort()
    memory_library = MemoryLibrary(
        agent_profiles=AgentProfileStore(tmp_path / "agent_profiles"),
        short_term=ShortTermMemoryStore(),
        mid_term=MidTermMemoryStore(primary=port),
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"), compress=False)
        ),
    )
    existing = _memory_atom()
    existing.index.title = "service listens on port 3000"
    existing.payload.content = "The service listens on port 3000."
    await memory_library.mid_term.upsert(existing, recompute_vectors=True)
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    llm = ScriptedLLMService(
        _edit_set_json(
            {"op": "replace", "field": "content", "old": "port 3000", "new": "port 8080"},
            {"op": "replace", "field": "title", "old": "port 3000", "new": "port 8080"},
            changelog="Changed the port from 3000 to 8080",
        )
    )
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=LLMMemoryExtractor(ExtractorConfig(), llm, LLMRequestParams(model="m")),
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)
    scope = _identity_scope()

    async def memory_get(memory_id, *, belong_to, from_actor):
        return await memory_library.mid_term.get(belong_to, memory_id, from_actor=from_actor)

    bus.register(PatchouliLocalRoutes.MEMORY_GET, memory_get)

    admission = await coordinator.submit_active(
        _update_task(str(existing.id)), "topic_1", belong_to=scope.workspace_identity
    )
    completed = await controller.wait_task(admission.task.task_id)

    assert completed.status == MemoryGenerationTaskStatus.COMPLETED
    search = memory_library.mid_term.search
    [hit] = await search(
        scope.workspace_identity, query="port 8080", top_k=5, from_actor=scope.actor_identity
    )
    assert (hit["memory"].id, hit["memory"].payload.content, port.recomputed[-1]) == (
        existing.id,
        "The service listens on port 8080.",
        True,
    )
    assert (
        await search(
            scope.workspace_identity, query="port 3000", top_k=5, from_actor=scope.actor_identity
        )
        == []
    )


@pytest.mark.asyncio
async def test_two_drafts_of_one_settlement_both_merge_into_the_memory_they_hit(
    memory_library, artifact_engine
):
    """同一次结算的两条草稿命中同一条记忆：先各自新建，两次合并都落在这条记忆上，没有中止。"""
    bus = PatchouliBus()
    coordinator, controller = _wire_generation_pipeline(bus)
    await controller.start()
    existing = _memory_atom()
    await memory_library.mid_term.upsert(existing, recompute_vectors=True)
    extractor = AsyncMock()
    extractor.extract.return_value = [
        _draft("deploy detail one", "first new detail"),
        _draft("deploy detail two", "second new detail"),
    ]

    async def merge(retained, absorbed, *, retry=None):
        return json.dumps(
            {"edits": [{"op": "append", "text": absorbed.payload.content}], "changelog": "merged"}
        )

    extractor.merge.side_effect = merge
    engine = MemoryGenerationEngine(
        mid_term=memory_library.mid_term,
        extractor=extractor,
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    _wire_generation_familiar(bus, memory_library, engine, artifact_engine)

    settle_task = await coordinator.submit_settlement(_settlement_task())
    await controller.wait_task(settle_task.task_id)
    # 合并任务在结算的新建落库之后才提交，等结算结束后再等它们。
    await controller.wait_all(timeout=5)
    tasks = await controller.list_tasks()

    assert sorted((task.source.value, task.status.value) for task in tasks) == [
        ("MERGE", "completed"),
        ("MERGE", "completed"),
        ("SETTLE", "completed"),
    ]
    [stored] = await _stored(memory_library)
    assert (stored.id, stored.meta.version) == (existing.id, 3)
    assert (
        "first new detail" in stored.payload.content,
        "second new detail" in stored.payload.content,
    ) == (True, True)
