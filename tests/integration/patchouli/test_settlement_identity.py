"""话题结算的发起者与生成查重可见性跨组件回归。"""

from types import SimpleNamespace

import pytest

from hivememory.config.patchouli import (
    ArtifactConfig,
    DeduplicatorConfig,
    SemanticFlowPerceptionConfig,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryVisibility,
    PayloadLayer,
    TurnEvent,
    WorkspaceIdentity,
    WriteFocus,
    system_actor_for_workspace,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
)
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTaskStatus
from hivememory.patchouli.control.interaction_apply_journal import InMemoryInteractionApplyJournal
from hivememory.patchouli.control.memory_generation.controller import MemoryGenerationTaskController
from hivememory.patchouli.control.memory_generation.coordinator import MemoryGenerationCoordinator
from hivememory.patchouli.memory_library.adapters.artifact import FilesystemArtifactStorageAdapter
from hivememory.patchouli.memory_library.adapters.long_term import FileBasedStorageAdapter
from hivememory.patchouli.memory_library.adapters.mid_term import QdrantStorageAdapter
from hivememory.patchouli.memory_library.library import MemoryLibrary
from hivememory.patchouli.memory_library.stores import (
    ArtifactStore,
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from tests.helpers.memory import make_memory_metadata


class _Clock:
    """测试驱动的单调时钟，无需等待真实空闲时间。"""

    value = 0.0

    def __call__(self) -> float:
        return self.value


class _Relay:
    """LLM 折叠端口替身；本测试的阈值不触发折叠。"""

    async def generate_summary(self, _blocks, _previous_summary):
        raise AssertionError("settlement identity test must not compact")


class _Extractor:
    """LLM 提取边界的确定性草稿替身。"""

    async def extract(self, **_kwargs):
        return ExtractedMemoryDraft(
            title="相似记忆",
            summary="新的总结",
            tags=[],
            memory_type="FACT",
            content="new content",
            confidence_score=1.0,
            has_value=True,
            alias_suffix="shared_fact",
        )


class _LeakyVectorStore:
    """向量服务即使泄漏受限命中，真实 adapter 也必须重验可见性。"""

    def __init__(self, memory: MemoryAtom):
        self.memory = memory

    async def search_memories(self, **_kwargs):
        return [{"memory": self.memory, "score": 0.9}]

    async def get_memory_ids_by_alias(self, *_args, **_kwargs):
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("active_write", [False, True], ids=["settle", "write"])
async def test_settle_skips_restricted_dedup_but_write_keeps_actor_visibility(
    active_write: bool,
) -> None:
    """SETTLE 合并受限记忆或 WRITE 丢失提交者可见性时必须失败。

    两种生成模式以不同发起者查重；受限策略的 PRIVATE/TEAM 判定矩阵归 core
    可读性单测，这里取 PRIVATE 代表。
    """
    belong_to = WorkspaceIdentity(
        owner_user_id="owner-1", workspace_key="main_workspace", workspace_id="main_workspace"
    )
    from_actor = ActorIdentity(user_id="owner-1", agent_id="agent-1", team_id="team-1")
    policy = MemoryAccessPolicy(visibility=MemoryVisibility.PRIVATE, target_agent_id="agent-1")
    existing = MemoryAtom(
        meta=make_memory_metadata(
            user_id="owner-1", source_agent_id="agent-1", access_policy=policy
        ),
        index=IndexLayer(title="相似记忆", summary="旧总结", memory_type="FACT"),
        payload=PayloadLayer(content="old content"),
    )
    engine = MemoryGenerationEngine(
        mid_term=MidTermMemoryStore(QdrantStorageAdapter(_LeakyVectorStore(existing))),
        extractor=_Extractor(),
        deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
    )
    request = GenerationRequest(
        context=GenerationContext(
            turns=[GenerationTurn(user_query="question", identity=from_actor)]
        ),
        write_focus=WriteFocus(content="new content") if active_write else None,
    )
    outcomes = await engine.process(
        request,
        belong_to=belong_to,
        from_actor=from_actor if active_write else system_actor_for_workspace(belong_to),
    )

    assert [outcome.duplicate_decision for outcome in outcomes] == [
        DuplicateDecision.UPDATE if active_write else DuplicateDecision.CREATE
    ]
    atom = outcomes[0].atom
    assert atom.workspace_identity == belong_to
    assert atom.payload.content == "new content"
    assert atom.meta.access_policy == (policy if active_write else MemoryAccessPolicy.public())
    assert (atom.id == existing.id) is active_write
    assert atom.meta.provenance.source_agent_id == ("agent-1" if active_write else SYSTEM_AGENT_ID)
    assert atom.meta.provenance.contributing_agent_ids == ("agent-1",)


class _WritableVectorStore:
    """外部 Qdrant 技术端口的内存替身，保留真实读写效果。"""

    def __init__(self, memory: MemoryAtom) -> None:
        self._memories = {memory.id: memory.model_copy(deep=True)}

    async def search_memories(self, **_kwargs):
        return [
            {"memory": memory.model_copy(deep=True), "score": 0.9}
            for memory in self._memories.values()
        ]

    async def get_memory_ids_by_alias(self, alias, *, limit, **_kwargs):
        return [memory.id for memory in self._memories.values() if memory.index.alias == alias][
            :limit
        ]

    async def upsert_memory(self, memory, **_kwargs):
        self._memories[memory.id] = memory.model_copy(deep=True)

    async def get_memory(self, key):
        memory = self._memories.get(key.memory_id)
        return memory.model_copy(deep=True) if memory is not None else None

    async def get_all_memories(self, **_kwargs):
        return [memory.model_copy(deep=True) for memory in self._memories.values()]


@pytest.mark.asyncio
async def test_idle_settlement_persists_public_memory_without_merging_similar_private(
    tmp_path,
) -> None:
    """完整空闲结算链必须新建 PUBLIC，既有高相似 PRIVATE 内容保持不变。"""
    belong_to = WorkspaceIdentity(
        owner_user_id="owner-1", workspace_key="main_workspace", workspace_id="main_workspace"
    )
    from_actor = ActorIdentity(user_id="owner-1", agent_id="agent-1", team_id="team-1")
    private = MemoryAtom(
        meta=make_memory_metadata(
            user_id="owner-1",
            source_agent_id="agent-1",
            visibility=MemoryVisibility.PRIVATE,
        ),
        index=IndexLayer(
            title="相似记忆", summary="旧总结", memory_type="FACT", alias="fact_private"
        ),
        payload=PayloadLayer(content="old private content"),
    )
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(_WritableVectorStore(private)))
    store = ShortTermMemoryStore()
    artifacts = ArtifactStore(
        FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts"))
    )
    library = MemoryLibrary(
        short_term=store,
        mid_term=mid_term,
        long_term=LongTermMemoryStore(
            FileBasedStorageAdapter(archive_dir=str(tmp_path / "archive"))
        ),
        artifact_store=artifacts,
    )
    bus = PatchouliBus()
    coordinator = MemoryGenerationCoordinator(bus=bus, recent_block_limit=5)
    controller = MemoryGenerationTaskController(bus=bus)
    generation = MemoryGenerationFamiliar(
        generation_engine=MemoryGenerationEngine(
            mid_term=mid_term,
            extractor=_Extractor(),
            deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
        ),
        memory_library=library,
        artifact_engine=ArtifactEngine.from_store(artifacts, ArtifactConfig(enabled=True)),
    )
    bus.register(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, coordinator.submit_settlement)
    bus.register(PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION, controller.submit_generation)
    bus.register(PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, generation.execute)
    clock = _Clock()
    perception = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            SemanticFlowPerceptionConfig(fold_token_threshold=999999), _Relay()
        ),
        store=store,
        working_set=TopicWorkingSet(clock=clock),
        bus=bus,
        config=SimpleNamespace(idle_timeout_seconds=5),
        interaction_journal=InMemoryInteractionApplyJournal(),
    )
    await controller.start()
    try:
        topic_id = await perception.apply_interaction(
            InteractionPayload(
                user_message="question",
                assistant_final_text="answer",
                turn_events=[
                    TurnEvent(
                        kind="assistant_message", sequence=0, role="assistant", content="answer"
                    )
                ],
            ),
            belong_to=belong_to,
            from_actor=from_actor,
            interaction_id="idle-interaction",
        )
        clock.value = 6
        assert await perception.scan_idle_buffers_once() == [topic_id]
        (task,) = await controller.list_tasks()
        completed = await controller.wait_task(task.task_id, timeout=2)

        assert completed.status == MemoryGenerationTaskStatus.COMPLETED
        assert completed.from_actor == system_actor_for_workspace(belong_to)
        assert completed.belong_to == belong_to
        assert store.get(belong_to, topic_id) is None
        stored_private = await mid_term.get(belong_to, private.id, from_actor=from_actor)
        assert stored_private.payload.content == "old private content"
        assert stored_private.meta.version == 1
        assert stored_private.meta.access_policy.visibility == MemoryVisibility.PRIVATE
        (public,) = await mid_term.scroll(
            belong_to, from_actor=system_actor_for_workspace(belong_to)
        )
        assert public.id != private.id
        assert public.index.alias == completed.canonical_alias
        assert public.payload.content == "new content"
        assert public.meta.access_policy == MemoryAccessPolicy.public()
        assert public.meta.provenance.source_agent_id == SYSTEM_AGENT_ID
        assert public.meta.provenance.contributing_agent_ids == ("agent-1",)
        assert len(await artifacts.list_by_memory(belong_to, str(public.id))) == 2
    finally:
        await controller.stop()
