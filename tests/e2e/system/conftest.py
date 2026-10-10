"""HTTP 系统验收的真实 Patchouli/Gateway 组合，仅替换外部存储与 LLM 端口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest_asyncio

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.gateway import SystemGatewayConfig
from hivememory.config.patchouli import (
    ArtifactConfig,
    DeduplicatorConfig,
    DenseRetrieverConfig,
    GarbageCollectorConfig,
    MemoryPerceptionConfig,
    ReinforcementEngineConfig,
    SemanticFlowPerceptionConfig,
    VitalityCalculatorConfig,
)
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import MemoryAtom, MemoryLifecycleState
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine
from hivememory.engines.lifecycle.garbage_collector import PeriodicGarbageCollector
from hivememory.engines.lifecycle.reinforcement import DynamicReinforcementEngine
from hivememory.engines.lifecycle.vitality import VitalityCalculator
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.retrieval.engine import RetrievalEngine
from hivememory.engines.retrieval.retriever import DenseRetriever
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.patchouli.application import (
    AgentProfileManagementService,
    InteractionSubmissionService,
    MemoryIntentSubmissionService,
    MemoryManagementService,
    MemoryTaskManagementService,
    ModelReadinessService,
    TopicManagementService,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_apply_journal import InMemoryInteractionApplyJournal
from hivememory.patchouli.control.interaction_submission import InteractionSubmissionQueue
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
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.patchouli.services.lifecycle import LifecycleFamiliar
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet


class _VectorStore:
    """外部向量服务的内存替身，真实 adapter 继续执行归属和读取策略校验。"""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryAtom] = {}
        self.client = self
        self.search_matches: dict[str, tuple[str, ...]] = {}

    async def get_collections(self) -> list:
        return []

    async def search_memories(
        self, *, query_text: str, top_k: int, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        """语义匹配由外部服务替身提供，领域侧的排序和编译保持真实。"""
        selected = self.search_matches.get(query_text, ())
        return [
            {"memory": atom.model_copy(deep=True), "score": 1.0}
            for atom in self._memories.values()
            if atom.index.alias in selected
        ][:top_k]

    async def get_memory_ids_by_alias(self, alias: str, *, limit: int, **_kwargs: Any) -> list:
        return [atom.id for atom in self._memories.values() if atom.index.alias == alias][:limit]

    async def upsert_memory(self, memory: MemoryAtom, **_kwargs: Any) -> None:
        self._memories[str(memory.id)] = memory.model_copy(deep=True)

    async def get_memory(self, key) -> MemoryAtom | None:
        atom = self._memories.get(str(key.memory_id))
        return atom.model_copy(deep=True) if atom is not None else None

    async def get_memory_by_alias(self, alias: str, **_kwargs: Any) -> MemoryAtom | None:
        matches = [atom for atom in self._memories.values() if atom.index.alias == alias]
        return matches[0].model_copy(deep=True) if matches else None

    async def patch_memory_payload(self, key, *, lifecycle=None, access_policy=None) -> None:
        """持久化生命周期整段 patch，便于通过正式读取断言引用计数。"""
        atom = self._memories[str(key.memory_id)]
        if lifecycle is not None:
            atom.meta.lifecycle = MemoryLifecycleState.model_validate(lifecycle)
        if access_policy is not None:
            atom.meta.access_policy = type(atom.meta.access_policy).model_validate(access_policy)


class _Extractor:
    """外部 LLM 提取端口的确定性替身，生成业务流程由真实 engine 执行。"""

    def extract(self, **_kwargs: Any) -> ExtractedMemoryDraft:
        return ExtractedMemoryDraft(
            title="跨边界写入",
            alias_suffix="http_chat",
            summary="保存本轮事实",
            tags=[],
            memory_type="FACT",
            content="HTTP chat materialized content",
            confidence_score=1.0,
            has_value=True,
        )


class _Relay:
    """本轮不会达到折叠阈值；意外调用模型边界时直接令测试失败。"""

    def generate_summary(self, *_args: Any) -> str:
        raise AssertionError("this chat must not compact")


@dataclass
class PatchouliChatStack:
    """供公开 HTTP 验收使用的真实组件与外部存储替身。"""

    bus: GlobalSystemBus
    short_term: ShortTermMemoryStore
    mid_term: MidTermMemoryStore
    controller: MemoryGenerationTaskController
    lifecycle: LifecycleFamiliar
    vector_store: _VectorStore


@pytest_asyncio.fixture
async def patchouli_chat_stack(tmp_path) -> AsyncIterator[PatchouliChatStack]:
    """完整 prepare/finalize、检索、引用和物化链路共用同一份真实资源所有者。"""
    global_bus = GlobalSystemBus()
    local_bus = PatchouliBus()
    short_term = ShortTermMemoryStore()
    vector_store = _VectorStore()
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(vector_store))
    artifacts = ArtifactStore(
        FilesystemArtifactStorageAdapter(root_dir=str(tmp_path / "artifacts"))
    )
    library = MemoryLibrary(
        short_term=short_term,
        mid_term=mid_term,
        long_term=LongTermMemoryStore(FileBasedStorageAdapter(archive_dir=str(tmp_path))),
        artifact_store=artifacts,
    )
    retrieval = RetrievalFamiliar(
        RetrievalEngine(DenseRetriever(mid_term, DenseRetrieverConfig())), library, local_bus
    )
    perception = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            SemanticFlowPerceptionConfig(fold_token_threshold=999999), _Relay()
        ),
        store=short_term,
        working_set=TopicWorkingSet(),
        bus=local_bus,
        config=MemoryPerceptionConfig(),
        interaction_journal=InMemoryInteractionApplyJournal(),
    )
    generation = MemoryGenerationFamiliar(
        generation_engine=MemoryGenerationEngine(
            mid_term=mid_term,
            extractor=_Extractor(),
            deduplicator=MemoryDeduplicator(DeduplicatorConfig()),
        ),
        memory_library=library,
        artifact_engine=ArtifactEngine.from_store(artifacts, ArtifactConfig()),
    )
    vitality = VitalityCalculator(VitalityCalculatorConfig())
    lifecycle = LifecycleFamiliar(
        lifecycle_engine=MemoryLifecycleEngine(
            mid_term,
            vitality,
            DynamicReinforcementEngine(mid_term, ReinforcementEngineConfig(), vitality),
            PeriodicGarbageCollector(library, GarbageCollectorConfig()),
        ),
        memory_library=library,
    )
    controller = MemoryGenerationTaskController(bus=local_bus)
    coordinator = MemoryGenerationCoordinator(bus=local_bus)

    async def storage_health() -> bool:
        """读取真实书库健康结果，省去验收范围之外的模型装配。"""
        return (await library.check_storage_health()).healthy

    for route, handler in (
        (PatchouliLocalRoutes.TOPIC_PREPARE, perception.prepare_topic),
        (PatchouliLocalRoutes.TOPIC_GET, retrieval.get_topic),
        (PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, retrieval.list_active_topics),
        (PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, perception.discard_if_empty),
        (PatchouliLocalRoutes.GET_AGENT_PROFILE, retrieval.get_agent_profile),
        (PatchouliLocalRoutes.MEMORY_RETRIEVE, retrieval.retrieve_async),
        (PatchouliLocalRoutes.MEMORY_RETRIEVE_BY_ALIASES, retrieval.retrieve_by_aliases_async),
        (PatchouliLocalRoutes.MEMORY_GET, retrieval.get_memory),
        (PatchouliLocalRoutes.MEMORY_RECORD_HIT, lifecycle.record_hit),
        (PatchouliLocalRoutes.MEMORY_RECORD_CITATION, lifecycle.record_citation),
        (PatchouliLocalRoutes.REFRESH_MEMORY_VITALITY, lifecycle.refresh_memory_vitality),
        (PatchouliLocalRoutes.RUNTIME_STORAGE_HEALTH, storage_health),
        (PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE, coordinator.submit_active),
        (PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC, generation.execute),
        (
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
            controller.submit_generation_many,
        ),
    ):
        local_bus.register(route, handler)
    queue = InteractionSubmissionQueue(perception.apply_interaction)
    chat_service = PatchouliService(local_bus, interaction_queue=queue)
    bridge = PatchouliBridge(
        local_bus=local_bus,
        global_bus=global_bus,
        public_api=PatchouliPublicApi(
            chat=chat_service,
            memory=MemoryManagementService(bus=local_bus),
            memory_tasks=MemoryTaskManagementService(bus=local_bus),
            agent_profiles=AgentProfileManagementService(bus=local_bus),
            interactions=InteractionSubmissionService(interaction_queue=queue),
            memory_intents=MemoryIntentSubmissionService(bus=local_bus),
            topics=TopicManagementService(bus=local_bus),
            readiness=ModelReadinessService(local_bus),
        ),
    )
    bridge.mount()
    # 无模型保守模式仍经过真实 Gateway workflow 和话题上下文读取。
    gateway = GatewayService(GatewayRuntime(config=SystemGatewayConfig(), global_bus=global_bus))
    global_bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway.process)
    await controller.start()
    await queue.start()
    try:
        yield PatchouliChatStack(
            global_bus, short_term, mid_term, controller, lifecycle, vector_store
        )
    finally:
        await chat_service.drain_active_finalizations()
        await queue.stop()
        await controller.stop()
        bridge.unmount()
        global_bus.unregister(GlobalRoutes.GATEWAY_PROCESS)
