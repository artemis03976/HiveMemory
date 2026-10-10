"""HTTP chat 经真实身份边界、任务进程和 Patchouli 物化链路的确定性 E2E。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.gateway import SystemGatewayConfig
from hivememory.config.patchouli import (
    ArtifactConfig,
    DeduplicatorConfig,
    DenseRetrieverConfig,
    MemoryPerceptionConfig,
    SemanticFlowPerceptionConfig,
)
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.models import ActorIdentity, MemoryAtom, TurnEvent, WriteFocus
from hivememory.engines.artifacts.engine import ArtifactEngine
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import ExtractedMemoryDraft
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
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationTaskStatus
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
from hivememory.patchouli.services.memory_generation import MemoryGenerationFamiliar
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.retrieval import RetrievalFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.server import deps
from hivememory.server.routers.chat import router
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)

pytestmark = pytest.mark.e2e


class _VectorStore:
    """外部向量服务的内存替身；真实存储 adapter 仍执行归属与可见性校验。"""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryAtom] = {}
        self.client = self

    async def get_collections(self) -> list:
        return []

    async def search_memories(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

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


@pytest.mark.asyncio
async def test_http_chat_preserves_split_identity_through_finalize_and_materialization(tmp_path):
    """HTTP 完成后交互与物化任务保留身份，done 话题池来自真实归属读取。"""
    workspace = make_workspace_identity(owner_user_id="u1")
    other_workspace = make_workspace_identity(owner_user_id="u1", workspace_id="other")
    actor = ActorIdentity(user_id="u1", agent_id="omni_doll")
    access = make_access_composition(
        [make_actor_access_record(owner_user_id="u1", agent_id="omni_doll")],
        adapters=("http",),
        default_workspace=workspace,
    )
    global_bus = GlobalSystemBus()
    local_bus = PatchouliBus()
    short_term = ShortTermMemoryStore()
    foreign_topic = short_term.create(other_workspace, topic_title="其他归属")
    mid_term = MidTermMemoryStore(QdrantStorageAdapter(_VectorStore()))
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
    controller = MemoryGenerationTaskController(bus=local_bus)
    coordinator = MemoryGenerationCoordinator(bus=local_bus)

    async def storage_health() -> bool:
        """定向组合读取真实书库健康结果，省去本测试范围之外的模型装配。"""
        return (await library.check_storage_health()).healthy

    for route, handler in (
        (PatchouliLocalRoutes.TOPIC_PREPARE, perception.prepare_topic),
        (PatchouliLocalRoutes.TOPIC_GET, retrieval.get_topic),
        (PatchouliLocalRoutes.TOPIC_LIST_ACTIVE, retrieval.list_active_topics),
        (PatchouliLocalRoutes.TOPIC_DISCARD_IF_EMPTY, perception.discard_if_empty),
        (PatchouliLocalRoutes.GET_AGENT_PROFILE, retrieval.get_agent_profile),
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
    # Gateway 的无模型保守模式保持真实 workflow 与公开话题上下文读取。
    gateway = GatewayService(GatewayRuntime(config=SystemGatewayConfig(), global_bus=global_bus))
    global_bus.register(GlobalRoutes.GATEWAY_PROCESS, gateway.process)
    submitted = []

    async def write(operations):
        """测试 CPU 经进程端口提交意图，物化任务由进程认领。"""
        submitted.append(await operations.submit_write_intent(WriteFocus(content="保存本轮事实")))

    cpu = ScriptedCPU(
        result=make_cpu_result(
            final_text="本轮完成",
            turn_events=[
                TurnEvent(
                    kind="assistant_message", sequence=0, role="assistant", content="本轮完成"
                )
            ],
        ),
        operation_script=write,
    )
    process_service = make_task_process_service(
        global_bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[deps.get_process_service] = lambda: process_service
    app.dependency_overrides[deps.get_server_principal_id] = lambda: access.principal.principal_id
    await controller.start()
    await queue.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                headers={"X-User-ID": "u1", "X-Workspace-ID": workspace.workspace_id},
                json={
                    "message": "保存本轮事实",
                    "agent_id": actor.agent_id,
                    "session_id": "external-session",
                    "enable_memory_retrieval": False,
                },
            )
        assert response.status_code == 200
        events = []
        event_name = ""
        for line in response.text.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line.removeprefix("data: "))))
        terminal, done = events[-1]
        assert terminal == "done"
        assert done["status"] == "completed"
        assert done["final_text"] == "本轮完成"
        (topic_snapshot,) = done["pool_topics"]
        assert topic_snapshot["workspace_identity"] == workspace.model_dump(mode="json")
        assert topic_snapshot["block_count"] == 1
        assert topic_snapshot["last_turn"] == {"user": "保存本轮事实", "assistant": "本轮完成"}
        topic = short_term.get(workspace, topic_snapshot["topic_id"])
        assert topic.blocks[0].turn.identity == actor
        assert "session_id" not in topic.blocks[0].turn.identity.model_dump()
        assert short_term.get(other_workspace, foreign_topic.topic_id) == foreign_topic

        (task_id,) = done["memory_task_ids"]
        task = await controller.wait_task(task_id, timeout=2)
        assert task.status == MemoryGenerationTaskStatus.COMPLETED
        assert task.belong_to == workspace
        assert task.from_actor == actor
        atom = await mid_term.get_by_alias(workspace, task.canonical_alias, from_actor=actor)
        assert atom.payload.content == "HTTP chat materialized content"
        assert atom.workspace_identity == workspace
        assert atom.meta.provenance.source_agent_id == actor.agent_id
        assert len(cpu.calls) == 1
        assert cpu.calls[0].manifest.identity_scope.actor_identity == actor
        assert cpu.calls[0].manifest.identity_scope.workspace_identity == workspace
        assert cpu.closed is True
    finally:
        await chat_service.drain_active_finalizations()
        await queue.stop()
        await controller.stop()
        bridge.unmount()
        global_bus.unregister(GlobalRoutes.GATEWAY_PROCESS)
        access.gateway.close()
        access.gateway.revoke_all_contexts()
