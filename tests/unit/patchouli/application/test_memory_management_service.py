"""ChatApplicationService / PassiveIngressService 委托测试"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from uuid import uuid4

from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.core.models import Artifacts
from hivememory.engines.lifecycle.models import EventType, ReinforcementResult
from hivememory.patchouli.application import MemoryManagementService
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.application import TopicManagementService
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import (
    AgentRunContext,
    AnalyzeAndRetrieveResult,
    AgentRunResult,
    EyeGazeResult,
    RetrievalResponse,
)
from hivememory.patchouli.models import (
    PreparedAgentRun,
    StreamPrelude,
)
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.memory_service import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
from hivememory.system.application.passive import PassiveIngressEvent
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


def _make_prepared_run(**overrides) -> PreparedAgentRun:
    identity = Identity(user_id="u1", agent_id="omni_doll")
    gaze_result = EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="resolved",
        search_keywords=["k"],
        worth_saving=True,
        raw_query="hi",
        identity=identity,
        target_topic="topic_1",
    )
    defaults = dict(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context={"blocks": [], "state_summary": ""},
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
        gaze_result=gaze_result,
        generation_options=None,
    )
    defaults.update(overrides)
    return PreparedAgentRun(**defaults)


def _make_chat_result() -> AgentRunResult:
    return AgentRunResult(
        final_text="hello!",
        mtp_iterations=0,
        total_iterations=1,
        turn_events=[],
    )


@pytest.fixture
def mock_global_bus():
    """模拟 GlobalSystemBus，根据路由返回不同结果。"""
    bus = MagicMock(spec=GlobalSystemBus)

    prepared = _make_prepared_run()
    chat_result = _make_chat_result()

    async def route_dispatch(route, *args, **kwargs):
        if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
            return prepared
        elif route == GlobalRoutes.ALICE_RUN_AGENT:
            return chat_result
        elif route == GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN:
            return None
        elif route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
            return True
        elif route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
            async def _stream():
                yield {"event": "token", "data": {"content": "hi"}}
                yield {"event": "done", "data": chat_result.model_dump()}
            return _stream()
        return None

    bus.request = AsyncMock(side_effect=route_dispatch)
    return bus


@pytest.fixture
def passive_config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler
    return config


def _make_analysis_result(
    *,
    target_topic: str = "NEW_TOPIC",
    memory: str | None = "<mem>ctx</mem>",
    worth_saving: bool = True,
) -> AnalyzeAndRetrieveResult:
    gaze_result = EyeGazeResult(
        intent=GatewayIntent.RAG,
        rewritten_query="resolved query",
        search_keywords=["resolved"],
        worth_saving=worth_saving,
        raw_query="raw query",
        identity=Identity(user_id="u1"),
        target_topic=target_topic,
    )
    retrieval_result = RetrievalResponse(
        memories=[],
        rendered_context=memory or "",
    )
    return AnalyzeAndRetrieveResult(
        gaze_result=gaze_result,
        retrieval_result=retrieval_result,
    )


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestMemoryManagementService:
    @pytest.fixture
    def storage(self):
        return MagicMock()

    def test_create_memory_writes_storage(self, storage):
        service = MemoryManagementService(storage=storage)
        atom = _make_memory_atom(title="Created memory")

        result = asyncio.run(service.create_memory(atom))

        assert result is atom
        storage.upsert_memory.assert_called_once_with(atom)

    def test_list_memories_uses_filters_and_refreshes_vitality(
        self,
        storage,
    ):
        atom = _make_memory_atom()
        lifecycle = MagicMock()
        lifecycle.refresh_vitality_batch.side_effect = (
            lambda atoms, persist=False: setattr(atoms[0].meta, "vitality_score", 33.0)
        )
        service = MemoryManagementService(
            storage=storage,
            lifecycle_engine=lifecycle,
        )
        storage.get_all_memories.return_value = [atom]

        atoms = asyncio.run(service.list_memories(
            filters={"meta.user_id": "u1", "index.memory_type": "FACT"},
            limit=10,
        ))

        storage.get_all_memories.assert_called_once_with(
            filters={"meta.user_id": "u1", "index.memory_type": "FACT"},
            limit=10,
        )
        lifecycle.refresh_vitality_batch.assert_called_once_with([atom], persist=False)
        assert atoms == [atom]
        assert atoms[0].meta.vitality_score == 33.0

    def test_list_memories_search_excludes_agent_profiles(self, storage):
        service = MemoryManagementService(storage=storage)
        fact = _make_memory_atom(title="Fact")
        profile = _make_memory_atom(title="Agent")
        profile.index.memory_type = MemoryType.AGENT_PROFILE
        storage.search_memories.return_value = [
            {"memory": fact, "score": 0.9},
            {"memory": profile, "score": 0.8},
        ]

        atoms = asyncio.run(service.list_memories(
            query="test",
            limit=5,
            exclude_types=[MemoryType.AGENT_PROFILE.value],
        ))

        storage.search_memories.assert_called_once_with(
            query_text="test",
            top_k=5,
            filters=None,
        )
        assert atoms == [fact]

    def test_get_memory_returns_none_when_not_found(self, storage):
        service = MemoryManagementService(storage=storage)
        storage.get_memory.return_value = None

        assert asyncio.run(service.get_memory(uuid4())) is None

    def test_update_memory_updates_editable_fields(self, storage):
        service = MemoryManagementService(storage=storage)
        atom = _make_memory_atom()
        storage.get_memory.return_value = atom

        updated = asyncio.run(service.update_memory(
            atom.id,
            title="Updated",
            summary="Updated summary",
            content="Updated content",
            alias="updated-alias",
            tags=["updated"],
            agent_config={"mode": "test"},
        ))

        assert updated is atom
        assert atom.index.title == "Updated"
        assert atom.index.summary == "Updated summary"
        assert atom.payload.content == "Updated content"
        assert atom.index.alias == "updated-alias"
        assert atom.index.tags == ["updated"]
        assert atom.payload.artifacts.agent_config == {"mode": "test"}
        storage.upsert_memory.assert_called_once_with(atom)

    def test_record_feedback_uses_lifecycle(self, storage):
        mid = uuid4()
        lifecycle = MagicMock()
        lifecycle.record_feedback.return_value = ReinforcementResult(
            memory_id=mid,
            previous_vitality=40.0,
            new_vitality=90.0,
            previous_confidence=0.8,
            new_confidence=0.8,
            event_type=EventType.FEEDBACK_POSITIVE,
        )
        service = MemoryManagementService(
            storage=storage,
            lifecycle_engine=lifecycle,
        )

        result = asyncio.run(service.record_feedback(
            mid,
            positive=True,
            source="ui.memory_ref",
        ))

        lifecycle.record_feedback.assert_called_once_with(
            mid,
            positive=True,
            source="ui.memory_ref",
        )
        assert result.memory_id == mid


