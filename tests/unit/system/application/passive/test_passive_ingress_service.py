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
            topic_context=None,
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_topics=[],
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


class TestPassiveIngressService:
    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def scheduler(self):
        return GlobalMaintenanceScheduler(tick_seconds=0.01, shutdown_wait_seconds=0.1)

    @pytest.mark.asyncio
    async def test_ingest_event_passes_through_bus(self, bus, passive_config, scheduler):
        submit_interaction = AsyncMock(return_value=None)
        bus.register(
            GlobalRoutes.PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE,
            AsyncMock(return_value=_make_analysis_result(memory="<memory>relevant</memory>")),
        )
        bus.register(GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION, submit_interaction)
        svc = PassiveIngressService(bus=bus, config=passive_config, scheduler=scheduler)
        event = PassiveIngressEvent(role="user", content="hello")
        result = await svc.ingest_event(
            event=event,
            user_id="u1",
            agent_id="agent_y",
            session_id="s2",
        )
        assert result["intent"] == "RAG"
        assert result["memory"] == "<memory>relevant</memory>"
        submit_interaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_ingressor(self, bus, passive_config, scheduler):
        bus.register(
            GlobalRoutes.PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE,
            AsyncMock(return_value=_make_analysis_result(target_topic="topic_1")),
        )
        submit_interaction = AsyncMock(return_value=None)
        bus.register(GlobalRoutes.PATCHOULI_SUBMIT_INTERACTION, submit_interaction)
        svc = PassiveIngressService(bus=bus, config=passive_config, scheduler=scheduler)
        await svc.ingest_event(
            event=PassiveIngressEvent(role="user", content="q"),
            user_id="u1",
            agent_id="a",
            session_id="s",
        )
        await svc.ingest_event(
            event=PassiveIngressEvent(role="assistant", content="a"),
            user_id="u1",
            agent_id="a",
            session_id="s",
        )
        result = await svc.flush_ingressor(user_id="u1", agent_id="a", session_id="s")
        assert result is True
        submit_interaction.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_and_stop_register_tasks_on_global_scheduler(
        self, bus, passive_config, scheduler
    ):
        passive_config.scheduler.enabled = True
        svc = PassiveIngressService(bus=bus, config=passive_config, scheduler=scheduler)

        await svc.start()
        task_keys = {spec.task_key for spec in scheduler.list_tasks()}
        assert "system.passive_ingress.observer_idle_flush" in task_keys

        await svc.stop()
        task_keys = {spec.task_key for spec in scheduler.list_tasks()}
        assert "system.passive_ingress.observer_idle_flush" not in task_keys
