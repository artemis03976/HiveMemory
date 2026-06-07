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


class TestChatApplicationService:
    @pytest.mark.asyncio
    async def test_chat_calls_prepare_run_finalize(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        result = await svc.chat(
            user_message="hi",
            user_id="u1",
            agent_id="omni_doll",
            session_id="s1",
            enable_memory_retrieval=True,
            generation_options={"max_tokens": 100},
        )
        assert result.final_text == "hello!"
        # 3 bus calls: prepare, run_agent, finalize
        assert mock_global_bus.request.await_count == 3
        routes_called = [
            call.args[0] for call in mock_global_bus.request.await_args_list
        ]
        assert GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN in routes_called
        assert GlobalRoutes.ALICE_RUN_AGENT in routes_called
        assert GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN in routes_called
        run_call = next(
            call for call in mock_global_bus.request.await_args_list
            if call.args[0] == GlobalRoutes.ALICE_RUN_AGENT
        )
        assert isinstance(run_call.kwargs["agent_run_context"], AgentRunContext)

    @pytest.mark.asyncio
    async def test_chat_stream_emits_prelude_and_done(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        event_types = [e["event"] for e in events]
        assert "generation_id" in event_types
        assert "topic_info" in event_types
        assert "memory_refs" in event_types
        assert "token" in event_types
        assert "done" in event_types

    def test_cancel_generation_returns_false_when_unknown(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        assert svc.cancel_generation("gen-1") is False

    @pytest.mark.asyncio
    async def test_chat_stream_cleans_up_prepared_run_on_runtime_error(self, mock_global_bus):
        prepared = _make_prepared_run()

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                assert "agent_run_context" in kwargs
                raise RuntimeError("boom")
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        svc = ChatApplicationService(global_bus=mock_global_bus)
        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        assert events[-1]["event"] == "error"
        routes_called = [
            call.args[0] for call in mock_global_bus.request.await_args_list
        ]
        assert GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN in routes_called

    @pytest.mark.asyncio
    async def test_cancel_generation_sets_registered_cancel_event(self, mock_global_bus):
        observed_cancel_event = None
        svc = ChatApplicationService(global_bus=mock_global_bus)
        prepared = _make_prepared_run()
        chat_result = _make_chat_result()

        async def route_dispatch(route, *args, **kwargs):
            nonlocal observed_cancel_event
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                assert "agent_run_context" in kwargs
                observed_cancel_event = kwargs["cancel_event"]

                async def _stream():
                    yield {"event": "token", "data": {"content": "hi"}}
                    await asyncio.sleep(0)
                    yield {"event": "done", "data": chat_result.model_dump()}

                return _stream()
            if route == GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN:
                return None
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)
            if e["event"] == "generation_id":
                assert svc.cancel_generation(e["data"]["generation_id"]) is True

        assert observed_cancel_event is not None
        assert observed_cancel_event.is_set() is True
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["stopped"] is True


