"""ChatApplicationService / PassiveIngressService 委托测试"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from hivememory.core.models import Identity, OMNI_DOLL_PROFILE
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.core.protocol.models import (
    AnalyzeAndRetrieveResult,
    ChatResult,
    EyeGazeResult,
    RetrievalResponse,
)
from hivememory.patchouli.models import (
    FinalizeContext,
    PreparedAgentRun,
    StreamPrelude,
)
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive import PassiveIngressEvent
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.contracts.routes import GlobalRoutes
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
        identity=identity,
        agent_id="omni_doll",
        topic_id="topic_1",
        user_message="hi",
        messages=[{"role": "user", "content": "hi"}],
        agent_profile=OMNI_DOLL_PROFILE,
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_snapshot={},
            memory_refs=[],
        ),
        finalize_context=FinalizeContext(
            gaze_result=gaze_result,
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
        ),
        generation_options=None,
    )
    defaults.update(overrides)
    return PreparedAgentRun(**defaults)


def _make_chat_result() -> ChatResult:
    return ChatResult(
        final_text="hello!",
        mtp_iterations=0,
        total_iterations=1,
        mtp_commands_executed=[],
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
