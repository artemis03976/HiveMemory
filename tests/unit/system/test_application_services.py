"""ChatApplicationService / PassiveIngressService 委托测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from hivememory.core.models import Identity
from hivememory.engines.gateway.models import GatewayIntent
from hivememory.patchouli.protocol.models import (
    AnalyzeAndRetrieveResult,
    EyeGazeResult,
    KernelHotResult,
)
from hivememory.system.application.chat_service import ChatApplicationService
from hivememory.system.application.passive import PassiveIngressEvent
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler


@pytest.fixture
def mock_patchouli_service():
    service = MagicMock()
    service.chat = AsyncMock(return_value="chat_result")
    service.chat_stream = MagicMock()
    service.cancel_generation = MagicMock(return_value=True)
    return service


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
    hot_result = KernelHotResult(
        intent="RAG",
        rewritten="resolved query",
        keywords=["resolved"],
        worth_saving=worth_saving,
        rendered_memory_context=memory,
    )
    return AnalyzeAndRetrieveResult(
        gaze_result=gaze_result,
        hot_result=hot_result,
    )


class TestChatApplicationService:
    @pytest.mark.asyncio
    async def test_chat_passes_all_args(self, mock_patchouli_service):
        svc = ChatApplicationService(patchouli_service=mock_patchouli_service)
        result = await svc.chat(
            user_message="hi",
            user_id="u1",
            agent_id="agent_x",
            session_id="s1",
            enable_memory_retrieval=False,
            generation_options={"max_tokens": 100},
        )
        mock_patchouli_service.chat.assert_called_once_with(
            user_message="hi",
            user_id="u1",
            agent_id="agent_x",
            session_id="s1",
            enable_memory_retrieval=False,
            generation_options={"max_tokens": 100},
        )
        assert result == "chat_result"

    @pytest.mark.asyncio
    async def test_chat_stream_yields_events(self, mock_patchouli_service):
        async def fake_stream(**kwargs):
            yield {"event": "token", "data": {"content": "hi"}}
            yield {"event": "done", "data": {}}

        mock_patchouli_service.chat_stream = fake_stream
        svc = ChatApplicationService(patchouli_service=mock_patchouli_service)
        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)
        assert len(events) == 2
        assert events[0]["event"] == "token"

    def test_cancel_generation(self, mock_patchouli_service):
        svc = ChatApplicationService(patchouli_service=mock_patchouli_service)
        mock_patchouli_service.cancel_generation.return_value = False
        assert svc.cancel_generation("gen-1") is False
        mock_patchouli_service.cancel_generation.assert_called_once_with("gen-1")


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
    async def test_flush_observer_session(self, bus, passive_config, scheduler):
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
        result = await svc.flush_observer_session(user_id="u1", agent_id="a", session_id="s")
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
