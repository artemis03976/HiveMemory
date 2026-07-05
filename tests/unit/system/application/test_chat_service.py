"""ChatApplicationService / PassiveIngressService 委托测试"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from types import SimpleNamespace
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
    AgentRunStatus,
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
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.gateway.commands import (
    CommandExecutionResult,
    CommandExecutionStatus,
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.system.system import HiveMemorySystem
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from hivememory.system.runtime.scheduler.global_scheduler import GlobalMaintenanceScheduler
from hivememory.system.runtime.control import ChatGenerationRun


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


def _make_command_gaze_result(raw_query: str, command) -> EyeGazeResult:
    identity = Identity(user_id="u1", agent_id="omni_doll")
    return EyeGazeResult(
        intent=GatewayIntent.SYSTEM,
        rewritten_query=raw_query,
        search_keywords=[],
        worth_saving=False,
        raw_query=raw_query,
        identity=identity,
        target_topic="NEW_TOPIC",
        command=command,
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
            return []
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
    async def test_chat_system_command_short_circuits_prepare_run_finalize(self, mock_global_bus):
        registry = create_builtin_command_registry()
        command_gaze = AsyncMock(
            return_value=_make_command_gaze_result("/help", registry.match("/help"))
        )
        svc = ChatApplicationService(
            global_bus=mock_global_bus,
            command_gaze=command_gaze,
            command_dispatcher=SystemCommandDispatcher(registry),
        )

        result = await svc.chat(user_message="/help", user_id="u1")

        assert isinstance(result, CommandExecutionResult)
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.command_id == "system.help"
        mock_global_bus.request.assert_not_awaited()
        command_gaze.assert_awaited_once()
        assert command_gaze.await_args.kwargs["topic_snapshots"] == []
        assert command_gaze.await_args.kwargs["identity"].user_id == "u1"

    @pytest.mark.asyncio
    async def test_chat_stream_system_command_emits_command_result_then_done(self, mock_global_bus):
        registry = create_builtin_command_registry()
        command_gaze = AsyncMock(
            return_value=_make_command_gaze_result("/clear", registry.match("/clear"))
        )
        svc = ChatApplicationService(
            global_bus=mock_global_bus,
            command_gaze=command_gaze,
            command_dispatcher=SystemCommandDispatcher(registry),
        )

        events = []
        async for e in svc.chat_stream(user_message="/clear", user_id="u1"):
            events.append(e)

        assert [e["event"] for e in events] == ["generation_id", "command_result", "done"]
        assert events[1]["data"]["command_id"] == "system.clear"
        assert events[1]["data"]["client_action"] == {"type": "clear_chat"}
        assert events[2]["data"]["status"] == "completed"
        assert events[2]["data"]["command_id"] == "system.clear"
        mock_global_bus.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chat_stream_non_command_skips_pre_gaze_and_keeps_normal_sequence(self, mock_global_bus):
        registry = create_builtin_command_registry()
        command_gaze = AsyncMock()
        svc = ChatApplicationService(
            global_bus=mock_global_bus,
            command_gaze=command_gaze,
            command_dispatcher=SystemCommandDispatcher(registry),
        )

        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        event_types = [e["event"] for e in events]
        assert event_types[:4] == ["generation_id", "topic_info", "memory_refs", "token"]
        assert event_types[-1] == "done"
        command_gaze.assert_not_awaited()
        routes_called = [call.args[0] for call in mock_global_bus.request.await_args_list]
        assert GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN in routes_called
        assert GlobalRoutes.ALICE_RUN_AGENT_STREAM in routes_called

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
        assert "cancel_event" in run_call.kwargs

    @pytest.mark.asyncio
    async def test_chat_stream_emits_prelude_and_done(self, mock_global_bus):
        recorder = RecordingRuntimeEventSink()
        svc = ChatApplicationService(global_bus=mock_global_bus, runtime_events=recorder)
        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        event_types = [e["event"] for e in events]
        assert "generation_id" in event_types
        assert "topic_info" in event_types
        assert "memory_refs" in event_types
        assert "token" in event_types
        assert "done" in event_types
        runtime_event_types = [event.event_type for event in recorder.events]
        assert RuntimeEventType.CHAT_RUN_CREATED in runtime_event_types
        assert RuntimeEventType.CHAT_RUN_COMPLETED in runtime_event_types

    @pytest.mark.asyncio
    async def test_chat_stream_uses_supplied_generation_id(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        events = []
        async for e in svc.chat_stream(
            user_message="hi",
            user_id="u1",
            generation_id="gen-supplied",
        ):
            events.append(e)

        generation_id_event = next(e for e in events if e["event"] == "generation_id")
        done_event = next(e for e in events if e["event"] == "done")
        assert generation_id_event["data"]["generation_id"] == "gen-supplied"
        assert done_event["data"]["generation_id"] == "gen-supplied"

    @pytest.mark.asyncio
    async def test_chat_stream_emits_finalizing_before_done_with_memory_task_ids(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        prepared = _make_prepared_run()
        chat_result = _make_chat_result()
        final_topic = SimpleNamespace(
            model_dump=lambda mode="json": {
                "topic_id": "topic_1",
                "topic_title": "Topic 1",
                "block_count": 1,
                "total_tokens": 42,
            }
        )

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                async def _stream():
                    yield {"event": "done", "data": chat_result.model_dump()}

                return _stream()
            if route == GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN:
                return [SimpleNamespace(task_id="memtask_1")]
            if route == GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE:
                assert kwargs["identity"].user_id == "u1"
                assert kwargs["include_empty"] is True
                return [final_topic]
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        event_types = [e["event"] for e in events]
        run_status_index = event_types.index("run_status")
        done_index = event_types.index("done")
        assert run_status_index < done_index
        assert events[run_status_index]["data"]["status"] == "finalizing"
        assert events[run_status_index]["data"]["generation_id"]
        assert events[done_index]["data"]["generation_id"] == events[run_status_index]["data"]["generation_id"]
        assert events[done_index]["data"]["memory_task_ids"] == ["memtask_1"]
        assert events[done_index]["data"]["pool_topics"] == [
            {
                "topic_id": "topic_1",
                "topic_title": "Topic 1",
                "block_count": 1,
                "total_tokens": 42,
            }
        ]

    def test_cancel_generation_returns_false_when_unknown(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        assert svc.cancel_generation("gen-1").cancelled is False

    def test_cancel_generation_preserves_first_reason(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        svc._registry.register(ChatGenerationRun(generation_id="gen-1"))
        first = svc.cancel_generation("gen-1", reason="client_disconnected")
        second = svc.cancel_generation("gen-1", reason="stream_closed")

        assert first.reason == "client_disconnected"
        assert second.reason == "client_disconnected"

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
    async def test_chat_stream_close_emits_cancelled_and_cleans_up(self, mock_global_bus):
        recorder = RecordingRuntimeEventSink()
        prepared = _make_prepared_run()
        stream_closed = False

        async def route_dispatch(route, *args, **kwargs):
            nonlocal stream_closed
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                async def _stream():
                    nonlocal stream_closed
                    try:
                        yield {"event": "token", "data": {"content": "hi"}}
                        await asyncio.Event().wait()
                    finally:
                        stream_closed = True

                return _stream()
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        svc = ChatApplicationService(global_bus=mock_global_bus, runtime_events=recorder)
        stream = svc.chat_stream(user_message="hi", user_id="u1")

        assert (await stream.__anext__())["event"] == "generation_id"
        assert (await stream.__anext__())["event"] == "topic_info"
        assert (await stream.__anext__())["event"] == "memory_refs"
        assert (await stream.__anext__())["event"] == "token"

        await stream.aclose()

        runtime_events = [event for event in recorder.events]
        assert runtime_events[-1].event_type == RuntimeEventType.CHAT_RUN_CANCELLED
        assert runtime_events[-1].status == "cancelled"
        assert runtime_events[-1].reason == "stream_closed"
        assert stream_closed is True
        routes_called = [
            call.args[0] for call in mock_global_bus.request.await_args_list
        ]
        assert GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN in routes_called

    @pytest.mark.asyncio
    async def test_chat_stream_close_preserves_existing_cancel_reason(self, mock_global_bus):
        recorder = RecordingRuntimeEventSink()
        prepared = _make_prepared_run()

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                async def _stream():
                    yield {"event": "token", "data": {"content": "hi"}}
                    await asyncio.Event().wait()

                return _stream()
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        svc = ChatApplicationService(global_bus=mock_global_bus, runtime_events=recorder)
        stream = svc.chat_stream(user_message="hi", user_id="u1", generation_id="gen-1")

        assert (await stream.__anext__())["event"] == "generation_id"
        assert (await stream.__anext__())["event"] == "topic_info"
        assert (await stream.__anext__())["event"] == "memory_refs"
        assert (await stream.__anext__())["event"] == "token"

        svc.cancel_generation("gen-1", reason="client_disconnected")
        await stream.aclose()

        runtime_events = [event for event in recorder.events]
        assert runtime_events[-1].event_type == RuntimeEventType.CHAT_RUN_CANCELLED
        assert runtime_events[-1].reason == "client_disconnected"
        assert runtime_events[-1].data["close_reason"] == "client_disconnected"

    @pytest.mark.asyncio
    async def test_chat_stream_close_after_task_driven_next_event_does_not_cross_context_fail(
        self,
        mock_global_bus,
    ):
        prepared = _make_prepared_run()

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                async def _stream():
                    yield {"event": "token", "data": {"content": "hi"}}
                    await asyncio.Event().wait()

                return _stream()
            if route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
                return True
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        svc = ChatApplicationService(global_bus=mock_global_bus)
        stream = svc.chat_stream(user_message="hi", user_id="u1")

        assert (await asyncio.create_task(stream.__anext__()))["event"] == "generation_id"
        assert (await asyncio.create_task(stream.__anext__()))["event"] == "topic_info"
        assert (await asyncio.create_task(stream.__anext__()))["event"] == "memory_refs"
        assert (await asyncio.create_task(stream.__anext__()))["event"] == "token"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_before_streaming_returns_cancelled_done(self, mock_global_bus):
        """취消在 ALICE stream 调用前触发 → 提前 return，done.status=cancelled。"""
        recorder = RecordingRuntimeEventSink()
        svc = ChatApplicationService(global_bus=mock_global_bus, runtime_events=recorder)
        prepared = _make_prepared_run()

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            raise AssertionError(f"Unexpected route: {route}")

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)
            if e["event"] == "generation_id":
                svc.cancel_generation(e["data"]["generation_id"])

        done = events[-1]
        assert done["event"] == "done"
        assert done["data"]["status"] == "cancelled"
        assert done["data"]["stopped"] is True
        assert done["data"]["memory_task_ids"] == []
        runtime_event_types = [event.event_type for event in recorder.events]
        assert RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED in runtime_event_types
        assert RuntimeEventType.CHAT_RUN_CANCELLED in runtime_event_types

    @pytest.mark.asyncio
    async def test_cancel_event_propagated_to_alice_during_stream(self, mock_global_bus):
        """cancel_event 被正确传入 Alice；stream 内取消后 done.status=cancelled。"""
        observed_cancel_event = None
        svc = ChatApplicationService(global_bus=mock_global_bus)
        prepared = _make_prepared_run()
        chat_result = _make_chat_result()
        # loop_result 携带 cancelled 状态，模拟 Alice 内部响应了 cancel_event
        cancelled_result = AgentRunResult(
            **{
                **chat_result.model_dump(),
                "status": AgentRunStatus.CANCELLED,
            }
        )

        async def route_dispatch(route, *args, **kwargs):
            nonlocal observed_cancel_event
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
                observed_cancel_event = kwargs["cancel_event"]

                async def _stream():
                    yield {"event": "token", "data": {"content": "hi"}}
                    yield {"event": "done", "data": cancelled_result.model_dump()}

                return _stream()
            return None

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        events = []
        async for e in svc.chat_stream(user_message="hi", user_id="u1"):
            events.append(e)

        assert observed_cancel_event is not None
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["status"] == "cancelled"


    @pytest.mark.asyncio
    async def test_chat_cancel_after_prepare_skips_non_streaming_run(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        prepared = _make_prepared_run()

        async def route_dispatch(route, *args, **kwargs):
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                svc.cancel_generation("gen-nonstream")
                return prepared
            raise AssertionError(f"Unexpected route: {route}")

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        result = await svc.chat(
            user_message="hi",
            user_id="u1",
            generation_id="gen-nonstream",
        )

        assert result.status == AgentRunStatus.CANCELLED.value
        routes_called = [call.args[0] for call in mock_global_bus.request.await_args_list]
        assert routes_called == [GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN]

    @pytest.mark.asyncio
    async def test_chat_cancel_during_non_streaming_run_skips_finalize(self, mock_global_bus):
        svc = ChatApplicationService(global_bus=mock_global_bus)
        prepared = _make_prepared_run()
        observed_cancel_event = None

        async def route_dispatch(route, *args, **kwargs):
            nonlocal observed_cancel_event
            if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
                return prepared
            if route == GlobalRoutes.ALICE_RUN_AGENT:
                observed_cancel_event = kwargs["cancel_event"]
                cancel_result = svc.cancel_generation("gen-nonstream")
                assert cancel_result.cancelled is True
                return AgentRunResult(
                    final_text="partial",
                    status=AgentRunStatus.CANCELLED,
                )
            raise AssertionError(f"Unexpected route: {route}")

        mock_global_bus.request = AsyncMock(side_effect=route_dispatch)

        result = await svc.chat(
            user_message="hi",
            user_id="u1",
            generation_id="gen-nonstream",
        )

        assert observed_cancel_event is not None
        assert observed_cancel_event.is_set()
        assert result.status == AgentRunStatus.CANCELLED.value
        assert result.final_text == "partial"
        routes_called = [call.args[0] for call in mock_global_bus.request.await_args_list]
        assert GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN not in routes_called
