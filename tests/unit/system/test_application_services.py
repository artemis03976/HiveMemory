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
    ChatResult,
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


def _make_chat_result() -> ChatResult:
    return ChatResult(
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


class TestApiApplicationServices:
    def test_services_keep_config_reference(self, mock_global_bus, passive_config):
        memory_service = MemoryApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )
        agent_service = AgentApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )
        topic_service = TopicApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )

        assert memory_service.config is passive_config
        assert agent_service.config is passive_config
        assert topic_service.config is passive_config

    def test_hivememory_system_build_exposes_api_services(self, passive_config):
        with (
            patch("hivememory.system.system.PatchouliSystem"),
            patch("hivememory.system.system.AliceSystem"),
        ):
            system = HiveMemorySystem.build(config=passive_config)

        assert isinstance(system.memory_service, MemoryApplicationService)
        assert isinstance(system.agent_service, AgentApplicationService)
        assert isinstance(system.topic_service, TopicApplicationService)
        assert system.memory_service.config is passive_config
        assert system.agent_service.config is passive_config
        assert system.topic_service.config is passive_config

    def test_server_deps_return_api_services(self, passive_config):
        from hivememory.server import deps

        previous_system = deps._system
        try:
            with (
                patch("hivememory.system.system.PatchouliSystem"),
                patch("hivememory.system.system.AliceSystem"),
            ):
                system = HiveMemorySystem.build(config=passive_config)
            deps._system = system

            assert deps.get_memory_service() is system.memory_service
            assert deps.get_agent_service() is system.agent_service
            assert deps.get_topic_service() is system.topic_service
        finally:
            deps._system = previous_system


class TestMemoryApplicationService:
    @pytest.fixture
    def service(self, mock_global_bus, passive_config):
        return MemoryApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_create_memory_uses_public_route(self, service, mock_global_bus):
        created = _make_memory_atom(title="Created memory")
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = created

        atom = await service.create_memory(
            title="Created memory",
            summary="A sufficiently long memory summary",
            content="Created memory content",
            memory_type="FACT",
            tags=["created", "ui"],
            alias="created-memory",
        )

        mock_global_bus.request.assert_awaited_once()
        route, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_MEMORY_CREATE
        assert payload.meta.source_agent_id == "ui"
        assert payload.meta.user_id == "default"
        assert payload.index.memory_type == MemoryType.FACT
        assert payload.index.alias == "created-memory"
        assert atom is created

    @pytest.mark.asyncio
    async def test_get_memory_not_found_raises_domain_error(self, service, mock_global_bus):
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = None

        with pytest.raises(MemoryNotFoundError):
            await service.get_memory(uuid4())

    @pytest.mark.asyncio
    async def test_record_feedback_without_lifecycle_raises_domain_error(
        self,
        service,
        mock_global_bus,
    ):
        mock_global_bus.request.side_effect = RuntimeError(
            "Memory lifecycle engine is unavailable"
        )

        with pytest.raises(MemoryLifecycleUnavailableError):
            await service.record_feedback(uuid4(), positive=True, source="ui.memory_ref")


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


class TestAgentApplicationService:
    @pytest.fixture
    def service(self, mock_global_bus, passive_config):
        return AgentApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_create_agent_profile_uses_public_route(self, service, mock_global_bus):
        created = _make_memory_atom(title="Worker")
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = created

        atom = await service.create_agent_profile(
            title="Worker",
            alias="worker",
            summary="",
            content="persona",
            tags=["agent"],
            agent_config={"allowed_mtp_verbs": ["SEARCH"]},
        )

        mock_global_bus.request.assert_awaited_once()
        route, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE
        assert payload.index.memory_type == MemoryType.AGENT_PROFILE
        assert payload.index.summary == "Worker agent profile"
        assert payload.index.alias == "worker"
        assert payload.payload.content == "persona"
        assert payload.payload.artifacts.agent_config == {"allowed_mtp_verbs": ["SEARCH"]}
        assert atom is created

    @pytest.mark.asyncio
    async def test_list_agent_profiles_uses_public_route(self, service, mock_global_bus):
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = []

        assert await service.list_agent_profiles() == []
        mock_global_bus.request.assert_awaited_once_with(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            limit=100,
        )


class TestAgentProfileManagementService:
    def test_create_agent_profile_writes_storage(self):
        storage = MagicMock()
        service = AgentProfileManagementService(storage=storage)
        atom = _make_memory_atom(title="Worker")

        result = asyncio.run(service.create_agent_profile(atom))

        assert result is atom
        storage.upsert_memory.assert_called_once_with(atom)

    def test_list_agent_profiles_uses_agent_profile_filter(self):
        storage = MagicMock()
        service = AgentProfileManagementService(storage=storage)
        storage.get_all_memories.return_value = []

        assert asyncio.run(service.list_agent_profiles()) == []
        storage.get_all_memories.assert_called_once_with(
            filters={"index.memory_type": "AGENT_PROFILE"},
            limit=100,
        )


class TestTopicApplicationService:
    @pytest.fixture
    def librarian_core(self):
        librarian = MagicMock()
        librarian.perception_layer.buffer_manager.pop_buffer.return_value = object()
        return librarian

    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def service(self, bus, passive_config, librarian_core):
        return TopicApplicationService(
            global_bus=bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_list_active_topics_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value=["snapshot"])
        bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, handler)

        assert await service.list_active_topics(user_id="u1") == ["snapshot"]
        handler.assert_awaited_once()
        identity = handler.await_args.kwargs["identity"]
        assert identity.user_id == "u1"

    @pytest.mark.asyncio
    async def test_archive_topic_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value={"success": True, "topic_id": "t1"})
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_ARCHIVE_TOPIC, handler)

        result = await service.archive_topic(topic_id="t1")

        assert result == {"success": True, "topic_id": "t1"}
        handler.assert_awaited_once_with(topic_id="t1")

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value={"success": True, "message": "话题 t1 已删除"})
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)

        result = await service.evict_topic(topic_id="t1")

        assert result == {"success": True, "message": "话题 t1 已删除"}
        handler.assert_awaited_once_with(topic_id="t1")


class TestTopicManagementService:
    @pytest.fixture
    def librarian_core(self):
        librarian = MagicMock()
        librarian.perception_layer.buffer_manager.pop_buffer.return_value = object()
        return librarian

    def test_list_active_topics_uses_librarian_core(self, librarian_core):
        identity = Identity(user_id="u1")
        librarian_core.get_active_topics_snapshots.return_value = ["snapshot"]
        service = TopicManagementService(librarian_core=librarian_core)

        assert asyncio.run(service.list_active_topics(identity=identity)) == ["snapshot"]
        librarian_core.get_active_topics_snapshots.assert_called_once_with(identity)

    def test_evict_topic_uses_buffer_manager(self, librarian_core):
        service = TopicManagementService(librarian_core=librarian_core)

        result = asyncio.run(service.evict_topic(topic_id="t1"))

        assert result == {"success": True, "message": "话题 t1 已删除"}
        librarian_core.perception_layer.buffer_manager.pop_buffer.assert_called_once_with("t1")


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
