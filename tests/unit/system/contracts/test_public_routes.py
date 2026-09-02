"""公开路由注册/卸载测试 — 验证 System 门面在生命周期中正确管理全局总线路由。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.system.contracts.events import GlobalEvents
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_identity_scope, make_runtime_scope

# ========== Alice ==========


def _make_memory(alias: str, content: str) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(user_id="test_user", source_agent_id="test_agent"),
        index=IndexLayer(
            title="Test Memory",
            summary="A test memory for public route behavior",
            tags=["test"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content=content),
    )


class TestAlicePublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()
        self.config = MagicMock()
        self.config.koakuma = MagicMock()
        self.config.koakuma.enabled = False
        self.config.llm = MagicMock()
        self.config.llm.worker = MagicMock()

    @pytest.mark.asyncio
    async def test_start_registers_public_routes_on_global_bus(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT in routes
        assert AliceRoutes.RUN_AGENT_STREAM in routes

    @pytest.mark.asyncio
    async def test_stop_removes_public_routes_from_global_bus(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()
        await system.stop()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT not in routes
        assert AliceRoutes.RUN_AGENT_STREAM not in routes

    @pytest.mark.asyncio
    async def test_request_through_global_bus_reaches_handler(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        received = []

        async def fake_run_agent(*, messages, identity):
            received.append((messages, identity))
            return "agent_result"

        system._service.run_agent = fake_run_agent
        await system.start()

        result = await self.global_bus.request(
            AliceRoutes.RUN_AGENT,
            messages=[],
            identity="id",
        )

        assert result == "agent_result"
        assert received == [([], "id")]

    @pytest.mark.asyncio
    async def test_stream_route_returns_async_generator(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)

        async def _stream(**kwargs):
            yield {"event": "token"}
            yield {"event": "done"}

        system._service.run_agent_stream = _stream
        await system.start()

        stream = await self.global_bus.request(
            AliceRoutes.RUN_AGENT_STREAM,
            messages=[],
            identity="id",
        )

        events = []
        async for event in stream:
            events.append(event)

        assert [e["event"] for e in events] == ["token", "done"]

    @pytest.mark.asyncio
    async def test_no_global_bus_skips_public_routes(self):
        system = AliceSystem(config=self.config, global_bus=None)
        await system.start()
        await system.stop()

    @pytest.mark.asyncio
    async def test_alice_local_bus_bridges_patchouli_memory_routes(self):
        received = []

        async def retrieve(*, request):
            received.append(("retrieve", request))
            return "retrieved"

        async def retrieve_by_aliases(*, aliases, identity_scope):
            received.append(("aliases", aliases, identity_scope))
            return "aliases"

        async def get_agent_profile(alias, *, identity_scope):
            received.append(("profile", alias, identity_scope))
            return "profile"

        async def record_citation(*, memory_id, source):
            received.append(("citation", memory_id, source))
            return "citation"

        self.global_bus.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, retrieve)
        self.global_bus.register(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            retrieve_by_aliases,
        )
        self.global_bus.register(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            get_agent_profile,
        )
        self.global_bus.register(
            GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
            record_citation,
        )

        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()
        identity_scope = make_identity_scope()

        result = await system.runtime.local_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE,
            request="request",
        )
        aliases_result = await system.runtime.local_bus.request(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            aliases=["a"],
            identity_scope=identity_scope,
        )
        profile_result = await system.runtime.local_bus.request(
            GlobalRoutes.PATCHOULI_GET_AGENT_PROFILE,
            "coder_doll",
            identity_scope=identity_scope,
        )
        citation_result = await system.runtime.local_bus.request(
            GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
            memory_id="mid",
            source="mtp.read",
        )

        assert result == "retrieved"
        assert aliases_result == "aliases"
        assert profile_result == "profile"
        assert citation_result == "citation"
        assert received == [
            ("retrieve", "request"),
            ("aliases", ["a"], identity_scope),
            ("profile", "coder_doll", identity_scope),
            ("citation", "mid", "mtp.read"),
        ]

    @pytest.mark.asyncio
    async def test_alice_unmount_unsubscribes_settlement_event(self):
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()

        assert GlobalEvents.PENDING_ATOM_SETTLED in self.global_bus.list_events()
        assert GlobalEvents.PENDING_ATOM_CANCELLED in self.global_bus.list_events()

        await system.stop()

        assert GlobalEvents.PENDING_ATOM_SETTLED not in self.global_bus.list_events()
        assert GlobalEvents.PENDING_ATOM_CANCELLED not in self.global_bus.list_events()

    @pytest.mark.asyncio
    async def test_cancelled_event_marks_alice_pending_atom_cancelled(self):
        from hivememory.core.models import Identity
        from hivememory.core.models.pending import PendingAtomStatus

        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()
        atom = system.runtime._pending_runtime.register_write(
            content="draft",
            title="Draft",
            reason=None,
            identity=Identity(user_id="test_user", agent_id="test_agent"),
            runtime_scope=make_runtime_scope(run_id="run-1"),
        )

        await self.global_bus.publish(
            GlobalEvents.PENDING_ATOM_CANCELLED,
            pending_alias=atom.pending_alias,
        )

        assert atom.status == PendingAtomStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_settlement_refreshes_alice_l1_atom_cache(self):
        """结算事件以原 scope 查询资源 owner，再刷新共享 L1 cache。"""
        from hivememory.core.models import Identity

        stale_atom = _make_memory("fact_canonical", "stale content")
        fresh_atom = _make_memory("fact_canonical", "fresh content")
        refresh_requests = []

        async def retrieve_by_aliases(*, aliases, identity_scope):
            refresh_requests.append((aliases, identity_scope))
            return SimpleNamespace(memories=[fresh_atom])

        self.global_bus.register(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            retrieve_by_aliases,
        )
        system = AliceSystem(config=self.config, global_bus=self.global_bus)
        await system.start()
        identity = Identity(user_id="test_user", agent_id="test_agent")
        identity_scope = make_identity_scope(actor_identity=identity)
        pending_runtime = system.runtime.alias_resolver.pending_runtime
        pending = pending_runtime.register_write(
            content="draft",
            title="Draft",
            reason=None,
            identity=identity,
            runtime_scope=make_runtime_scope(actor_identity=identity, run_id="run-1"),
        )
        pending_runtime.start_materializing(pending.pending_alias)
        system.runtime.atom_cache.ingest_atom(stale_atom)

        settlement = PendingAtomSettlement(
            pending_alias=pending.pending_alias,
            intent_id=pending.intent_id,
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_canonical",
            canonical_uuid=str(fresh_atom.id),
        )

        await self.global_bus.publish(
            GlobalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

        assert refresh_requests == [(["fact_canonical"], identity_scope)]
        assert system.runtime.atom_cache.get_atom_by_alias("fact_canonical") is fresh_atom
        assert (
            system.runtime.atom_cache.get_atom_by_uuid(
                str(stale_atom.id),
            )
            is None
        )


# ========== Patchouli (lightweight — full integration tested in test_bootstrap) ==========


class TestPatchouliPublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()

    @pytest.mark.asyncio
    async def test_public_route_constants_are_consistent(self):
        assert PatchouliRoutes.MEMORY_RETRIEVE == "patchouli.public.memory.retrieve"
        assert PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES == "patchouli.public.memory.retrieve_by_aliases"
        assert PatchouliRoutes.MEMORY_TASK_LIST == "patchouli.public.memory_task.list"
        assert PatchouliRoutes.MEMORY_TASK_GET == "patchouli.public.memory_task.get"
        assert PatchouliRoutes.MEMORY_TASK_CANCEL == "patchouli.public.memory_task.cancel"
        assert PatchouliRoutes.PREPARE_AGENT_RUN == "patchouli.public.prepare_agent_run"
        assert PatchouliRoutes.FINALIZE_AGENT_RUN == "patchouli.public.finalize_agent_run"
        assert PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN == "patchouli.public.cleanup_prepared_agent_run"
        assert PatchouliRoutes.TOPIC_GET_DATA == "patchouli.public.topic.get_data"
        assert PatchouliRoutes.EVICT_TOPIC == "patchouli.public.evict_topic"
        assert PatchouliRoutes.RECORD_MEMORY_CITATION == "patchouli.public.record_memory_citation"
        assert PatchouliRoutes.WARMUP_MODELS == "patchouli.public.models.warmup"
        assert PatchouliRoutes.MODELS_READY == "patchouli.public.models.ready"
        assert AliceRoutes.RUN_AGENT == "alice.public.run_agent"
        assert AliceRoutes.RUN_AGENT_STREAM == "alice.public.run_agent_stream"

    @pytest.mark.asyncio
    async def test_patchouli_public_routes_register_and_unregister(self):
        bridge = self._make_bridge()

        bridge.mount()

        routes = self.global_bus.list_routes()
        assert "patchouli.public.submit_interaction" not in routes
        assert PatchouliRoutes.FINALIZE_AGENT_RUN in routes
        assert PatchouliRoutes.TOPIC_GET_DATA in routes
        assert PatchouliRoutes.EVICT_TOPIC in routes
        assert PatchouliRoutes.MEMORY_TASK_LIST in routes
        assert PatchouliRoutes.MEMORY_TASK_GET in routes
        assert PatchouliRoutes.MEMORY_TASK_CANCEL in routes
        assert PatchouliRoutes.RECORD_MEMORY_CITATION in routes
        assert PatchouliRoutes.WARMUP_MODELS in routes
        assert PatchouliRoutes.MODELS_READY in routes

        ready = await self.global_bus.request(PatchouliRoutes.MODELS_READY)
        assert ready is True
        tasks = await self.global_bus.request(PatchouliRoutes.MEMORY_TASK_LIST)
        assert tasks == ["task"]

        bridge.unmount()

        routes = self.global_bus.list_routes()
        assert PatchouliRoutes.FINALIZE_AGENT_RUN not in routes
        assert PatchouliRoutes.TOPIC_GET_DATA not in routes
        assert PatchouliRoutes.EVICT_TOPIC not in routes
        assert PatchouliRoutes.MEMORY_TASK_LIST not in routes
        assert PatchouliRoutes.MEMORY_TASK_GET not in routes
        assert PatchouliRoutes.MEMORY_TASK_CANCEL not in routes
        assert PatchouliRoutes.RECORD_MEMORY_CITATION not in routes
        assert PatchouliRoutes.WARMUP_MODELS not in routes
        assert PatchouliRoutes.MODELS_READY not in routes

    @pytest.mark.asyncio
    async def test_patchouli_local_settlement_event_bridges_to_global_bus(self):
        bridge = self._make_bridge()
        subscriber = AsyncMock()
        self.global_bus.subscribe(GlobalEvents.PENDING_ATOM_SETTLED, subscriber)
        settlement = object()

        bridge.mount()
        await bridge._test_local_bus.publish(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

        subscriber.assert_awaited_once_with(settlement=settlement)

        subscriber.reset_mock()
        bridge.unmount()
        await bridge._test_local_bus.publish(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

        subscriber.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_patchouli_local_cancelled_event_bridges_to_global_bus(self):
        bridge = self._make_bridge()
        subscriber = AsyncMock()
        self.global_bus.subscribe(GlobalEvents.PENDING_ATOM_CANCELLED, subscriber)

        bridge.mount()
        await bridge._test_local_bus.publish(
            PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
            pending_alias="draft_cancelled",
        )

        subscriber.assert_awaited_once_with(pending_alias="draft_cancelled")

    @pytest.mark.asyncio
    async def test_patchouli_bridge_keeps_pending_atom_internal_to_function_bus(self):
        bridge = self._make_bridge()
        subscriber = AsyncMock()
        self.global_bus.subscribe(GlobalEvents.PENDING_ATOM_SETTLED, subscriber)
        settlement = PendingAtomSettlement(
            pending_alias="draft_memory_1234",
            intent_id="intent_1234",
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_canonical",
            canonical_uuid="atom-uuid-1",
        )

        bridge.mount()
        await bridge._test_local_bus.publish(
            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
            settlement=settlement,
        )

        subscriber.assert_awaited_once_with(settlement=settlement)

    def _make_bridge(self):
        service = MagicMock()
        service.prepare_agent_run = AsyncMock()
        service.finalize_agent_run = AsyncMock()
        service.cleanup_prepared_agent_run = AsyncMock()
        service.record_memory_citation = AsyncMock()

        local_bus = PatchouliBus()

        memory_management_service = MagicMock()
        memory_management_service.create_memory = AsyncMock()
        memory_management_service.list_memories = AsyncMock()
        memory_management_service.get_memory = AsyncMock()
        memory_management_service.update_memory = AsyncMock()
        memory_management_service.delete_memory = AsyncMock()
        memory_management_service.record_feedback = AsyncMock()
        memory_management_service.retrieve = AsyncMock()
        memory_management_service.retrieve_by_aliases = AsyncMock()

        memory_task_management_service = MagicMock()
        memory_task_management_service.list_memory_tasks = AsyncMock(return_value=["task"])
        memory_task_management_service.get_memory_task = AsyncMock()
        memory_task_management_service.cancel_memory_task = AsyncMock()

        agent_profile_management_service = MagicMock()
        agent_profile_management_service.create_agent_profile = AsyncMock()
        agent_profile_management_service.list_agent_profiles = AsyncMock()
        agent_profile_management_service.get_agent_profile = AsyncMock()

        topic_management_service = MagicMock()
        topic_management_service.list_active_topics = AsyncMock()
        topic_management_service.get_topic_data = AsyncMock()
        topic_management_service.settle_topic = AsyncMock()
        topic_management_service.evict_topic = AsyncMock()

        model_readiness_service = MagicMock()
        model_readiness_service.warmup_models = AsyncMock()
        model_readiness_service.is_models_ready = AsyncMock(return_value=True)

        public_api = PatchouliPublicApi(
            chat=service,
            memory=memory_management_service,
            memory_tasks=memory_task_management_service,
            agent_profiles=agent_profile_management_service,
            topics=topic_management_service,
            readiness=model_readiness_service,
        )
        bridge = PatchouliBridge(
            local_bus=local_bus,
            public_api=public_api,
            global_bus=self.global_bus,
        )
        bridge._test_local_bus = local_bus
        return bridge
