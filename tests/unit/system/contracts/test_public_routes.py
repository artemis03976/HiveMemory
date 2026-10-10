"""公开路由注册/卸载测试 — 验证 System 门面在生命周期中正确管理全局总线路由。"""

import dataclasses
import inspect
import typing
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.alice.application.agent_run_service import AgentRunService
from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.alice.system import AliceSystem
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.core.contracts.events import GlobalEvents
from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
)
from hivememory.core.protocol.models import AgentRunContext, InteractionPayload
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.service import PatchouliService
from hivememory.workspace.contracts import CPUInputManifest
from tests.helpers.memory import make_memory_metadata
from tests.helpers.operations import OperationsHarness

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
        self.operation_entry = OperationsHarness().entry
        self.config = MagicMock()
        self.config.koakuma = MagicMock()
        self.config.koakuma.enabled = False
        self.config.llm = MagicMock()
        self.config.llm.worker = MagicMock()

    @pytest.mark.asyncio
    async def test_start_registers_public_routes_on_global_bus(self):
        """Alice 只挂载一条统一执行路由，已删除的流式路由名不再注册。"""
        system = AliceSystem(
            config=self.config, global_bus=self.global_bus, operation_entry=self.operation_entry
        )
        await system.start()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT in routes
        assert "alice.public.run_agent_stream" not in routes

    @pytest.mark.asyncio
    async def test_stop_removes_public_routes_from_global_bus(self):
        system = AliceSystem(
            config=self.config, global_bus=self.global_bus, operation_entry=self.operation_entry
        )
        await system.start()
        await system.stop()

        routes = self.global_bus.list_routes()
        assert AliceRoutes.RUN_AGENT not in routes

    @pytest.mark.asyncio
    async def test_request_through_global_bus_reaches_handler(self):
        system = AliceSystem(
            config=self.config, global_bus=self.global_bus, operation_entry=self.operation_entry
        )
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
    async def test_stream_mode_returns_async_generator(self):
        system = AliceSystem(
            config=self.config, global_bus=self.global_bus, operation_entry=self.operation_entry
        )

        async def _stream(**kwargs):
            yield {"event": "token"}
            yield {"event": "done"}

        system._service.run_agent = _stream
        await system.start()

        stream = await self.global_bus.request(
            AliceRoutes.RUN_AGENT,
            messages=[],
            identity="id",
            stream=True,
        )

        events = []
        async for event in stream:
            events.append(event)

        assert [e["event"] for e in events] == ["token", "done"]

    @pytest.mark.asyncio
    async def test_no_global_bus_skips_public_routes(self):
        system = AliceSystem(
            config=self.config, global_bus=None, operation_entry=self.operation_entry
        )
        await system.start()
        await system.stop()

    @pytest.mark.asyncio
    async def test_alice_does_not_subscribe_to_intent_settlement_events(self):
        """结算投影只由 workspace 登记订阅，Alice 生命周期不注册订阅者。"""
        system = AliceSystem(
            config=self.config, global_bus=self.global_bus, operation_entry=self.operation_entry
        )
        await system.start()
        assert GlobalEvents.PENDING_ATOM_SETTLED not in self.global_bus.list_events()
        assert GlobalEvents.PENDING_ATOM_CANCELLED not in self.global_bus.list_events()
        await system.stop()


# ========== Patchouli（轻量级 — 完整集成在 test_bootstrap 中测试） ==========


class TestChatHandoffContractShapes:
    """prepare 拆分后的交接契约形状（PreparedAgentRun 与公开路由签名）。"""

    def test_prepared_agent_run_no_longer_carries_cpu_side_payload(self):
        """PreparedAgentRun 只承载 Topic 与检索：不再有 Profile/租借/编译文本，也不回传用户消息与 Gateway 决定。"""
        fields = {field.name for field in dataclasses.fields(PreparedAgentRun)}
        for removed in (
            "agent_run_context",
            "stream_prelude",
            "attachment_leases",
            "generation_options",
            "agent_profile",
            "user_message",
            "gateway_decision",
            "identity_scope",
        ):
            assert removed not in fields
        for kept in (
            "belong_to",
            "interaction_id",
            "topic_id",
            "is_new_topic",
            "retrieval_result",
            "storage_available",
        ):
            assert kept in fields

    def test_patchouli_chat_route_signages_no_longer_expose_agent_run_models(self):
        """Patchouli chat 路由签名不再出现 AgentRunContext/AgentRunResult 或附件参数。"""
        prepare_hints = typing.get_type_hints(PatchouliService.prepare_agent_run)
        assert AgentRunContext not in prepare_hints.values()
        assert prepare_hints["return"] is PreparedAgentRun
        prepare_params = inspect.signature(PatchouliService.prepare_agent_run).parameters
        for removed in ("user_message", "generation_options", "selected_attachments"):
            assert removed not in prepare_params

        finalize_hints = typing.get_type_hints(PatchouliService.finalize_agent_run)
        assert finalize_hints["payload"] is InteractionPayload
        finalize_params = inspect.signature(PatchouliService.finalize_agent_run).parameters
        assert set(finalize_params) == {"self", "prepared_run", "payload", "identity_scope"}

        cleanup_params = inspect.signature(
            PatchouliService.cleanup_prepared_agent_run,
        ).parameters
        assert set(cleanup_params) == {"self", "prepared_run", "identity_scope"}

    def test_alice_run_routes_receive_cpu_input_manifest(self):
        """Alice 统一执行入口以 CPUInputManifest 为输入；AgentRunContext 仅内部使用。"""
        hints = typing.get_type_hints(AgentRunService.run_agent)
        assert hints["input_manifest"] is CPUInputManifest
        params = inspect.signature(AgentRunService.run_agent).parameters
        assert "agent_run_context" not in params
        assert "process_id" not in params
        # 统一入口以 stream 参数控制是否流式，不再有两个入口
        assert params["stream"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["stream"].default is True
        assert not hasattr(AgentRunService, "run_agent_stream")


class TestPatchouliPublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()

    @pytest.mark.asyncio
    async def test_public_route_constants_are_consistent(self):
        assert PatchouliRoutes.MEMORY_RETRIEVE == "patchouli.public.memory.retrieve"
        assert (
            PatchouliRoutes.MEMORY_RETRIEVE_BY_ALIASES
            == "patchouli.public.memory.retrieve_by_aliases"
        )
        assert PatchouliRoutes.MEMORY_TASK_LIST == "patchouli.public.memory_task.list"
        assert PatchouliRoutes.MEMORY_TASK_GET == "patchouli.public.memory_task.get"
        assert PatchouliRoutes.MEMORY_TASK_CANCEL == "patchouli.public.memory_task.cancel"
        assert PatchouliRoutes.MEMORY_READ == "patchouli.public.memory.read"
        assert PatchouliRoutes.INTERACTION_SUBMIT == "patchouli.public.interaction.submit"
        assert PatchouliRoutes.MEMORY_INTENT_SUBMIT == "patchouli.public.memory_intent.submit"
        assert PatchouliRoutes.PREPARE_AGENT_RUN == "patchouli.public.prepare_agent_run"
        assert PatchouliRoutes.FINALIZE_AGENT_RUN == "patchouli.public.finalize_agent_run"
        assert (
            PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN
            == "patchouli.public.cleanup_prepared_agent_run"
        )
        assert PatchouliRoutes.TOPIC_GET_DATA == "patchouli.public.topic.get_data"
        assert PatchouliRoutes.EVICT_TOPIC == "patchouli.public.evict_topic"
        assert PatchouliRoutes.RECORD_MEMORY_CITATION == "patchouli.public.record_memory_citation"
        assert PatchouliRoutes.WARMUP_MODELS == "patchouli.public.models.warmup"
        assert PatchouliRoutes.MODELS_READY == "patchouli.public.models.ready"
        assert AliceRoutes.RUN_AGENT == "alice.public.run_agent"

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
            interactions=MagicMock(),
            memory_intents=MagicMock(),
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
