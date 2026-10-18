"""公开路由注册/卸载测试 — 验证 System 门面在生命周期中正确管理全局总线路由。"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

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
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes
from hivememory.patchouli.runtime.bridge import PatchouliBridge, PatchouliPublicApi
from hivememory.patchouli.runtime.bus import PatchouliBus
from tests.helpers.memory import make_memory_metadata


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


# ========== Patchouli（轻量级 — 完整集成在 test_bootstrap 中测试） ==========


class TestPatchouliPublicRoutes:

    def setup_method(self):
        self.global_bus = GlobalSystemBus()

    @pytest.mark.asyncio
    async def test_patchouli_public_routes_register_and_unregister(self):
        bridge = self._make_bridge()

        bridge.mount()

        routes = self.global_bus.list_routes()
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
